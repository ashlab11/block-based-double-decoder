#!/usr/bin/env python3
"""
Post-hoc SFT + eval driver.

For every ``*_pct100.pt`` under ``--search-dir`` (skipping any whose
``*_postsft_pct100.pt`` sibling already exists when ``--skip-existing`` is
set), this script sequentially:

  1. Loads the pretrain checkpoint and rebuilds the matching model.
  2. SFTs for ``--sft-tokens`` on UltraChat using the same per-model-type
     collators / batch-sizing rules as ``parallel_scaling.py --run-sft``.
  3. Saves the SFT'd weights as ``<basename>_postsft_pct100.pt`` next to
     the original checkpoint.
  4. Runs the ``--eval-suite`` benchmark group on the in-memory SFT'd
     model and writes a ``<basename>_postsft_pct100.json`` sidecar with
     SFT loss + eval results.
  5. Optionally uploads both files (weights + sidecar JSON) to
     ``--hf-repo`` via ``training.hf_hub.upload_checkpoint``.

The script is restartable: passing ``--skip-existing`` makes it safe to
re-run after a pod restart — only the checkpoints that haven't yet been
SFT'd will be processed.

Example:
    python scripts/post_hoc_sft.py \\
        --search-dir checkpoints/ \\
        --sft-train-file data/SFT/ultrachat.jsonl \\
        --sft-tokens 50000000 \\
        --eval-suite paper \\
        --eval-max-examples 500 \\
        --skip-existing \\
        --hf-repo "$HF_REPO"

To target a subset (e.g. only the dd 50M runs):
    python scripts/post_hoc_sft.py --only-pattern '*dd_50M*' ...
"""

import sys
import os
import json
import time
import argparse
import traceback
from fnmatch import fnmatch
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exact pretrain helpers so SFT mechanics stay identical to the
# in-process --run-sft path. Any future change to sft_one_model / collators /
# build_model is automatically inherited here.
from scripts.parallel_scaling import (
    build_model,
    build_sft_collator,
    sft_one_model,
    detect_gpu_tflops,
    _load_tuned_lrs,
    _load_tuned_wds,
    SEQ_LEN,
)
from evals.run_evals import run_evals_on_model, resolve_eval_names
from training.hf_hub import upload_checkpoint, available as hf_available


def find_pretrain_checkpoints(search_dir):
    """Recursively find ``*_pct100.pt`` and exclude already-SFT'd siblings."""
    paths = sorted(Path(search_dir).rglob("*_pct100.pt"))
    return [p for p in paths if "_postsft_" not in p.name]


def postsft_path_for(ckpt_path):
    return ckpt_path.parent / ckpt_path.name.replace(
        "_pct100.pt", "_postsft_pct100.pt")


def load_pretrain_checkpoint(ckpt_path, device):
    """Reconstruct the model from a parallel_scaling.py-format checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    hparams = ckpt["hparams"]
    arch = {
        "dim": hparams["dim"],
        "num_encoder_layers": hparams["num_encoder_layers"],
        "num_decoder_layers": hparams["num_decoder_layers"],
    }
    model_type = ckpt["model_type"]
    vocab_size = ckpt["vocab_size"]

    # use_compile=False — torch.compile would re-trace on the first SFT
    # forward anyway, and skipping it makes per-checkpoint setup ~30s faster.
    model = build_model(model_type, arch, vocab_size, device, use_compile=False)

    # Strip the _orig_mod. prefix that torch.compile adds to state_dict keys
    # when models are saved post-compile. Pretrain checkpoints are saved from
    # the eager submodule, but be defensive in case a future caller saves the
    # compiled wrapper directly.
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    return model, arch, model_type, hparams, ckpt


def parse_run_id_from_filename(ckpt_path):
    """Recover (model_type, arch_label, tok_label) from filename if present.
    Filenames are ``{model_type}_{arch_label}_{tok_label}tok_pct100.pt``.
    Returns ``(model_type, arch_label, tok_label)`` or all-None on parse failure.
    """
    stem = ckpt_path.stem  # e.g. "dd_50M_6Btok_pct100"
    parts = stem.split("_")
    if len(parts) < 4 or not parts[-1].startswith("pct"):
        return None, None, None
    # parts: [model_type, arch_label, "{N}tok", "pct100"]
    model_type = parts[0]
    tok_part = parts[-2]  # "6Btok"
    arch_label = "_".join(parts[1:-2])
    tok_label = tok_part.removesuffix("tok") if tok_part.endswith("tok") else tok_part
    return model_type, arch_label, tok_label


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument("--search-dir", type=str, default="checkpoints/",
                   help="Root directory to recursively scan for *_pct100.pt")
    p.add_argument("--only-pattern", type=str, default=None,
                   help="fnmatch glob applied to checkpoint paths "
                        "(e.g. '*dd_50M*' or '*sub6B*')")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a ckpt if its _postsft_pct100.pt already exists. "
                        "Idempotent — safe to re-run after pod restarts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned work and exit without loading models.")

    # SFT data + hyperparameters (defaults match parallel_scaling.py --run-sft)
    p.add_argument("--sft-train-file", type=str,
                   default="data/SFT/ultrachat.jsonl")
    p.add_argument("--sft-tokens", type=int, default=50_000_000,
                   help="Approximate target token budget for SFT.")
    p.add_argument("--sft-lr", type=float, default=2e-5)
    p.add_argument("--sft-grad-accum", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8,
                   help="SFT micro-batch size; auto-halved for dd/sed (which "
                        "carry both encoder and decoder activations through "
                        "backward, doubling per-batch headroom).")

    # Eval
    p.add_argument("--eval-suite", type=str, default="paper",
                   help="Group name (e.g. 'paper', 'paper_full', 'quick') or "
                        "comma-separated list of eval names.")
    p.add_argument("--eval-max-examples", type=int, default=500)
    p.add_argument("--eval-file", type=str,
                   default="data/Pretrain/slimpajama_eval_packed.jsonl")
    p.add_argument("--tokenizer-file", type=str,
                   default="tokenizer/tokenizer_32k.json")

    # HF persistence
    p.add_argument("--hf-repo", type=str, default=None,
                   help="If set, uploads each *_postsft_pct100.pt and its JSON "
                        "sidecar to this repo. Use the same repo as your "
                        "pretrain --hf-repo so all artifacts live together.")
    p.add_argument("--hf-private", action="store_true")

    args = p.parse_args()

    # Read tuned LRs/WDs so any logging in sft_one_model that references them
    # has the right numbers (sft_one_model itself uses --sft-lr explicitly).
    _load_tuned_lrs()
    _load_tuned_wds()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_tflops = detect_gpu_tflops()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}  (peak BF16: {gpu_tflops:.0f} TFLOPS)")
    else:
        device = torch.device("cpu")
        gpu_tflops = 0.0
        print("WARN: no GPU detected; SFT will be unusably slow on CPU.")

    # ── HF preflight (fail fast on auth/typo before loading any models) ────
    upload_to_hf = bool(args.hf_repo)
    if upload_to_hf:
        if not hf_available():
            print("ERROR: --hf-repo set but huggingface_hub not installed. "
                  "Run: pip install huggingface_hub")
            sys.exit(1)
        from training.hf_hub import verify_repo
        try:
            verify_repo(args.hf_repo, private=args.hf_private, repo_type="model")
        except Exception as e:
            print(f"ERROR: HF preflight failed for {args.hf_repo}: {e}")
            sys.exit(1)

    # ── Discover work ──────────────────────────────────────────────────────
    ckpts = find_pretrain_checkpoints(args.search_dir)
    if args.only_pattern:
        ckpts = [p for p in ckpts if fnmatch(str(p), args.only_pattern)]
    if args.skip_existing:
        ckpts = [p for p in ckpts if not postsft_path_for(p).exists()]

    if not ckpts:
        print(f"[plan] no checkpoints to process under {args.search_dir} "
              f"(pattern={args.only_pattern}, skip_existing={args.skip_existing})")
        return

    print(f"[plan] {len(ckpts)} checkpoint(s) to process:")
    for cp in ckpts:
        out = postsft_path_for(cp)
        print(f"  - {cp}  →  {out.name}")

    if args.dry_run:
        print("[dry-run] exiting without running.")
        return

    # ── Tokenizer + SFT data (load once, reuse for every checkpoint) ───────
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_file)
    print(f"[data] loading SFT data from {args.sft_train_file}...")
    sft_train_ds = load_dataset("json", data_files=args.sft_train_file,
                                split="train")
    print(f"[data] loaded {len(sft_train_ds):,} SFT examples")

    eval_names = resolve_eval_names(args.eval_suite)
    print(f"[eval] suite '{args.eval_suite}' → {len(eval_names)} eval(s): {eval_names}")

    # ── Per-checkpoint loop ────────────────────────────────────────────────
    n_ok = 0
    n_fail = 0
    for i, ckpt_path in enumerate(ckpts):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(ckpts)}] {ckpt_path}")
        print(f"{'='*70}")
        t_total = time.time()
        model = None  # set inside try, freed in finally

        try:
            print(f"  [load] reading {ckpt_path}...")
            model, arch, model_type, hparams, raw_ckpt = load_pretrain_checkpoint(
                ckpt_path, device)
            ne = (sum(p.numel() for p in model.parameters())
                  - model.embedding.weight.numel())
            arch_label = hparams.get("arch_label")
            tok_label = parse_run_id_from_filename(ckpt_path)[2]
            display = f"{model_type}_{arch_label or 'unknown'}"
            print(f"  [load] {display}  dim={arch['dim']}  "
                  f"enc/dec={arch['num_encoder_layers']}/{arch['num_decoder_layers']}  "
                  f"non_emb={ne:,}  vocab={raw_ckpt['vocab_size']:,}  "
                  f"pretrain_tok={tok_label}")

            # ── SFT setup ──────────────────────────────────────────────────
            # Mirror parallel_scaling.py:1169-1174 — enc-dec architectures
            # carry both encoder and decoder activations through backward
            # during SFT, so halve bs and double ga to keep effective batch
            # constant while staying inside the OOM ceiling.
            if model_type in ("dd", "sed"):
                sft_bs = max(1, args.batch_size // 2)
                sft_ga = max(1, args.sft_grad_accum * 2)
            else:
                sft_bs = args.batch_size
                sft_ga = args.sft_grad_accum

            sft_collator = build_sft_collator(model_type, tokenizer, SEQ_LEN)
            sft_loader = DataLoader(
                sft_train_ds, batch_size=sft_bs, shuffle=True,
                collate_fn=sft_collator, drop_last=True,
                num_workers=2, pin_memory=True,
            )
            sft_total_micro = max(sft_ga,
                                  args.sft_tokens // (sft_bs * SEQ_LEN))
            sft_total_micro = min(sft_total_micro, len(sft_loader))
            sft_steps = sft_total_micro // sft_ga

            # sft_one_model expects a "trainer" dict with these exact keys.
            trainer = {
                "model": model,
                "arch": arch,
                "model_type": model_type,
                "ne": ne,
                "display": display,
            }

            print(f"  [sft] {display}: ~{args.sft_tokens/1e6:.0f}M tokens  "
                  f"({sft_total_micro} micro / {sft_steps} steps  "
                  f"bs={sft_bs} ga={sft_ga} lr={args.sft_lr})")

            sft_t0 = time.time()
            avg_sft_loss, sft_train_time, sft_n_steps, sft_mfu = sft_one_model(
                trainer, sft_loader, device, sft_total_micro,
                sft_ga, args.sft_lr,
                gpu_tflops=gpu_tflops,
                tokens_per_micro=sft_bs * SEQ_LEN,
            )
            print(f"  [sft] done: loss={avg_sft_loss:.3f}  "
                  f"time={sft_train_time:.0f}s  MFU={sft_mfu:.1f}%")

            # ── Save SFT'd weights ─────────────────────────────────────────
            postsft_path = postsft_path_for(ckpt_path)
            sft_hparams = dict(hparams)
            sft_hparams.update({
                "phase": "sft",
                "sft_tokens_target": args.sft_tokens,
                "sft_tokens_actual": sft_total_micro * sft_bs * SEQ_LEN,
                "sft_lr": args.sft_lr,
                "sft_batch_size": sft_bs,
                "sft_grad_accum": sft_ga,
                "sft_train_file": args.sft_train_file,
                "source_pretrain_ckpt": ckpt_path.name,
            })
            eager = getattr(model, "_orig_mod", model)
            torch.save({
                "state_dict": eager.state_dict(),
                "hparams": sft_hparams,
                "vocab_size": eager.embedding.weight.shape[0],
                "model_type": model_type,
                "phase": "sft",
                "sft_final_loss": float(avg_sft_loss),
                "sft_train_time_sec": round(sft_train_time, 1),
                "sft_total_steps": sft_n_steps,
            }, postsft_path)
            print(f"  [save] {postsft_path}")

            # ── Eval on SFT'd model ────────────────────────────────────────
            is_enc_dec = model_type in ("dd", "sed")
            eager.eval()
            print(f"  [eval] running {len(eval_names)} eval(s) on SFT'd "
                  f"{display} (max_examples={args.eval_max_examples})...")
            eval_t0 = time.time()
            try:
                bench_sft = run_evals_on_model(
                    model=eager, tokenizer=tokenizer, device=device,
                    is_enc_dec=is_enc_dec, eval_names=eval_names,
                    max_examples=args.eval_max_examples,
                    batch_size=args.batch_size,
                    eval_file=args.eval_file,
                )
                eval_time = time.time() - eval_t0
                print(f"  [eval] done in {eval_time:.0f}s")
                eval_error = None
            except Exception as e:
                eval_time = time.time() - eval_t0
                print(f"  [eval] FAILED after {eval_time:.0f}s: {e}")
                traceback.print_exc()
                bench_sft = {}
                eval_error = str(e)
            finally:
                eager.train()

            # ── JSON sidecar ───────────────────────────────────────────────
            sidecar_path = postsft_path.with_suffix(".json")
            sidecar = {
                "source_pretrain_ckpt": ckpt_path.name,
                "source_pretrain_dir": str(ckpt_path.parent),
                "model_type": model_type,
                "arch_label": arch_label,
                "tok_label": tok_label,
                "arch": arch,
                "non_emb_params": ne,
                "vocab_size": raw_ckpt["vocab_size"],
                "phase": "sft",
                "sft_hparams": {
                    "lr": args.sft_lr,
                    "batch_size": sft_bs,
                    "grad_accum": sft_ga,
                    "tokens_target": args.sft_tokens,
                    "tokens_actual": sft_total_micro * sft_bs * SEQ_LEN,
                    "train_file": args.sft_train_file,
                },
                "sft_final_loss": float(avg_sft_loss),
                "sft_train_time_sec": round(sft_train_time, 1),
                "sft_total_steps": sft_n_steps,
                "sft_mfu_pct": round(sft_mfu, 2),
                "sft_evals": bench_sft,
                "sft_eval_time_sec": round(eval_time, 1),
                "sft_evals_error": eval_error,
                "eval_suite": args.eval_suite,
                "eval_max_examples": args.eval_max_examples,
                "wall_time_sec": round(time.time() - t_total, 1),
            }
            with open(sidecar_path, "w") as f:
                json.dump(sidecar, f, indent=2, default=str)
            print(f"  [save] {sidecar_path}")

            # ── HF upload ──────────────────────────────────────────────────
            if upload_to_hf:
                rel_dir = postsft_path.parent.name
                for f in (postsft_path, sidecar_path):
                    rel = f"postsft/{rel_dir}/{f.name}"
                    try:
                        upload_checkpoint(f, args.hf_repo, rel,
                                          repo_type="model")
                    except Exception as e:
                        print(f"  [hf-upload] FAILED for {f}: {e}  "
                              f"-- continuing")

            print(f"  [done] {display} in {time.time()-t_total:.0f}s")
            n_ok += 1

        except Exception as e:
            print(f"  [ERROR] failed processing {ckpt_path}: {e}")
            traceback.print_exc()
            print(f"  [ERROR] continuing to next checkpoint")
            n_fail += 1
        finally:
            # Free GPU memory before the next checkpoint loads. Without this,
            # the previous model's parameters + optimizer state stay resident
            # until Python gc runs, which on a 24 GB card is the difference
            # between fitting the next 100M model and OOMing.
            if model is not None:
                del model
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"[summary] {n_ok} succeeded, {n_fail} failed, "
          f"{len(ckpts) - n_ok - n_fail} skipped")
    print(f"{'='*70}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
