#!/usr/bin/env python3
"""Post-hoc prefixLM SFT + held-out loss eval driver.

For every ``*_pct100.pt`` under ``--search-dir`` (skipping ``*_postprefixlm_*``),
this script sequentially:

  1. Loads the pretrain checkpoint and rebuilds the matching model.
  2. SFTs for ``--sft-tokens`` on packed SlimPajama prefixLM data using:
       - dd, sed: collators/encoder_decoder/prefixlm.py:PrefixLMCollator
       - dec:     collators/decoder/prefixlm.py:DecoderPrefixLMCollator
                  (loss-only-on-suffix; standard causal attention)
  3. Saves SFT'd weights as ``<basename>_postprefixlm_pct100.pt``.
  4. Computes mean cross-entropy loss on the held-out prefixLM eval set.
  5. Writes ``<basename>_postprefixlm_pct100.json`` sidecar with SFT loss
     curve + held-out eval loss.
  6. Optionally uploads weights + JSON sidecar to ``--hf-repo``.

The point: bring all three architectures (dd/sed/dec) onto a single training
objective (prefixLM) so their held-out loss numbers are comparable on one axis.
The pretrain phase trained them on different objectives (NTP, NTP, span
corruption); this script equalizes them post-hoc.

Inputs (produced by data/retrieval_scripts/slimpajama_prefixlm.py --pack):
    data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl    (~500M tokens)
    data/Pretrain/slimpajama_prefixlm_eval_packed.jsonl   (~100M tokens)

Idempotent: --skip-existing skips checkpoints whose _postprefixlm sibling
already exists; safe to re-run after pod restarts.

Example (process all checkpoints under checkpoints/):
    python scripts/post_hoc_prefixlm.py \\
        --search-dir checkpoints/ \\
        --sft-tokens 50000000 \\
        --skip-existing \\
        --hf-repo "$HF_REPO"

Subset by glob:
    python scripts/post_hoc_prefixlm.py --only-pattern '*dd_50M*' ...
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from fnmatch import fnmatch
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the pretrain helpers so model construction, optimizer, and the
# train loop stay identical to parallel_scaling.py — any future change there
# is automatically inherited here.
from scripts.parallel_scaling import (
    build_model,
    sft_one_model,
    detect_gpu_tflops,
    _load_tuned_lrs,
    _load_tuned_wds,
    SEQ_LEN,
)
from training.hf_hub import upload_checkpoint, available as hf_available

from collators.encoder_decoder.prefixlm import PrefixLMCollator
from collators.decoder.prefixlm import DecoderPrefixLMCollator


# ── PrefixLM collator selector ─────────────────────────────────────────────

def build_prefixlm_collator(model_type, tokenizer, seq_len, seed):
    """Pick the right prefixLM collator for this model_type."""
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0
    common = dict(
        max_seq_len=seq_len,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        global_seed=seed,
    )
    if model_type == "dec":
        return DecoderPrefixLMCollator(**common)
    elif model_type in ("dd", "sed"):
        return PrefixLMCollator(**common)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── Discovery + checkpoint loading (mirror post_hoc_sft.py) ────────────────

def find_pretrain_checkpoints(search_dir):
    """Return all *_pct100.pt under search_dir, excluding any postprefixlm/postsft
    siblings (those are derived files, not pretrain checkpoints)."""
    paths = sorted(Path(search_dir).rglob("*_pct100.pt"))
    return [p for p in paths
            if "_postsft_" not in p.name and "_postprefixlm_" not in p.name]


def postprefixlm_path_for(ckpt_path):
    """Map foo_pct100.pt → foo_postprefixlm_pct100.pt (sibling in same dir)."""
    return ckpt_path.with_name(ckpt_path.name.replace(
        "_pct100.pt", "_postprefixlm_pct100.pt"))


def load_pretrain_checkpoint(ckpt_path, device):
    """Reconstruct the model from a parallel_scaling.py-format checkpoint.
    Returns (model, arch, model_type, hparams, raw_ckpt)."""
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    arch = raw["arch"]
    model_type = raw["model_type"]
    hparams = raw.get("hparams", {})
    vocab_size = raw["vocab_size"]
    model = build_model(model_type, arch, vocab_size, device, use_compile=False)
    state_dict = raw["state_dict"]
    # Strip torch.compile prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model, arch, model_type, hparams, raw


def parse_run_id_from_filename(ckpt_path):
    """Extract (model_type, arch, tok_label) from filename if encoded.
    Returns ('?', '?', '?') if parsing fails — tolerant on purpose."""
    parts = ckpt_path.stem.split("_")
    try:
        return parts[0], parts[1], parts[2].rstrip("tok")
    except IndexError:
        return "?", "?", "?"


# ── Held-out eval: mean prefixLM loss on the eval dataloader ────────────────

@torch.no_grad()
def eval_prefixlm_loss(model, loader, device, max_batches=None):
    """Compute token-weighted mean cross-entropy loss on the held-out set.
    Returns (mean_loss, n_batches, n_valid_tokens, elapsed_s).

    Uses the same collator-supplied label masking as training, so loss is
    accumulated only over suffix positions (prefix labels are -100)."""
    eager = getattr(model, "_orig_mod", model)
    eager.eval()

    total_loss_weighted = 0.0
    total_n_valid = 0
    n_batches = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = eager(**batch)
            loss = out["loss"]
        n_valid = int((batch["labels"] != -100).sum().item())
        total_loss_weighted += float(loss.item()) * n_valid
        total_n_valid += n_valid
        n_batches += 1

    mean_loss = total_loss_weighted / max(total_n_valid, 1)
    return mean_loss, n_batches, total_n_valid, time.time() - t0


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument("--search-dir", type=str, default="checkpoints/",
                   help="Root directory to recursively scan for *_pct100.pt")
    p.add_argument("--only-pattern", type=str, default=None,
                   help="fnmatch glob to filter checkpoint paths "
                        "(e.g. '*dd_50M*' or '*ben_sweep*')")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a ckpt if its _postprefixlm_pct100.pt sibling exists. "
                        "Safe to re-run after pod restarts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned work and exit without loading models.")

    # PrefixLM SFT data + hyperparameters
    p.add_argument("--sft-train-file", type=str,
                   default="data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl",
                   help="Packed prefixLM training data (produced by "
                        "data/retrieval_scripts/slimpajama_prefixlm.py --pack)")
    p.add_argument("--sft-eval-file", type=str,
                   default="data/Pretrain/slimpajama_prefixlm_eval_packed.jsonl",
                   help="Packed prefixLM held-out eval data")
    p.add_argument("--sft-tokens", type=int, default=50_000_000,
                   help="Approximate target token budget for SFT (default 50M).")
    p.add_argument("--sft-lr", type=float, default=2e-5)
    p.add_argument("--sft-grad-accum", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8,
                   help="SFT micro-batch size; auto-halved for dd/sed (which "
                        "carry both encoder and decoder activations through "
                        "backward, doubling per-batch memory headroom).")

    # Eval
    p.add_argument("--eval-batches", type=int, default=None,
                   help="Cap on eval batches (None = whole eval set, ~1-2 min)")
    p.add_argument("--eval-batch-size", type=int, default=16,
                   help="Eval batch size — can be larger than train since no backward")

    p.add_argument("--tokenizer-file", type=str,
                   default="tokenizer/tokenizer_32k.json")

    # HF persistence (defaults to the same repo as checkpoint storage)
    p.add_argument("--hf-repo", type=str, default=None,
                   help="If set, uploads each *_postprefixlm_pct100.pt and its JSON "
                        "sidecar to this repo. Use the same repo as your pretrain "
                        "--hf-repo so all artifacts live together.")
    p.add_argument("--hf-private", action="store_true")

    p.add_argument("--seed", type=int, default=42,
                   help="Global seed for deterministic SFT/eval data order.")

    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Read tuned LR/WD tables (same convention as parallel_scaling.py).
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
        ckpts = [p for p in ckpts if not postprefixlm_path_for(p).exists()]

    if not ckpts:
        print(f"[plan] no checkpoints to process under {args.search_dir} "
              f"(pattern={args.only_pattern}, skip_existing={args.skip_existing})")
        return

    print(f"[plan] {len(ckpts)} checkpoint(s) to process:")
    for cp in ckpts:
        out = postprefixlm_path_for(cp)
        print(f"  - {cp}  →  {out.name}")

    if args.dry_run:
        print("[dry-run] exiting without running.")
        return

    # ── Tokenizer + datasets (load once, reuse for every checkpoint) ──────
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_file)
    print(f"[data] loading SFT data from {args.sft_train_file}...")
    sft_train_ds = load_dataset("json", data_files=args.sft_train_file, split="train")
    print(f"[data] loaded {len(sft_train_ds):,} packed SFT sequences "
          f"(~{len(sft_train_ds) * SEQ_LEN / 1e6:.0f}M tokens)")
    print(f"[data] loading eval data from {args.sft_eval_file}...")
    eval_ds = load_dataset("json", data_files=args.sft_eval_file, split="train")
    print(f"[data] loaded {len(eval_ds):,} packed eval sequences "
          f"(~{len(eval_ds) * SEQ_LEN / 1e6:.0f}M tokens)")

    sft_train_ds = sft_train_ds.shuffle(seed=args.seed)
    # Eval not shuffled — deterministic ordering for reproducible loss numbers.

    # ── Per-checkpoint loop (sequential, OOM-safe via finally) ─────────────
    n_ok = 0
    n_fail = 0
    summary_rows = []
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

            # ── PrefixLM SFT setup ─────────────────────────────────────────
            # Halve bs / double ga for enc-dec to keep effective batch the
            # same while staying inside memory headroom (mirrors
            # parallel_scaling.py:1169-1174 and post_hoc_sft.py:418-423).
            if model_type in ("dd", "sed"):
                sft_bs = max(1, args.batch_size // 2)
                sft_ga = max(1, args.sft_grad_accum * 2)
            else:
                sft_bs = args.batch_size
                sft_ga = args.sft_grad_accum

            sft_collator = build_prefixlm_collator(
                model_type, tokenizer, SEQ_LEN, seed=args.seed)
            sft_loader = DataLoader(
                sft_train_ds, batch_size=sft_bs, shuffle=True,
                collate_fn=sft_collator, drop_last=True,
                num_workers=2, pin_memory=True,
            )
            sft_total_micro = max(sft_ga,
                                  args.sft_tokens // (sft_bs * SEQ_LEN))
            sft_total_micro = min(sft_total_micro, len(sft_loader))
            sft_steps = sft_total_micro // sft_ga

            print(f"  [sft] {display}: ~{args.sft_tokens/1e6:.0f}M tokens  "
                  f"({sft_total_micro} micro / {sft_steps} steps  "
                  f"bs={sft_bs} ga={sft_ga} lr={args.sft_lr})")

            sft_t0 = time.time()
            avg_sft_loss, sft_train_time, sft_n_steps, sft_mfu = sft_one_model(
                {"model": model, "arch": arch, "model_type": model_type,
                 "ne": ne, "display": display},
                sft_loader, device, sft_total_micro, sft_ga, args.sft_lr,
                gpu_tflops=gpu_tflops, tokens_per_micro=sft_bs * SEQ_LEN,
            )
            print(f"  [sft] done: train_loss={avg_sft_loss:.3f}  "
                  f"time={sft_train_time:.0f}s  MFU={sft_mfu:.1f}%")

            # ── Held-out eval (same collator, fresh DataLoader, no shuffle) ─
            # Eval batch can be larger since no backward — use args.eval_batch_size.
            # For enc-dec models we still halve to keep memory parity; eval
            # peaks lower than train but the safety margin matters here too.
            eval_bs = max(1, args.eval_batch_size // 2) if model_type in ("dd", "sed") \
                      else args.eval_batch_size
            eval_collator = build_prefixlm_collator(
                model_type, tokenizer, SEQ_LEN, seed=args.seed + 1)
            eval_loader = DataLoader(
                eval_ds, batch_size=eval_bs, shuffle=False,
                collate_fn=eval_collator, drop_last=True,
                num_workers=2, pin_memory=True,
            )
            print(f"  [eval] held-out prefixLM loss (bs={eval_bs}"
                  + (f", capped at {args.eval_batches} batches" if args.eval_batches else "")
                  + ")...")
            eval_loss, eval_n_batches, eval_n_valid, eval_elapsed = eval_prefixlm_loss(
                model, eval_loader, device, max_batches=args.eval_batches)
            print(f"  [eval] held-out loss = {eval_loss:.4f}  "
                  f"({eval_n_batches} batches, {eval_n_valid:,} suffix tokens, "
                  f"{eval_elapsed:.0f}s)")

            # ── Save SFT'd weights + JSON sidecar ──────────────────────────
            postpfx_path = postprefixlm_path_for(ckpt_path)
            sft_hparams = dict(hparams)
            sft_hparams.update({
                "phase": "post_hoc_prefixlm",
                "sft_objective": "prefixlm",
                "sft_tokens_target": args.sft_tokens,
                "sft_tokens_actual": sft_total_micro * sft_bs * SEQ_LEN,
                "sft_lr": args.sft_lr,
                "sft_batch_size": sft_bs,
                "sft_grad_accum": sft_ga,
                "sft_train_file": args.sft_train_file,
                "sft_eval_file": args.sft_eval_file,
                "source_pretrain_ckpt": ckpt_path.name,
            })
            eager = getattr(model, "_orig_mod", model)
            torch.save({
                "state_dict": eager.state_dict(),
                "arch": arch,
                "model_type": model_type,
                "hparams": sft_hparams,
                "vocab_size": raw_ckpt["vocab_size"],
            }, str(postpfx_path))
            print(f"  [save] weights → {postpfx_path}")

            sidecar = {
                "model_type": model_type,
                "arch": arch,
                "arch_label": arch_label,
                "non_emb_params": ne,
                "pretrain_ckpt": ckpt_path.name,
                "sft_objective": "prefixlm",
                "sft_train_loss": avg_sft_loss,
                "sft_train_time_s": sft_train_time,
                "sft_n_steps": sft_n_steps,
                "sft_mfu_pct": sft_mfu,
                "sft_tokens_target": args.sft_tokens,
                "sft_tokens_actual": sft_total_micro * sft_bs * SEQ_LEN,
                "sft_lr": args.sft_lr,
                "sft_batch_size": sft_bs,
                "sft_grad_accum": sft_ga,
                "eval_loss_held_out": eval_loss,
                "eval_n_batches": eval_n_batches,
                "eval_n_valid_tokens": eval_n_valid,
                "eval_elapsed_s": eval_elapsed,
                "wallclock_s": time.time() - t_total,
            }
            sidecar_path = postpfx_path.with_suffix(".json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2))
            print(f"  [save] sidecar → {sidecar_path}")

            # ── HF upload ──────────────────────────────────────────────────
            if upload_to_hf:
                for f in (postpfx_path, sidecar_path):
                    try:
                        upload_checkpoint(
                            ckpt_path=f,
                            repo_id=args.hf_repo,
                            path_in_repo=f"{model_type}/{arch_label or 'unknown'}/"
                                         f"{tok_label}tok/{f.name}",
                            private=args.hf_private,
                        )
                        print(f"  [hf-upload] ✓ {f.name}")
                    except Exception as e:
                        print(f"  [hf-upload] FAILED for {f}: {e}  "
                              "(weights still on local disk; manual retry possible)")

            n_ok += 1
            summary_rows.append({
                "ckpt": ckpt_path.name,
                "model_type": model_type,
                "arch_label": arch_label,
                "sft_train_loss": avg_sft_loss,
                "eval_loss_held_out": eval_loss,
                "wallclock_s": time.time() - t_total,
            })
            print(f"  [done] total wallclock {time.time() - t_total:.0f}s")

        except Exception as e:
            print(f"  [ERROR] failed processing {ckpt_path}: {e}")
            traceback.print_exc()
            print(f"  [ERROR] continuing to next checkpoint")
            n_fail += 1
        finally:
            # Free GPU memory before the next checkpoint loads. Without this
            # the previous model's params + optimizer state stay resident
            # until Python gc runs, which on a 24 GB card is the difference
            # between fitting the next 100M model and OOMing.
            if model is not None:
                del model
            torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[summary] {n_ok} succeeded, {n_fail} failed, "
          f"{len(ckpts) - n_ok - n_fail} skipped")
    print(f"{'='*70}")
    if summary_rows:
        print(f"\n  {'model_type':<12} {'arch':<10} {'sft_loss':>10} "
              f"{'eval_loss':>10} {'wallclock':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in summary_rows:
            print(f"  {r['model_type']:<12} {str(r['arch_label'] or '?'):<10} "
                  f"{r['sft_train_loss']:>10.3f} {r['eval_loss_held_out']:>10.4f} "
                  f"{r['wallclock_s']:>9.0f}s")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
