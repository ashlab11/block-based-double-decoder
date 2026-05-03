#!/usr/bin/env python3
"""μP base hparam sweep — single phase, full grid, multi-seed, mid-training coord check.

Replaces the old two-phase design (cheap LR sweep at base width, then a 3-LR
verification at 3 widths). The two-phase design produced misleadingly clean
"PASS"/"WARN" verdicts because Phase 2 only sampled `{best/2, best, best*2}`
around a single-seed Phase 1 winner, so any noise in Phase 1 cascaded into a
grid that couldn't resolve sub-grid-step shifts.

The new design runs the full Phase-1 LR grid at every width, with two seeds per
cell, at double the original Phase 1 token budget. A mid-training coord check
fires at 25%/50%/75% of training to catch residual-stream / attention-statistic
drift that the static checks in mup_full_check.py can't see (those only probe
init + 5 Adam steps).

Output:
  checkpoints/mup_base_sweep/results.json — all (arch, dim, lr, seed) cells
  configs/mup_tuned.json — per-arch base LR (argmin at base width, seed mean)

Usage:
    python scripts/mup_base_sweep.py --archs dd,sed,dec
    python scripts/mup_base_sweep.py --archs dd --lrs 1e-3,2e-3,4e-3 --seeds 0
    python scripts/mup_base_sweep.py --plot-only
    python scripts/mup_base_sweep.py --dry-run
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Doubled from the old Phase 1 default (50M) — the entire token budget that
# used to fund Phase 1 + Phase 2 now goes into a single, longer Phase 2 cell.
DEFAULT_TOKENS = 100_000_000
DEFAULT_DIMS = [64, 128, 256]
DEFAULT_LRS = [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2]
DEFAULT_SEEDS = [0, 1]
DEFAULT_COORD_FRACS = (0.25, 0.5, 0.75)
BASE_DIM = 64  # the dim whose argmin defines per-arch base_lr in mup_tuned.json

OUT_DIR = PROJECT_ROOT / "checkpoints" / "mup_base_sweep"
TUNED_PATH = PROJECT_ROOT / "configs" / "mup_tuned.json"


def _train_one(model_type, dim, enc, dec, tokens, lr, seed, run_tag,
               dry_run=False, use_compile=True):
    """Subprocess-isolated single training run. Process isolation flushes
    torch.compile caches and CUDA allocator state between runs."""
    if dry_run:
        return {"model_type": model_type, "dim": dim, "lr": lr, "seed": seed,
                "tokens": tokens, "final_eval_loss": None, "dry_run": True}

    result_path = OUT_DIR / run_tag / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--single-run",
           "--single-mt", model_type,
           "--single-dim", str(dim),
           "--single-enc", str(enc),
           "--single-dec", str(dec),
           "--single-lr", repr(lr),
           "--single-seed", str(seed),
           "--single-tokens", str(tokens),
           "--single-run-tag", run_tag,
           "--single-result-path", str(result_path)]
    if not use_compile:
        cmd.append("--no-compile")

    print(f"  → subprocess: {model_type} dim={dim} lr={lr:.2e} seed={seed} "
          f"tokens={tokens:,}")
    subprocess.run(cmd, check=True)
    return json.loads(result_path.read_text())


# ── Mid-training coord probe ────────────────────────────────────────────────

def _make_coord_callback(record, fractions, model_type, coord_batch):
    """Return a step_callback for train_one_budget that snapshots residual /
    attn / (DD only) gate stats at fractional progress points."""
    fired = set()

    def callback(trainers, step, total_steps):
        for f in fractions:
            target = max(1, int(f * total_steps))
            if step >= target and f not in fired:
                fired.add(f)
                for t in trainers:
                    snap = _coord_snapshot(t["model"], coord_batch, t["model_type"])
                    snap.update(step=step, fraction=f, total_steps=total_steps)
                    record.append(snap)
    return callback


def _coord_snapshot(model, batch, model_type):
    """One-batch probe: residual-stream RMS per layer, logits RMS, qk_std,
    softmax entropy, and (DD only) sigmoid-gate mean. Cheap; runs in eval
    mode under no_grad on a fixed batch."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    eager = getattr(model, "_orig_mod", model)
    was_training = eager.training
    eager.eval()

    layer_rms = {}
    hooks = []

    def _hook(name):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            layer_rms[name] = t.detach().float().pow(2).mean().sqrt().item()
        return fn

    enc_layers = getattr(eager, "encoder_layers", None)
    if enc_layers is not None:
        for i, lyr in enumerate(enc_layers):
            hooks.append(lyr.register_forward_hook(_hook(f"enc_{i}")))
    dec_layers = getattr(eager, "decoder_layers", None) or getattr(eager, "layers", None)
    if dec_layers is not None:
        prefix = "dec" if getattr(eager, "decoder_layers", None) is not None else "lyr"
        for i, lyr in enumerate(dec_layers):
            hooks.append(lyr.register_forward_hook(_hook(f"{prefix}_{i}")))

    qk_stds, entropies = [], []
    orig_sdpa = F.scaled_dot_product_attention

    def _sdpa_wrap(q, k, v, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, **kw):
        with torch.no_grad():
            head_dim = q.shape[-1]
            eff = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
            qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * eff
            if is_causal:
                Lq, Lk = qk.shape[-2], qk.shape[-1]
                m = torch.triu(torch.ones(Lq, Lk, device=qk.device,
                                          dtype=torch.bool), diagonal=1)
                qk = qk.masked_fill(m, float('-inf'))
            elif attn_mask is not None and attn_mask.dtype == torch.bool:
                qk = qk.masked_fill(~attn_mask, float('-inf'))
            elif attn_mask is not None:
                qk = qk + attn_mask
            finite = qk[qk.isfinite()]
            if finite.numel():
                qk_stds.append(finite.std().item())
                probs = qk.softmax(dim=-1)
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
                row_ok = qk.isfinite().any(-1)
                if row_ok.any():
                    entropies.append(ent[row_ok].mean().item())
        return orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, scale=scale, **kw)

    # DD-only: capture the combo gate by intercepting paired _attn_with_lse calls.
    # ComboAttention.forward calls _attn_with_lse twice per layer (dec, then enc);
    # we pair them by module id and compute sigmoid(dec_lse - enc_lse).mean().
    dec_w_means = []
    pending = {}
    orig_attn_with_lse = None
    if model_type == "dd":
        from components.attention import ComboAttention
        orig_attn_with_lse = ComboAttention._attn_with_lse

        def _attn_wrap(self, query, key, value, mask, default_causal):
            out, lse = orig_attn_with_lse(self, query, key, value, mask, default_causal)
            mid = id(self)
            if mid not in pending:
                pending[mid] = lse.detach()
            else:
                dec_lse = pending.pop(mid)
                with torch.no_grad():
                    dec_w_means.append(torch.sigmoid(dec_lse - lse.detach()).mean().item())
            return out, lse
        ComboAttention._attn_with_lse = _attn_wrap

    F.scaled_dot_product_attention = _sdpa_wrap

    needs_blocks = model_type in ("dd", "sed")
    embed_rms = logits_rms = loss_val = None
    try:
        with torch.no_grad():
            raw_emb = eager.embedding(batch["input_ids"])
            embed_rms = raw_emb.detach().float().pow(2).mean().sqrt().item()
            if needs_blocks:
                out = eager(**batch)
            else:
                out = eager(input_ids=batch["input_ids"], labels=batch["labels"])
            logits_rms = out["logits"].detach().float().pow(2).mean().sqrt().item()
            if out.get("loss") is not None:
                loss_val = float(out["loss"].item())
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        if orig_attn_with_lse is not None:
            from components.attention import ComboAttention
            ComboAttention._attn_with_lse = orig_attn_with_lse
        for h in hooks:
            h.remove()
        if was_training:
            eager.train()

    return {
        "model_type": model_type,
        "embed_rms": embed_rms,
        "logits_rms": logits_rms,
        "loss": loss_val,
        "layer_rms": layer_rms,
        "qk_std": float(np.mean(qk_stds)) if qk_stds else None,
        "softmax_entropy": float(np.mean(entropies)) if entropies else None,
        "dec_w_mean": float(np.mean(dec_w_means)) if dec_w_means else None,
    }


def _build_coord_batch(model_type, vocab_size, seq_len, device, seed):
    """Fixed batch the probe reuses at every fractional checkpoint. Same seed
    for every cell so coord-check trajectories are comparable across runs."""
    import torch
    g = torch.Generator(device='cpu').manual_seed(12345 + seed * 7)
    bs = 4
    input_ids = torch.randint(0, vocab_size, (bs, seq_len), generator=g).to(device)
    blocks = (torch.full((bs,), seq_len, device=device, dtype=torch.long)
              if model_type == "sed"
              else torch.tensor([seq_len // 4, seq_len // 2, 3 * seq_len // 4],
                                device=device))
    return {"input_ids": input_ids, "labels": input_ids.clone(),
            "blocks": blocks, "sft": False}


# ── Subprocess body ─────────────────────────────────────────────────────────

def _run_single_in_process(model_type, dim, enc, dec, tokens, lr, seed,
                           run_tag, result_path, use_compile=True):
    """Execute a single training cell. Called in a fresh subprocess."""
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import PreTrainedTokenizerFast

    import scripts.parallel_scaling as ps

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(PROJECT_ROOT / "tokenizer" / "tokenizer_32k.json"))
    vocab_size = tokenizer.vocab_size
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0

    ps.BASE_LR = lr
    ps.TUNED_LRS = {model_type: {"base_lr": lr}}

    arch = {"dim": dim, "num_encoder_layers": enc, "num_decoder_layers": dec}
    model = ps.build_model(model_type, arch, vocab_size, device,
                           use_compile=use_compile)
    ne = ps.non_emb_param_count(model)
    info = [{"name": run_tag, "model_type": model_type, "arch": arch,
             "model": model, "ne": ne,
             "needs_blocks": model_type in ("dd", "sed")}]

    bs, ga = 8, 4

    train_ds = load_dataset(
        "json",
        data_files=str(PROJECT_ROOT / "data/Pretrain/slimpajama_6b_packed.jsonl"),
        split="train", streaming=True)
    # Per-seed shuffle so seed=0 and seed=1 see different token orderings.
    train_ds = train_ds.shuffle(seed=seed, buffer_size=10_000)
    eval_ds = load_dataset(
        "json",
        data_files=str(PROJECT_ROOT / "data/Pretrain/slimpajama_6b_eval_packed.jsonl"),
        split="train")

    collator_seed = 42 + seed * 10_000
    loaders = {
        mt: DataLoader(train_ds, batch_size=bs,
                       collate_fn=ps.build_pretrain_collator(
                           mt, bos_id, eos_id, pad_id, ps.SEQ_LEN,
                           global_seed=collator_seed),
                       drop_last=True)
        for mt in ps.MODEL_TYPES
    }
    eval_loaders = {
        mt: DataLoader(eval_ds, batch_size=bs,
                       collate_fn=ps.build_pretrain_collator(
                           mt, bos_id, eos_id, pad_id, ps.SEQ_LEN,
                           global_seed=collator_seed),
                       drop_last=True)
        for mt in ps.MODEL_TYPES
    }

    gpu_tflops = ps.detect_gpu_tflops()

    coord_record = []
    coord_batch = _build_coord_batch(model_type, vocab_size, ps.SEQ_LEN, device,
                                     seed=0)
    coord_cb = _make_coord_callback(coord_record, DEFAULT_COORD_FRACS,
                                    model_type, coord_batch)

    out_subdir = OUT_DIR / run_tag
    results = ps.train_one_budget(
        info, tok_label="sweep", total_tokens=tokens,
        batch_size=bs, grad_accum=ga,
        train_loaders=loaders, eval_loaders=eval_loaders,
        device=device, gpu_tflops=gpu_tflops,
        eval_batches=10,
        output_dir=str(out_subdir),
        mid_eval_points=0,
        run_full_eval=False,
        tokenizer=tokenizer,
        run_sft=False,
        step_callback=coord_cb)

    # Final coord snapshot (fraction=1.0) so the curve includes end-state.
    coord_record.append({
        **_coord_snapshot(model, coord_batch, model_type),
        "step": -1, "fraction": 1.0, "total_steps": -1,
    })

    key = f"{model_type}_{run_tag}"
    run = results.get(key, {})

    result = {
        "model_type": model_type, "dim": dim, "enc": enc, "dec": dec,
        "lr": lr, "seed": seed, "tokens": tokens,
        "final_eval_loss": run.get("final_eval_loss"),
        "final_eval_ppl": run.get("final_eval_ppl"),
        "non_emb_params": run.get("non_emb_params"),
        "training_time_sec": run.get("training_time_sec"),
        "coord_curve": coord_record,
    }
    Path(result_path).write_text(json.dumps(result, indent=2))


# ── Sweep driver ────────────────────────────────────────────────────────────

def _sweep(archs, dims, lrs, seeds, enc, dec, tokens, dry_run=False,
           use_compile=True, existing_results=None):
    """Run the full grid: archs × dims × lrs × seeds. Skip cells already in
    `existing_results` so resume works."""
    done = {(r["model_type"], r["dim"], r["lr"], r.get("seed", 0))
            for r in (existing_results or [])
            if r.get("final_eval_loss") is not None}

    plan = [(mt, d, lr, s) for mt in archs for d in dims for lr in lrs
            for s in seeds if (mt, d, lr, s) not in done]
    skipped = (len(archs) * len(dims) * len(lrs) * len(seeds)) - len(plan)
    print(f"\n{'═'*60}\n  Full grid sweep\n"
          f"  archs={archs}  dims={dims}  lrs={lrs}  seeds={seeds}\n"
          f"  enc={enc} dec={dec} tokens={tokens:,}\n"
          f"  {len(plan)} runs to do  ({skipped} cached)  compile={use_compile}\n"
          f"{'═'*60}")
    results = []
    for i, (mt, d, lr, s) in enumerate(plan, 1):
        run_tag = f"sweep_{mt}_dim{d}_lr{lr:.0e}_s{s}"
        print(f"\n[{i}/{len(plan)}] {run_tag}")
        t0 = time.time()
        r = _train_one(mt, d, enc, dec, tokens, lr, s, run_tag,
                       dry_run=dry_run, use_compile=use_compile)
        r["run_tag"] = run_tag
        results.append(r)
        if not dry_run:
            print(f"  loss={r.get('final_eval_loss')}  "
                  f"({(time.time()-t0):.0f}s)")
    return results


def _best_lr_per_arch(all_results, base_dim):
    """Per arch, pick LR that minimizes seed-mean eval loss at base width."""
    by = {}
    for r in all_results:
        if r.get("final_eval_loss") is None:
            continue
        if r["dim"] != base_dim:
            continue
        by.setdefault((r["model_type"], r["lr"]), []).append(r["final_eval_loss"])

    by_arch = {}
    for (mt, lr), losses in by.items():
        by_arch.setdefault(mt, []).append((lr, sum(losses) / len(losses), len(losses)))

    out = {}
    for mt, rows in by_arch.items():
        rows.sort(key=lambda x: x[1])
        best_lr, best_mean, n = rows[0]
        out[mt] = {
            "base_lr": best_lr,
            "best_loss": best_mean,
            "n_seeds": n,
            "phase1_dim": base_dim,
            "lr_curve": [(lr, mean) for lr, mean, _ in sorted(rows, key=lambda x: x[0])],
        }
    return out


def _write_tuned_json(lrs_per_arch):
    if TUNED_PATH.exists():
        existing = json.loads(TUNED_PATH.read_text())
    else:
        existing = {}
    existing.update(lrs_per_arch)
    TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(json.dumps(existing, indent=2))
    print(f"\nWrote tuned LRs to: {TUNED_PATH}")
    for mt, info in lrs_per_arch.items():
        print(f"  {mt}: base_lr={info['base_lr']:.2e}  "
              f"(seed-mean loss={info['best_loss']:.4f}, n_seeds={info['n_seeds']})")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", default="dd,sed,dec",
                        help="Comma-separated archs (default: dd,sed,dec)")
    parser.add_argument("--dims", default=None,
                        help=f"Comma-separated widths (default: {DEFAULT_DIMS})")
    parser.add_argument("--lrs", default=None,
                        help="Comma-separated LRs "
                             f"(default: {','.join(f'{x:.0e}' for x in DEFAULT_LRS)})")
    parser.add_argument("--seeds", default=None,
                        help=f"Comma-separated seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"Tokens per run (default: {DEFAULT_TOKENS:,})")
    parser.add_argument("--enc", type=int, default=4,
                        help="Encoder layers (default 4)")
    parser.add_argument("--dec", type=int, default=2,
                        help="Decoder layers (default 2)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-render via mup_verdict.py from existing results.json")
    parser.add_argument("--no-compile", action="store_true")

    parser.add_argument("--single-run", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--single-mt", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-dim", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--single-enc", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--single-dec", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--single-lr", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--single-seed", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--single-tokens", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--single-run-tag", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-result-path", default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    use_compile = not args.no_compile

    if args.single_run:
        _run_single_in_process(
            model_type=args.single_mt, dim=args.single_dim,
            enc=args.single_enc, dec=args.single_dec,
            tokens=args.single_tokens, lr=args.single_lr,
            seed=args.single_seed,
            run_tag=args.single_run_tag,
            result_path=args.single_result_path,
            use_compile=use_compile)
        return

    archs = [a.strip() for a in args.archs.split(",")]
    dims = ([int(x) for x in args.dims.split(",")]
            if args.dims else list(DEFAULT_DIMS))
    lrs = ([float(x) for x in args.lrs.split(",")]
           if args.lrs else list(DEFAULT_LRS))
    seeds = ([int(x) for x in args.seeds.split(",")]
             if args.seeds else list(DEFAULT_SEEDS))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    if args.plot_only:
        if not results_path.exists():
            print(f"No results at {results_path} — run without --plot-only first.")
            sys.exit(1)
        subprocess.run([sys.executable,
                        str(Path(__file__).resolve().parent / "mup_verdict.py"),
                        "--results", str(results_path)], check=True)
        return

    all_results = []
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        print(f"Resuming from {len(all_results)} prior results in {results_path}")

    new_results = _sweep(archs, dims, lrs, seeds, args.enc, args.dec,
                         args.tokens, dry_run=args.dry_run,
                         use_compile=use_compile,
                         existing_results=all_results)
    all_results.extend(new_results)
    if not args.dry_run:
        results_path.write_text(json.dumps(all_results, indent=2))

    best = _best_lr_per_arch(all_results, base_dim=BASE_DIM)
    if best and not args.dry_run:
        _write_tuned_json(best)

    if not args.dry_run:
        subprocess.run([sys.executable,
                        str(Path(__file__).resolve().parent / "mup_verdict.py"),
                        "--results", str(results_path)], check=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
