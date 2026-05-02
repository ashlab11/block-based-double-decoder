#!/usr/bin/env python3
"""μP base hparam sweep — find per-arch optimal LR before the big runs.

Item #3 of the paper-prep plan. The big sweep in parallel_scaling.py uses a
single global BASE_LR=2e-3 for all three archs. μP says LR transfers across
widths *if* the base width is tuned, but it makes no claim across architectures
— DD's combo cross-attn, SED's bidirectional encoder, and DEC's pure causal
stack each have a different optimum. This script finds those optima.

Two phases:

  Phase 1 — LR sweep at base width (cheap, definitive)
    For each arch in {dd, sed, dec}, train at the smallest scale (default
    dim=64, ~0.5M params, 50M tokens) with 6–7 candidate LRs. Best LR per
    arch defines the base LR that gets baked into configs/mup_tuned.json.

  Phase 2 — LR transfer verification across widths (optional but useful)
    For each arch, train at 3 widths × 3 LRs near the Phase 1 optimum.
    Confirms the U-curve minimum lands at the same LR across widths — the
    signature of a working μP implementation. If transfer fails, that's a
    bug to fix before the big runs (cheap to find now, expensive after).

Output: configs/mup_tuned.json — read at startup by parallel_scaling.py to
override the per-arch base LR. Format:
    {"dd":  {"base_lr": 0.002, "best_loss": 5.83, "phase1_dim": 64, ...},
     "sed": {"base_lr": 0.001, ...},
     "dec": {"base_lr": 0.003, ...}}

Usage:
    # Phase 1 only (default; ~1 hour on one GPU):
    python scripts/mup_base_sweep.py --archs dd,sed,dec

    # Phase 1 + Phase 2 (verify width transfer; ~3 hours):
    python scripts/mup_base_sweep.py --archs dd,sed,dec --verify-transfer

    # Subset / debug:
    python scripts/mup_base_sweep.py --archs dd --lrs 1e-3,2e-3,4e-3 --tokens 10000000

    # Dry-run (planning only, no training):
    python scripts/mup_base_sweep.py --dry-run

    # Just plot existing results:
    python scripts/mup_base_sweep.py --plot-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 1 default constants — chosen so the sweep finishes in ~1 hour.
# 0.5M params × 50M tokens ≈ 100× Chinchilla-overtrained, which is fine for
# finding LR optima (the U-curve is visible long before convergence).
DEFAULT_PHASE1_DIM = 64           # = MUP_BASE_DIM in parallel_scaling
DEFAULT_PHASE1_TOKENS = 50_000_000
DEFAULT_LRS = [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2]

# Phase 2: verify transfer across these widths (must include base for a
# self-consistency check). 3 widths is enough to draw the LR-transfer curve
# without making the sweep prohibitively expensive.
DEFAULT_PHASE2_DIMS = [64, 128, 256]
PHASE2_LRS_NEAR_OPT = 3  # take optimum from phase 1 ± 1 step on each side

OUT_DIR = PROJECT_ROOT / "checkpoints" / "mup_base_sweep"
TUNED_PATH = PROJECT_ROOT / "configs" / "mup_tuned.json"


def _arch_to_dim_layers(arch_label):
    """Map a small arch label to a (dim, enc, dec) tuple. Inputs to phase 1
    deliberately stay at the smallest scale (dim=64, enc=4, dec=2) so each
    run takes seconds, not minutes."""
    return {
        "tiny": (64, 4, 2),       # Phase 1 default
        "base": (64, 8, 4),       # Same as MUP_BASE_DIM in production sweep
        "dim128": (128, 8, 4),
        "dim256": (256, 8, 4),
        "dim512": (512, 8, 4),
    }.get(arch_label, (64, 4, 2))


def _train_one(model_type, dim, enc, dec, tokens, lr, run_tag, dry_run=False,
               use_compile=True):
    """Train a single (model_type, dim, lr) point in an isolated subprocess.

    Process isolation is the only reliable way to keep torch.compile state
    (dynamo cache, inductor PyCodeCache, async-compile workers, FlexAttention
    HOP closures, allocator fragmentation) from accumulating across runs.
    Each subprocess imports torch fresh, compiles one model, writes its
    result JSON, and exits — the OS reclaims everything. ~5–8s startup
    overhead per run is well under 10% of a 50M-token training run.
    """
    if dry_run:
        return {"model_type": model_type, "dim": dim, "lr": lr,
                "tokens": tokens, "final_eval_loss": None, "dry_run": True}

    result_path = OUT_DIR / run_tag / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()  # avoid reading a stale result if subprocess dies

    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--single-run",
           "--single-mt", model_type,
           "--single-dim", str(dim),
           "--single-enc", str(enc),
           "--single-dec", str(dec),
           "--single-lr", repr(lr),
           "--single-tokens", str(tokens),
           "--single-run-tag", run_tag,
           "--single-result-path", str(result_path)]
    if not use_compile:
        cmd.append("--no-compile")

    print(f"  → subprocess: {model_type} dim={dim} lr={lr:.2e} tokens={tokens:,}")
    subprocess.run(cmd, check=True)
    return json.loads(result_path.read_text())


def _run_single_in_process(model_type, dim, enc, dec, tokens, lr, run_tag,
                           result_path, use_compile=True):
    """Body of a single training run, executed inside the subprocess spawned
    by _train_one. No cross-run hygiene is needed here: process exit handles
    cache flushing, allocator reclamation, and reference cleanup automatically.
    """
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import PreTrainedTokenizerFast

    import scripts.parallel_scaling as ps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(PROJECT_ROOT / "tokenizer" / "tokenizer_32k.json"))
    vocab_size = tokenizer.vocab_size
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0

    # Override BASE_LR for this run. No restore needed — the process exits
    # right after train_one_budget returns.
    ps.BASE_LR = lr
    ps.TUNED_LRS = {model_type: {"base_lr": lr}}

    arch = {"dim": dim, "num_encoder_layers": enc, "num_decoder_layers": dec}
    model = ps.build_model(model_type, arch, vocab_size, device,
                           use_compile=use_compile)
    ne = ps.non_emb_param_count(model)
    info = [{"name": run_tag, "model_type": model_type, "arch": arch,
             "model": model, "ne": ne,
             "needs_blocks": model_type in ("dd", "sed")}]

    # Single-trainer batches: stay tiny so dim=64 runs in <1 min each.
    bs = 8
    ga = 4  # effective batch 32 — small but adequate for LR optima search

    train_ds = load_dataset(
        "json",
        data_files=str(PROJECT_ROOT / "data/Pretrain/slimpajama_6b_packed.jsonl"),
        split="train", streaming=True)
    eval_ds = load_dataset(
        "json",
        data_files=str(PROJECT_ROOT / "data/Pretrain/slimpajama_6b_eval_packed.jsonl"),
        split="train")

    loaders = {
        mt: DataLoader(train_ds, batch_size=bs,
                       collate_fn=ps.build_pretrain_collator(
                           mt, bos_id, eos_id, pad_id, ps.SEQ_LEN,
                           global_seed=42),
                       drop_last=True)
        for mt in ps.MODEL_TYPES
    }
    eval_loaders = {
        mt: DataLoader(eval_ds, batch_size=bs,
                       collate_fn=ps.build_pretrain_collator(
                           mt, bos_id, eos_id, pad_id, ps.SEQ_LEN,
                           global_seed=42),
                       drop_last=True)
        for mt in ps.MODEL_TYPES
    }

    gpu_tflops = ps.detect_gpu_tflops()

    out_subdir = OUT_DIR / run_tag
    results = ps.train_one_budget(
        info, tok_label="phase", total_tokens=tokens,
        batch_size=bs, grad_accum=ga,
        train_loaders=loaders, eval_loaders=eval_loaders,
        device=device, gpu_tflops=gpu_tflops,
        eval_batches=10,
        output_dir=str(out_subdir),
        mid_eval_points=0,
        run_full_eval=False,
        tokenizer=tokenizer,
        run_sft=False)

    key = f"{model_type}_{run_tag}"
    run = results.get(key, {})

    result = {
        "model_type": model_type, "dim": dim, "enc": enc, "dec": dec,
        "lr": lr, "tokens": tokens,
        "final_eval_loss": run.get("final_eval_loss"),
        "final_eval_ppl": run.get("final_eval_ppl"),
        "non_emb_params": run.get("non_emb_params"),
        "training_time_sec": run.get("training_time_sec"),
    }
    Path(result_path).write_text(json.dumps(result, indent=2))


def _phase1(archs, dim, enc, dec, tokens, lrs, dry_run=False, use_compile=True):
    """Sweep LR at base width per arch. Returns list of result dicts."""
    plan = [(mt, lr) for mt in archs for lr in lrs]
    print(f"\n{'═'*60}\n  Phase 1: LR sweep at base "
          f"(dim={dim}, enc={enc}, dec={dec}, tokens={tokens:,})\n"
          f"  {len(plan)} runs ({len(archs)} archs × {len(lrs)} LRs)  "
          f"compile={use_compile}\n{'═'*60}")
    results = []
    for i, (mt, lr) in enumerate(plan, 1):
        run_tag = f"p1_{mt}_dim{dim}_lr{lr:.0e}"
        print(f"\n[{i}/{len(plan)}] {run_tag}")
        t0 = time.time()
        r = _train_one(mt, dim, enc, dec, tokens, lr, run_tag,
                       dry_run=dry_run, use_compile=use_compile)
        r["phase"] = 1
        r["run_tag"] = run_tag
        results.append(r)
        if not dry_run:
            print(f"  loss={r.get('final_eval_loss')}  "
                  f"({(time.time()-t0):.0f}s)")
    return results


def _best_lr_per_arch(phase1_results):
    """Pick the LR with the lowest final_eval_loss for each arch.
    Returns {arch: {best_lr, best_loss, all_lrs}}."""
    by_arch = {}
    for r in phase1_results:
        if r.get("final_eval_loss") is None:
            continue
        by_arch.setdefault(r["model_type"], []).append(r)
    out = {}
    for mt, runs in by_arch.items():
        runs_sorted = sorted(runs, key=lambda r: r["final_eval_loss"])
        best = runs_sorted[0]
        out[mt] = {
            "base_lr": best["lr"],
            "best_loss": best["final_eval_loss"],
            "phase1_dim": best["dim"],
            "phase1_enc": best["enc"],
            "phase1_dec": best["dec"],
            "phase1_tokens": best["tokens"],
            "lr_curve": [(r["lr"], r["final_eval_loss"]) for r in runs],
        }
    return out


def _phase2(archs, lrs_per_arch, dims, enc, dec, tokens, dry_run=False,
            use_compile=True):
    """Verify LR transfer across widths. For each arch, train at each (dim, lr)
    near the phase-1 optimum. Returns list of result dicts."""
    plan = []
    for mt in archs:
        if mt not in lrs_per_arch:
            print(f"[Phase 2] Skipping {mt}: no phase-1 best LR")
            continue
        best_lr = lrs_per_arch[mt]["base_lr"]
        # ±1 step on a log-spaced grid to span the bottom of the U-curve
        try_lrs = sorted({best_lr / 2, best_lr, best_lr * 2})
        for d in dims:
            for lr in try_lrs:
                plan.append((mt, d, lr))
    print(f"\n{'═'*60}\n  Phase 2: LR transfer across widths "
          f"(dims={dims}, enc={enc}, dec={dec}, tokens={tokens:,})\n"
          f"  {len(plan)} runs  compile={use_compile}\n{'═'*60}")
    results = []
    for i, (mt, d, lr) in enumerate(plan, 1):
        run_tag = f"p2_{mt}_dim{d}_lr{lr:.0e}"
        print(f"\n[{i}/{len(plan)}] {run_tag}")
        t0 = time.time()
        r = _train_one(mt, d, enc, dec, tokens, lr, run_tag,
                       dry_run=dry_run, use_compile=use_compile)
        r["phase"] = 2
        r["run_tag"] = run_tag
        results.append(r)
        if not dry_run:
            print(f"  loss={r.get('final_eval_loss')}  "
                  f"({(time.time()-t0):.0f}s)")
    return results


def _write_tuned_json(lrs_per_arch):
    """Write configs/mup_tuned.json with one block per arch. parallel_scaling
    reads this at startup and uses base_lr per arch."""
    # Preserve any pre-existing entries so partial sweeps don't clobber tuned
    # values for archs we didn't re-sweep.
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
              f"(best_loss={info['best_loss']:.4f} at dim={info['phase1_dim']})")


def _plot(all_results):
    """Render LR-vs-loss curves per arch (Phase 1) and per-width LR-transfer
    panels (Phase 2). Saved to checkpoints/mup_base_sweep/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: one panel per arch, x=lr, y=loss
    p1 = [r for r in all_results if r.get("phase") == 1
          and r.get("final_eval_loss") is not None]
    if p1:
        archs = sorted(set(r["model_type"] for r in p1))
        fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4),
                                 sharey=True, squeeze=False)
        for ax, mt in zip(axes[0], archs):
            runs = sorted([r for r in p1 if r["model_type"] == mt],
                          key=lambda r: r["lr"])
            xs = [r["lr"] for r in runs]
            ys = [r["final_eval_loss"] for r in runs]
            ax.plot(xs, ys, "o-", linewidth=2, markersize=8)
            best = min(runs, key=lambda r: r["final_eval_loss"])
            ax.plot(best["lr"], best["final_eval_loss"], "*",
                    markersize=18, color="red", zorder=5,
                    label=f"best: lr={best['lr']:.0e}")
            ax.set_xscale("log")
            ax.set_xlabel("learning rate")
            ax.set_title(f"{mt} (Phase 1)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[0][0].set_ylabel("final eval loss")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"phase1_lr_sweep.{ext}", dpi=150,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {OUT_DIR / 'phase1_lr_sweep.png'}")

    # Phase 2: one panel per arch, x=lr, y=loss, lines=width
    p2 = [r for r in all_results if r.get("phase") == 2
          and r.get("final_eval_loss") is not None]
    if p2:
        archs = sorted(set(r["model_type"] for r in p2))
        fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4),
                                 sharey=True, squeeze=False)
        for ax, mt in zip(axes[0], archs):
            runs = [r for r in p2 if r["model_type"] == mt]
            dims = sorted(set(r["dim"] for r in runs))
            for d in dims:
                drs = sorted([r for r in runs if r["dim"] == d],
                             key=lambda r: r["lr"])
                xs = [r["lr"] for r in drs]
                ys = [r["final_eval_loss"] for r in drs]
                ax.plot(xs, ys, "o-", label=f"dim={d}", linewidth=2)
            ax.set_xscale("log")
            ax.set_xlabel("learning rate")
            ax.set_title(f"{mt} (Phase 2: width transfer)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[0][0].set_ylabel("final eval loss")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"phase2_width_transfer.{ext}", dpi=150,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {OUT_DIR / 'phase2_width_transfer.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", default="dd,sed,dec",
                        help="Comma-separated archs to sweep (default: dd,sed,dec)")
    parser.add_argument("--lrs", default=None,
                        help="Comma-separated LRs for Phase 1 "
                             "(default: 3e-4,6e-4,1e-3,2e-3,4e-3,8e-3,1.5e-2)")
    parser.add_argument("--tokens", type=int, default=DEFAULT_PHASE1_TOKENS,
                        help=f"Tokens per Phase 1 run (default: {DEFAULT_PHASE1_TOKENS:,})")
    parser.add_argument("--phase1-dim", type=int, default=DEFAULT_PHASE1_DIM,
                        help=f"Base dim for Phase 1 (default: {DEFAULT_PHASE1_DIM} = MUP_BASE_DIM)")
    parser.add_argument("--phase1-enc", type=int, default=4,
                        help="Encoder layers in Phase 1 (default 4 — kept tiny for speed)")
    parser.add_argument("--phase1-dec", type=int, default=2,
                        help="Decoder layers in Phase 1 (default 2 — kept tiny for speed)")
    parser.add_argument("--verify-transfer", action="store_true",
                        help="After Phase 1, run Phase 2 LR-transfer verification "
                             "across widths. Adds ~3 hours; recommended before the big sweep.")
    parser.add_argument("--phase2-dims", default=None,
                        help="Comma-separated dims for Phase 2 "
                             f"(default: {DEFAULT_PHASE2_DIMS})")
    parser.add_argument("--phase2-tokens", type=int, default=None,
                        help="Tokens per Phase 2 run (default: same as --tokens)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; don't train")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-render plots from existing results.json; no training")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (default: compile on). Use as "
                             "an escape hatch if compile breaks for some arch.")

    # --single-run: internal subprocess entrypoint invoked by _train_one for
    # one (model_type, dim, lr) point. Not for direct human use; it exists so
    # each training run gets its own clean torch process (no compile-cache or
    # allocator state inherited from prior runs).
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
            run_tag=args.single_run_tag,
            result_path=args.single_result_path,
            use_compile=use_compile)
        return

    archs = [a.strip() for a in args.archs.split(",")]
    lrs = ([float(x) for x in args.lrs.split(",")] if args.lrs else DEFAULT_LRS)
    phase2_dims = ([int(x) for x in args.phase2_dims.split(",")]
                   if args.phase2_dims else DEFAULT_PHASE2_DIMS)
    phase2_tokens = args.phase2_tokens or args.tokens

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    if args.plot_only:
        if not results_path.exists():
            print(f"No results at {results_path} — run without --plot-only first.")
            sys.exit(1)
        all_results = json.loads(results_path.read_text())
        _plot(all_results)
        return

    all_results = []
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        print(f"Resuming from {len(all_results)} prior results in {results_path}")

    # Phase 1
    p1 = _phase1(archs, args.phase1_dim, args.phase1_enc, args.phase1_dec,
                 args.tokens, lrs, dry_run=args.dry_run, use_compile=use_compile)
    all_results.extend(p1)
    if not args.dry_run:
        results_path.write_text(json.dumps(all_results, indent=2))

    best = _best_lr_per_arch([r for r in all_results if r.get("phase") == 1])
    if best and not args.dry_run:
        _write_tuned_json(best)

    # Phase 2 (optional)
    if args.verify_transfer:
        p2 = _phase2(archs, best, phase2_dims, args.phase1_enc, args.phase1_dec,
                     phase2_tokens, dry_run=args.dry_run, use_compile=use_compile)
        all_results.extend(p2)
        if not args.dry_run:
            results_path.write_text(json.dumps(all_results, indent=2))

    if not args.dry_run:
        _plot(all_results)
    print("\nDone.")
    if best:
        print("Per-arch tuned base LRs (also written to configs/mup_tuned.json):")
        for mt, info in best.items():
            print(f"  {mt}: base_lr={info['base_lr']:.2e}")


if __name__ == "__main__":
    main()
