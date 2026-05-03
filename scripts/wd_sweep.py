#!/usr/bin/env python3
"""Per-arch weight-decay sweep — run AFTER mup_base_sweep + mup_verdict.

Workflow this script slots into:

    1. python scripts/mup_base_sweep.py --archs dd,sed,dec --verify-transfer
    2. python scripts/mup_verdict.py             # confirm PASS / PASS*
    3. python scripts/wd_sweep.py                # this script
    4. python scripts/parallel_scaling.py ...    # the big run

What it does:
    For each arch in {dd, sed, dec}, hold the μP-tuned base LR fixed (read from
    configs/mup_tuned.json) and sweep WD over a small log-spaced grid at the μP
    base width (dim=64, ~0.5M params, 50M tokens — same cell shape as Phase 1
    of mup_base_sweep, so wall-clock is comparable: ~1 hour for 5 WDs × 3 archs
    on one GPU).

Why no Phase-2-style width transfer for WD:
    μP makes a transfer claim for LR, not WD. The AdamW WD optimum is set by
    the relative scale of (decay term) vs (gradient·lr) per parameter, and once
    LR is locked per arch via μP, the WD optimum is essentially per-arch — not
    per-arch-per-width. Sweeping WD at the base width and reusing it at every
    width is the standard practice. If you ever want to falsify that assumption
    on this repo, run the same sweep at a second width and compare argmins —
    but it's not the default expectation and adding it would burn ~3× the
    compute for what is, in the literature, a non-finding.

Output: configs/wd_tuned.json — read at startup by parallel_scaling.py to
override the per-arch WD. Format:
    {"dd":  {"weight_decay": 0.1, "best_loss": 5.83, "tuned_lr_used": 0.002,
             "wd_curve": [(0.0, 5.91), (0.01, 5.86), ...]},
     "sed": {...},
     "dec": {...}}

Usage:
    # Default sweep (5 WDs × 3 archs × 2 seeds = 30 cells):
    python scripts/wd_sweep.py --archs dd,sed,dec

    # Single-seed quick run (debug / smoke test):
    python scripts/wd_sweep.py --archs dd --wds 0.0,0.05,0.1 \
        --seeds 0 --tokens 10000000

    # Dry-run (planning only, no training):
    python scripts/wd_sweep.py --dry-run

    # Just plot existing results:
    python scripts/wd_sweep.py --plot-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Defaults chosen for direct comparability with mup_base_sweep Phase 1.
# Same dim/enc/dec/tokens means each WD cell costs ≈ each LR cell of Phase 1,
# so wall-clock is predictable: ~1 hr for 5 WDs × 3 archs on one GPU.
DEFAULT_DIM = 64           # = MUP_BASE_DIM in parallel_scaling
DEFAULT_ENC = 4
DEFAULT_DEC = 2
DEFAULT_TOKENS = 50_000_000

# Log-spaced grid centered on the canonical AdamW default (0.1). 0.0 is
# included as the "no WD" reference; 0.2 is the upper end you'd plausibly use.
# Five points is enough to draw a U-curve and pick a minimum without burning
# the LR sweep's compute budget over again.
DEFAULT_WDS = [0.0, 0.01, 0.05, 0.1, 0.2]

# Two seeds matches mup_base_sweep.DEFAULT_SEEDS so each (arch, wd) cell is
# averaged over the same noise budget the LR sweep used. The argmin operates
# on seed-mean loss rather than any single-seed loss.
DEFAULT_SEEDS = [0, 1]

OUT_DIR = PROJECT_ROOT / "checkpoints" / "wd_sweep"
TUNED_LR_PATH = PROJECT_ROOT / "configs" / "mup_tuned.json"
TUNED_WD_PATH = PROJECT_ROOT / "configs" / "wd_tuned.json"


def _load_tuned_lrs():
    """Read configs/mup_tuned.json and return {arch: base_lr}.

    Hard fail if the file is missing — this script's whole premise is "the LR
    sweep already ran". A WD sweep at the wrong LR repeats the exact mistake
    the codebase just wiped (commits b7c33dc / 85a5567 / 627dbf4).
    """
    if not TUNED_LR_PATH.exists():
        sys.exit(
            f"ERROR: {TUNED_LR_PATH} not found.\n"
            "Run scripts/mup_base_sweep.py (and ideally mup_verdict.py) first;\n"
            "this script reads the per-arch tuned LR from that file before\n"
            "sweeping WD around it. Sweeping WD at an untuned LR would repeat\n"
            "the exact failure the codebase just wiped."
        )
    data = json.loads(TUNED_LR_PATH.read_text())
    out = {}
    for arch, info in data.items():
        if isinstance(info, dict) and "base_lr" in info:
            out[arch] = float(info["base_lr"])
    if not out:
        sys.exit(f"ERROR: {TUNED_LR_PATH} has no base_lr entries.")
    return out


def _train_one(model_type, dim, enc, dec, tokens, lr, wd, seed, run_tag,
               dry_run=False, use_compile=True):
    """Train a single (model_type, wd, seed) cell in an isolated subprocess.

    Same process-isolation rationale as mup_base_sweep._train_one: torch.compile
    state, FlexAttention HOP closures, and inductor caches accumulate across
    runs and eventually crash. One subprocess per cell, OS reclaims everything.
    """
    if dry_run:
        return {"model_type": model_type, "dim": dim, "lr": lr, "wd": wd,
                "seed": seed, "tokens": tokens,
                "final_eval_loss": None, "dry_run": True}

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
           "--single-wd", repr(wd),
           "--single-seed", str(seed),
           "--single-tokens", str(tokens),
           "--single-run-tag", run_tag,
           "--single-result-path", str(result_path)]
    if not use_compile:
        cmd.append("--no-compile")

    print(f"  → subprocess: {model_type} dim={dim} lr={lr:.2e} wd={wd:.3g} "
          f"seed={seed} tokens={tokens:,}")
    subprocess.run(cmd, check=True)
    return json.loads(result_path.read_text())


def _run_single_in_process(model_type, dim, enc, dec, tokens, lr, wd, seed,
                           run_tag, result_path, use_compile=True):
    """Subprocess body. Sets parallel_scaling module globals for this single
    cell, builds the model, runs train_one_budget, writes result JSON, exits.
    """
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import PreTrainedTokenizerFast

    import scripts.parallel_scaling as ps

    # Match mup_base_sweep._run_single_in_process seeding so seed=0/1 here
    # produce comparable noise to the LR sweep cells averaged into mup_tuned.json.
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

    # Pin LR and WD for this run. No restore needed — subprocess exits after
    # train_one_budget returns. TUNED_LRS is also populated so logging shows
    # the tuned LR rather than the BASE_LR fallback path.
    ps.BASE_LR = lr
    ps.TUNED_LRS = {model_type: {"base_lr": lr}}
    ps.WD_OVERRIDE = wd

    arch = {"dim": dim, "num_encoder_layers": enc, "num_decoder_layers": dec}
    model = ps.build_model(model_type, arch, vocab_size, device,
                           use_compile=use_compile)
    ne = ps.non_emb_param_count(model)
    info = [{"name": run_tag, "model_type": model_type, "arch": arch,
             "model": model, "ne": ne,
             "needs_blocks": model_type in ("dd", "sed")}]

    # Same micro/effective batch as mup_base_sweep Phase 1. Effective 32 is
    # tiny but adequate for differentiating WD optima — the loss surface w.r.t.
    # WD is shallower than LR, so the same noise budget is enough.
    bs = 8
    ga = 4

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

    out_subdir = OUT_DIR / run_tag
    results = ps.train_one_budget(
        info, tok_label="wd", total_tokens=tokens,
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
        "lr": lr, "wd": wd, "seed": seed, "tokens": tokens,
        "final_eval_loss": run.get("final_eval_loss"),
        "final_eval_ppl": run.get("final_eval_ppl"),
        "non_emb_params": run.get("non_emb_params"),
        "training_time_sec": run.get("training_time_sec"),
    }
    Path(result_path).write_text(json.dumps(result, indent=2))


def _sweep(archs, tuned_lrs, wds, seeds, dim, enc, dec, tokens, dry_run=False,
           use_compile=True, existing_results=None):
    """Sweep WD × seed per arch at the per-arch tuned LR. Skip cells already in
    `existing_results` so resume works at (arch, wd, seed) granularity.
    Old single-seed results without a `seed` field are treated as seed=0."""
    done = {(r["model_type"], r["wd"], r.get("seed", 0))
            for r in (existing_results or [])
            if r.get("final_eval_loss") is not None}

    plan = [(mt, wd, s) for mt in archs for wd in wds for s in seeds
            if (mt, wd, s) not in done]
    skipped = (len(archs) * len(wds) * len(seeds)) - len(plan)
    print(f"\n{'═'*60}\n  WD sweep at base width "
          f"(dim={dim}, enc={enc}, dec={dec}, tokens={tokens:,})\n"
          f"  {len(plan)} runs to do  ({skipped} cached)  "
          f"{len(archs)} archs × {len(wds)} WDs × {len(seeds)} seeds  "
          f"compile={use_compile}\n"
          f"  Per-arch LRs (from configs/mup_tuned.json):\n"
          + "".join(f"    {mt}: {tuned_lrs[mt]:.2e}\n" for mt in archs)
          + f"{'═'*60}")
    results = []
    for i, (mt, wd, s) in enumerate(plan, 1):
        if mt not in tuned_lrs:
            print(f"[{i}/{len(plan)}] SKIP {mt}: no tuned LR for this arch in "
                  f"configs/mup_tuned.json — re-run mup_base_sweep --archs {mt}")
            continue
        lr = tuned_lrs[mt]
        run_tag = f"{mt}_dim{dim}_lr{lr:.0e}_wd{wd:.3g}_s{s}"
        print(f"\n[{i}/{len(plan)}] {run_tag}")
        t0 = time.time()
        r = _train_one(mt, dim, enc, dec, tokens, lr, wd, s, run_tag,
                       dry_run=dry_run, use_compile=use_compile)
        r["run_tag"] = run_tag
        results.append(r)
        if not dry_run:
            print(f"  loss={r.get('final_eval_loss')}  "
                  f"({(time.time()-t0):.0f}s)")
    return results


def _best_wd_per_arch(results, tuned_lrs):
    """Per arch, pick the WD that minimizes seed-mean eval loss.
    Mirrors mup_base_sweep._best_lr_per_arch's averaging convention so the
    two configs (mup_tuned / wd_tuned) report comparable noise budgets."""
    by = {}
    for r in results:
        if r.get("final_eval_loss") is None:
            continue
        by.setdefault((r["model_type"], r["wd"]), []).append(r)

    by_arch = {}
    for (mt, wd), cells in by.items():
        mean_loss = sum(c["final_eval_loss"] for c in cells) / len(cells)
        # Carry one sample to read dim/tokens/lr off (those are constant across
        # seeds for a given (mt, wd) — only seed and final_eval_loss differ).
        by_arch.setdefault(mt, []).append((wd, mean_loss, len(cells), cells[0]))

    out = {}
    for mt, rows in by_arch.items():
        rows.sort(key=lambda x: x[1])
        best_wd, best_mean, n_seeds, sample = rows[0]
        out[mt] = {
            "weight_decay": best_wd,
            "best_loss": best_mean,
            "n_seeds": n_seeds,
            "tuned_lr_used": tuned_lrs.get(mt, sample["lr"]),
            "sweep_dim": sample["dim"],
            "sweep_tokens": sample["tokens"],
            "wd_curve": [(wd, mean) for wd, mean, _, _ in
                         sorted(rows, key=lambda x: x[0])],
        }
    return out


def _write_tuned_json(wds_per_arch):
    """Write configs/wd_tuned.json. Preserve any prior entries so partial
    sweeps don't clobber tuned values for archs we didn't re-sweep."""
    if TUNED_WD_PATH.exists():
        existing = json.loads(TUNED_WD_PATH.read_text())
    else:
        existing = {}
    existing.update(wds_per_arch)
    TUNED_WD_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_WD_PATH.write_text(json.dumps(existing, indent=2))
    print(f"\nWrote tuned WDs to: {TUNED_WD_PATH}")
    for mt, info in wds_per_arch.items():
        print(f"  {mt}: weight_decay={info['weight_decay']:.3g}  "
              f"(seed-mean loss={info['best_loss']:.4f}, "
              f"n_seeds={info['n_seeds']}, "
              f"lr={info['tuned_lr_used']:.2e})")


def _plot(all_results):
    """Render WD-vs-loss U-curves, one panel per arch. Saved to
    checkpoints/wd_sweep/wd_sweep.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    finished = [r for r in all_results if r.get("final_eval_loss") is not None]
    if not finished:
        print("No finished runs to plot.")
        return

    archs = sorted(set(r["model_type"] for r in finished))
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4),
                             sharey=True, squeeze=False)
    for ax, mt in zip(axes[0], archs):
        cells = [r for r in finished if r["model_type"] == mt]
        # Group by wd, compute seed-mean and per-seed spread (min/max). The
        # mean drives the argmin; the band shows whether neighbouring WDs are
        # within seed noise of each other.
        by_wd = {}
        for r in cells:
            by_wd.setdefault(r["wd"], []).append(r["final_eval_loss"])
        rows = sorted([(wd, sum(losses) / len(losses),
                        min(losses), max(losses), len(losses))
                       for wd, losses in by_wd.items()], key=lambda t: t[0])
        xs = [r[0] for r in rows]
        means = [r[1] for r in rows]
        lo = [r[2] for r in rows]
        hi = [r[3] for r in rows]
        n_max = max(r[4] for r in rows)
        # Use symlog so wd=0 is on-axis (a plain log scale would drop it).
        ax.plot(xs, means, "o-", linewidth=2, markersize=8,
                label=f"seed-mean (n≤{n_max})")
        if n_max > 1:
            ax.fill_between(xs, lo, hi, alpha=0.2,
                            label="seed min/max")
        best_idx = min(range(len(rows)), key=lambda i: rows[i][1])
        ax.plot(rows[best_idx][0], rows[best_idx][1], "*",
                markersize=18, color="red", zorder=5,
                label=f"best: wd={rows[best_idx][0]:.3g}")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("weight decay")
        ax.set_title(f"{mt} (lr={cells[0]['lr']:.2e})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("final eval loss")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"wd_sweep.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'wd_sweep.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", default="dd,sed,dec",
                        help="Comma-separated archs to sweep (default: dd,sed,dec)")
    parser.add_argument("--wds", default=None,
                        help=f"Comma-separated WDs (default: {DEFAULT_WDS})")
    parser.add_argument("--seeds", default=None,
                        help=f"Comma-separated seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"Tokens per cell (default: {DEFAULT_TOKENS:,})")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"Sweep width (default: {DEFAULT_DIM} = MUP_BASE_DIM)")
    parser.add_argument("--enc", type=int, default=DEFAULT_ENC,
                        help=f"Encoder layers (default: {DEFAULT_ENC})")
    parser.add_argument("--dec", type=int, default=DEFAULT_DEC,
                        help=f"Decoder layers (default: {DEFAULT_DEC})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; don't train")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-render plots from existing results.json; no training")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (escape hatch)")

    # --single-run: subprocess entrypoint invoked by _train_one. Not for direct
    # human use; exists so each training cell gets its own clean torch process.
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
    parser.add_argument("--single-wd", type=float, default=None,
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
            tokens=args.single_tokens, lr=args.single_lr, wd=args.single_wd,
            seed=args.single_seed,
            run_tag=args.single_run_tag,
            result_path=args.single_result_path,
            use_compile=use_compile)
        return

    archs = [a.strip() for a in args.archs.split(",")]
    wds = ([float(x) for x in args.wds.split(",")] if args.wds else DEFAULT_WDS)
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds
             else list(DEFAULT_SEEDS))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    if args.plot_only:
        if not results_path.exists():
            print(f"No results at {results_path} — run without --plot-only first.")
            sys.exit(1)
        _plot(json.loads(results_path.read_text()))
        return

    tuned_lrs = _load_tuned_lrs()

    all_results = []
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        print(f"Resuming from {len(all_results)} prior results in {results_path}")

    new = _sweep(archs, tuned_lrs, wds, seeds, args.dim, args.enc, args.dec,
                 args.tokens, dry_run=args.dry_run, use_compile=use_compile,
                 existing_results=all_results)
    all_results.extend(new)
    if not args.dry_run:
        results_path.write_text(json.dumps(all_results, indent=2))

    best = _best_wd_per_arch(all_results, tuned_lrs)
    if best and not args.dry_run:
        _write_tuned_json(best)
        _plot(all_results)
    print("\nDone.")
    if best:
        print("Per-arch tuned WDs (also written to configs/wd_tuned.json):")
        for mt, info in best.items():
            print(f"  {mt}: weight_decay={info['weight_decay']:.3g}")


if __name__ == "__main__":
    main()
