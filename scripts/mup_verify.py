#!/usr/bin/env python3
"""Verify μP scaling by comparing LR transfer across model widths.

The standard μP verification: sweep learning rates at multiple widths and
check that the optimal LR stays constant under μP (unlike standard param
where optimal LR shifts with width).

Usage:
    # Run all experiments (GPU required):
    python scripts/mup_verify.py --run

    # Just plot from saved results:
    python scripts/mup_verify.py --plot

    # Custom grid:
    python scripts/mup_verify.py --run --tokens 10000000 \
        --params 500000,2500000,5000000 --lrs 1e-4,3e-4,1e-3,3e-3,1e-2

    # Single mode (SP only or μP only):
    python scripts/mup_verify.py --run --mode mup
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.api import train
from configs.scaling import ARCHITECTURES, interpolate_architecture, lr_for_dim

# ── Defaults ──────────────────────────────────────────────────────────────────

MUP_BASE_DIM = 64
VERIFY_DIR = Path("checkpoints/mup_verify")
RESULTS_FILE = VERIFY_DIR / "results.json"

DEFAULT_PARAMS = [500_000, 2_500_000, 5_000_000, 15_000_000]
DEFAULT_LRS = [3e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2]
DEFAULT_TOKENS = 50_000_000


# ── Running experiments ───────────────────────────────────────────────────────

def _run_one(params, lr, tokens, mup_base_dim, mode_tag):
    """Run a single training job and return the result dict."""
    arch = interpolate_architecture(params)
    dim = arch["dim"]
    name = f"mup_verify_{mode_tag}_{params}p_dim{dim}_lr{lr:.0e}"

    print(f"\n{'='*60}")
    print(f"  {mode_tag.upper()} | params={params:,} (dim={dim}) | lr={lr:.1e}")
    print(f"  tokens={tokens:,} | mup_base_dim={mup_base_dim}")
    print(f"{'='*60}")

    try:
        r = train(
            params=params, tokens=tokens,
            mup_base_dim=mup_base_dim, lr=lr, run_name=name,
        )
        return {
            "params": params,
            "dim": dim,
            "lr": lr,
            "mode": mode_tag,
            "final_eval_loss": r["final_eval_loss"],
            "final_eval_ppl": r["final_eval_ppl"],
            "tokens_seen": r["tokens_seen"],
            "train_curve": r.get("train_curve", []),
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        return {
            "params": params, "dim": dim, "lr": lr, "mode": mode_tag,
            "final_eval_loss": None, "final_eval_ppl": None,
            "tokens_seen": 0, "train_curve": [],
        }


def run_experiments(param_sizes, lrs, tokens, modes):
    """Run the full grid of (params x LR x mode) experiments."""
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    # Load any existing results to allow incremental runs
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = []

    existing_keys = {
        (r["params"], r["lr"], r["mode"]) for r in all_results
    }

    for params in param_sizes:
        for lr in lrs:
            for mode in modes:
                key = (params, lr, mode)
                if key in existing_keys:
                    print(f"Skipping {mode} params={params} lr={lr} (already done)")
                    continue

                mup_base_dim = MUP_BASE_DIM if mode == "mup" else 0
                result = _run_one(params, lr, tokens, mup_base_dim, mode)
                all_results.append(result)

                # Save after each run for crash resilience
                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)

    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_lr_transfer(results):
    """Plot final eval loss vs LR for each width, SP vs μP side by side."""
    modes = sorted(set(r["mode"] for r in results))
    n_panels = len(modes)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    mode_titles = {"sp": "Standard Parameterization", "mup": "μP (Maximal Update Param)"}
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 10))

    for ax, mode in zip(axes, modes):
        mode_results = [r for r in results if r["mode"] == mode]

        # Group by param size
        by_params = {}
        for r in mode_results:
            p = r["params"]
            if p not in by_params:
                by_params[p] = []
            by_params[p].append(r)

        for i, params in enumerate(sorted(by_params.keys())):
            runs = sorted(by_params[params], key=lambda r: r["lr"])
            lrs = [r["lr"] for r in runs]
            losses = [r["final_eval_loss"] for r in runs]

            # Filter out failed runs
            valid = [(l, lo) for l, lo in zip(lrs, losses) if lo is not None]
            if not valid:
                continue
            lrs_v, losses_v = zip(*valid)

            dim = runs[0]["dim"]
            label = f"{params/1e6:.1f}M (dim={dim})"
            ax.plot(lrs_v, losses_v, "o-", label=label, color=colors[i % len(colors)],
                    linewidth=2, markersize=6)

            # Mark the minimum
            best_idx = np.argmin(losses_v)
            ax.plot(lrs_v[best_idx], losses_v[best_idx], "*",
                    color=colors[i % len(colors)], markersize=14, zorder=5)

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=13)
        ax.set_ylabel("Final Eval Loss" if ax == axes[0] else "", fontsize=13)
        ax.set_title(mode_titles.get(mode, mode), fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    plt.suptitle("μP Verification: LR Transfer Across Widths", fontsize=15, y=1.02)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = VERIFY_DIR / f"mup_lr_transfer.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {VERIFY_DIR / 'mup_lr_transfer.png'}")
    plt.close(fig)


def plot_train_curves(results):
    """Plot training loss curves for each width at the best LR, SP vs μP."""
    modes = sorted(set(r["mode"] for r in results))
    if len(modes) < 2:
        return  # need both modes for comparison

    # Find best LR per (mode, params)
    best = {}
    for r in results:
        if r["final_eval_loss"] is None:
            continue
        key = (r["mode"], r["params"])
        if key not in best or r["final_eval_loss"] < best[key]["final_eval_loss"]:
            best[key] = r

    param_sizes = sorted(set(r["params"] for r in results))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(param_sizes)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    mode_titles = {"sp": "SP (best LR per width)", "mup": "μP (best LR per width)"}

    for ax, mode in zip(axes, modes):
        for i, params in enumerate(param_sizes):
            key = (mode, params)
            if key not in best or not best[key].get("train_curve"):
                continue
            r = best[key]
            steps, losses = zip(*r["train_curve"])
            dim = r["dim"]
            label = f"{params/1e6:.1f}M (dim={dim}, lr={r['lr']:.1e})"
            ax.plot(steps, losses, label=label, color=colors[i], linewidth=1.5)

        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel("Train Loss" if ax == axes[0] else "", fontsize=13)
        ax.set_title(mode_titles.get(mode, mode), fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves at Best LR", fontsize=15, y=1.02)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(VERIFY_DIR / f"mup_train_curves.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved: {VERIFY_DIR / 'mup_train_curves.png'}")
    plt.close(fig)


def plot_optimal_lr_vs_width(results):
    """Plot optimal LR vs model dim — should be flat for μP, decreasing for SP."""
    modes = sorted(set(r["mode"] for r in results))

    fig, ax = plt.subplots(figsize=(8, 5))
    mode_styles = {"sp": ("o--", "Standard Param"), "mup": ("s-", "μP")}

    for mode in modes:
        mode_results = [r for r in results if r["mode"] == mode and r["final_eval_loss"] is not None]

        # Find best LR per param size
        by_params = {}
        for r in mode_results:
            p = r["params"]
            if p not in by_params or r["final_eval_loss"] < by_params[p]["final_eval_loss"]:
                by_params[p] = r

        if not by_params:
            continue

        dims = [by_params[p]["dim"] for p in sorted(by_params)]
        best_lrs = [by_params[p]["lr"] for p in sorted(by_params)]

        style, label = mode_styles.get(mode, ("o-", mode))
        ax.plot(dims, best_lrs, style, label=label, linewidth=2, markersize=8)

    ax.set_xlabel("Model Dimension (width)", fontsize=13)
    ax.set_ylabel("Optimal Learning Rate", fontsize=13)
    ax.set_yscale("log")
    ax.set_title("Optimal LR vs Width", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(VERIFY_DIR / f"mup_optimal_lr.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved: {VERIFY_DIR / 'mup_optimal_lr.png'}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify μP scaling: sweep LRs across widths and plot"
    )
    parser.add_argument("--run", action="store_true",
                        help="Run training experiments (GPU required)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots from saved results")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"Token budget per run (default: {DEFAULT_TOKENS:,})")
    parser.add_argument("--params", type=str, default=None,
                        help="Comma-separated param counts (e.g. 500000,2500000)")
    parser.add_argument("--lrs", type=str, default=None,
                        help="Comma-separated learning rates (e.g. 1e-3,3e-3)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["sp", "mup", "both"],
                        help="Run standard param, μP, or both")
    args = parser.parse_args()

    param_sizes = ([int(x) for x in args.params.split(",")]
                   if args.params else DEFAULT_PARAMS)
    lrs = ([float(x) for x in args.lrs.split(",")]
           if args.lrs else DEFAULT_LRS)
    modes = ["sp", "mup"] if args.mode == "both" else [args.mode]

    if args.run:
        results = run_experiments(param_sizes, lrs, args.tokens, modes)
        plot_lr_transfer(results)
        plot_train_curves(results)
        plot_optimal_lr_vs_width(results)

    elif args.plot:
        if not RESULTS_FILE.exists():
            print(f"No results found at {RESULTS_FILE}. Run with --run first.")
            sys.exit(1)
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        plot_lr_transfer(results)
        plot_train_curves(results)
        plot_optimal_lr_vs_width(results)

    else:
        parser.print_help()
        print(f"\nGrid size: {len(param_sizes)} widths x {len(lrs)} LRs x "
              f"{len(modes)} modes = {len(param_sizes) * len(lrs) * len(modes)} runs")
        print(f"Token budget: {args.tokens:,} per run")
        for p in param_sizes:
            arch = interpolate_architecture(p)
            print(f"  {p:>10,} params -> dim={arch['dim']}, "
                  f"enc={arch['num_encoder_layers']}, dec={arch['num_decoder_layers']}")


if __name__ == "__main__":
    main()
