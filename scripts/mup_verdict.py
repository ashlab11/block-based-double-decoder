#!/usr/bin/env python3
"""Render the μP transfer verdict figure from mup_base_sweep results.

Reads checkpoints/mup_base_sweep/results.json (full grid: archs × dims × lrs ×
seeds) and produces:

  checkpoints/mup_base_sweep/mup_verdict.png

Four panels per arch:
  1. Base-width LR sweep (seed-mean ± seed-range)
  2. Full LR×width grid (seed-mean curves; stars at argmin per width)
  3. Argmin-LR vs width on log-log axes (the verdict plot)
  4. Mid-training coord-check trajectory (logits-RMS at fractional progress,
     one line per width, taken at each width's empirical-best LR)

Verdicts (from spread of argmin LRs across widths):
  PASS  — every width's seed-mean argmin lands on the same LR
  PASS* — argmins span 1 grid step, but not monotonic across widths
  WARN  — argmins drift monotonically by 1 grid step (try a wider/finer grid)
  FAIL  — spread > 2× — μP transfer is genuinely broken

Usage:
    python scripts/mup_verdict.py
    python scripts/mup_verdict.py --results path/to/results.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "mup_base_sweep" / "results.json"
DEFAULT_OUT = PROJECT_ROOT / "mup_base_sweep" / "mup_verdict.png"


def _classify(argmins_by_width):
    """PASS / PASS* / WARN / FAIL from {dim: argmin_lr}."""
    if len(argmins_by_width) < 2:
        return "INSUFFICIENT", "Need at least 2 widths to judge transfer"

    dims_sorted = sorted(argmins_by_width)
    lrs = [argmins_by_width[d] for d in dims_sorted]
    spread = max(lrs) / min(lrs)

    monotonic_up = all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1)) and lrs[0] < lrs[-1]
    monotonic_dn = all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)) and lrs[0] > lrs[-1]
    monotonic = monotonic_up or monotonic_dn

    if spread <= 1.001:
        return "PASS", f"all widths argmin at lr={lrs[0]:.0e}"
    if spread <= 2.001 and not monotonic:
        return "PASS*", f"argmin spans 1 grid step ({min(lrs):.0e}..{max(lrs):.0e}) but not monotonic — within resolution"
    if spread <= 2.001 and monotonic:
        return "WARN", f"argmin drifts monotonically by 1 grid step across widths ({lrs[0]:.0e}->{lrs[-1]:.0e})"
    return "FAIL", f"argmin spans full {spread:.0f}x grid ({min(lrs):.0e}..{max(lrs):.0e}) — μP transfer is broken"


def _aggregate(records):
    """Group by (model_type, dim, lr) and compute seed mean / range."""
    by = {}
    for r in records:
        if r.get("final_eval_loss") is None:
            continue
        key = (r["model_type"], r["dim"], r["lr"])
        by.setdefault(key, []).append(r)
    agg = {}
    for key, runs in by.items():
        losses = [r["final_eval_loss"] for r in runs]
        agg[key] = {
            "mean": sum(losses) / len(losses),
            "min": min(losses),
            "max": max(losses),
            "n": len(losses),
            "runs": runs,
        }
    return agg


def _argmin_per_width(agg, mt):
    """For each dim of arch `mt`, return LR with lowest seed-mean eval loss."""
    by_dim = {}
    for (model_type, dim, lr), v in agg.items():
        if model_type != mt:
            continue
        by_dim.setdefault(dim, []).append((lr, v["mean"]))
    return {d: min(rows, key=lambda x: x[1])[0] for d, rows in by_dim.items()}


def _best_lr_at_width(agg, mt, dim):
    """Per-width best LR (used to pick which coord-curve to plot)."""
    rows = [(lr, v["mean"]) for (m, d, lr), v in agg.items()
            if m == mt and d == dim]
    return min(rows, key=lambda x: x[1])[0] if rows else None


def _coord_series(records, mt, dim, lr, metric):
    """Average a coord-curve metric across seeds for a (mt, dim, lr) cell."""
    runs = [r for r in records if r["model_type"] == mt and r["dim"] == dim
            and abs(r["lr"] - lr) < 1e-12 and r.get("coord_curve")]
    if not runs:
        return [], []
    by_frac = {}
    for r in runs:
        for c in r["coord_curve"]:
            f = c.get("fraction")
            if f is None:
                continue
            v = c.get(metric)
            if v is None:
                # logits_rms / embed_rms / qk_std map straight to a key; for
                # layer_rms we accept "last_dec" / "last_enc" pseudo-keys.
                if metric == "last_layer":
                    lr_dict = c.get("layer_rms") or {}
                    if not lr_dict:
                        continue
                    last_key = max(lr_dict.keys(),
                                   key=lambda k: int(k.rsplit("_", 1)[-1]) if k.rsplit("_", 1)[-1].isdigit() else -1)
                    v = lr_dict[last_key]
                else:
                    continue
            by_frac.setdefault(f, []).append(v)
    fracs = sorted(by_frac)
    means = [statistics.mean(by_frac[f]) for f in fracs]
    return fracs, means


def render(results_path, out_path):
    if not results_path.exists():
        print(f"No results at {results_path}", file=sys.stderr)
        sys.exit(1)
    all_results = json.loads(results_path.read_text())
    finished = [r for r in all_results if r.get("final_eval_loss") is not None]
    if not finished:
        print("No finished runs in results.json yet — sweep still running?",
              file=sys.stderr)
        sys.exit(1)

    archs = sorted({r["model_type"] for r in finished})
    agg = _aggregate(finished)

    base_dim = min({r["dim"] for r in finished})

    n = len(archs)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)
    overall_verdict = []

    for row, mt in enumerate(archs):
        # ── Panel 1: base-width LR sweep with seed band ──────────────────
        ax = axes[row][0]
        cells = sorted([((m, d, lr), v) for (m, d, lr), v in agg.items()
                        if m == mt and d == base_dim],
                       key=lambda x: x[0][2])
        if cells:
            xs = [k[2] for k, _ in cells]
            means = [v["mean"] for _, v in cells]
            mins = [v["min"] for _, v in cells]
            maxs = [v["max"] for _, v in cells]
            ax.fill_between(xs, mins, maxs, color="steelblue", alpha=0.2,
                            label="seed range")
            ax.plot(xs, means, "o-", color="steelblue", linewidth=2,
                    markersize=7, label="seed mean")
            best_idx = means.index(min(means))
            ax.plot(xs[best_idx], means[best_idx], "*", markersize=18,
                    color="red", zorder=5)
            ax.annotate(f"best lr={xs[best_idx]:.0e}",
                        xy=(xs[best_idx], means[best_idx]),
                        xytext=(8, 8), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("final eval loss")
        ax.set_title(f"{mt} — base width (dim={base_dim})\nseed-mean ± seed-range")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        # ── Panel 2: full LR × width grid ─────────────────────────────────
        ax = axes[row][1]
        argmins = {}
        dims = sorted({d for (m, d, _) in agg if m == mt})
        cmap = plt.get_cmap("viridis", max(len(dims), 2))
        for i, d in enumerate(dims):
            cells_d = sorted([((m, dd, lr), v) for (m, dd, lr), v in agg.items()
                              if m == mt and dd == d],
                             key=lambda x: x[0][2])
            if not cells_d:
                continue
            xs = [k[2] for k, _ in cells_d]
            means = [v["mean"] for _, v in cells_d]
            mins = [v["min"] for _, v in cells_d]
            maxs = [v["max"] for _, v in cells_d]
            color = cmap(i)
            ax.fill_between(xs, mins, maxs, color=color, alpha=0.15)
            ax.plot(xs, means, "o-", color=color, linewidth=2, label=f"dim={d}")
            best_idx = means.index(min(means))
            argmins[d] = xs[best_idx]
            ax.plot(xs[best_idx], means[best_idx], "*", markersize=18,
                    color=color, markeredgecolor="black", markeredgewidth=1.0,
                    zorder=5)
        ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_title(f"{mt} — full LR × width grid\n(stars = argmin per width)")
        ax.grid(True, alpha=0.3)

        # ── Panel 3: argmin LR vs width with verdict ─────────────────────
        ax = axes[row][2]
        verdict, detail = _classify(argmins)
        overall_verdict.append((mt, verdict, detail))
        if argmins:
            dims_sorted = sorted(argmins)
            ax.plot(dims_sorted, [argmins[d] for d in dims_sorted],
                    "o-", color="darkorange", linewidth=2.5, markersize=10)
            for d in dims_sorted:
                ax.annotate(f"{argmins[d]:.0e}", xy=(d, argmins[d]),
                            xytext=(6, 6), textcoords="offset points")
            median = sorted(argmins.values())[len(argmins) // 2]
            ax.axhline(median, color="gray", linestyle="--", alpha=0.5,
                       label="perfect transfer")
            ax.legend()
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("width (dim)"); ax.set_ylabel("argmin LR")
        color = {"PASS": "green", "PASS*": "olive",
                 "WARN": "darkorange", "FAIL": "crimson",
                 "INSUFFICIENT": "gray"}.get(verdict, "black")
        ax.set_title(f"{mt} — verdict: {verdict}", color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)

    overall = ", ".join(f"{mt}: {v}" for mt, v, _ in overall_verdict)
    n_seeds = len({r.get("seed", 0) for r in finished})
    fig.suptitle(f"μP transfer check — {overall}  "
                 f"(n_seeds={n_seeds}, full LR×width grid)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved {out_path}")
    print(f"Saved {pdf_path}")
    print("\nVerdict per arch:")
    for mt, v, detail in overall_verdict:
        print(f"  {mt}: {v} — {detail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help=f"Path to results.json (default: {DEFAULT_RESULTS})")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Output PNG path (default: {DEFAULT_OUT})")
    args = ap.parse_args()
    render(Path(args.results), Path(args.out))


if __name__ == "__main__":
    main()
