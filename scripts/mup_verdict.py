#!/usr/bin/env python3
"""Render a single MuP-working-or-not verdict figure from mup_base_sweep results.

Reads checkpoints/mup_base_sweep/results.json (written by mup_base_sweep.py
when run with --verify-transfer) and produces:

  checkpoints/mup_base_sweep/mup_verdict.png

Three panels per arch:
  1. Phase 1 U-curve at base width (full LR sweep, basin shape)
  2. Phase 2 U-curves, one per width, with argmin starred (the MuP coord check)
  3. argmin-LR vs width on log-log axes (flat horizontal => transfer working)

Verdict (printed and rendered in the figure title):
  PASS  — every width has the same argmin LR on the Phase 2 grid
  PASS* — argmins span one grid step (±2x) but not monotonic across widths
  WARN  — argmins drift monotonically across widths by one step
  FAIL  — argmins span the full 4x grid (best/2 at one width, best*2 at another)

Usage:
    python scripts/mup_verdict.py
    python scripts/mup_verdict.py --results path/to/results.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "checkpoints" / "mup_base_sweep" / "results.json"
DEFAULT_OUT = PROJECT_ROOT / "checkpoints" / "mup_base_sweep" / "mup_verdict.png"


def _classify(argmins_by_width):
    """Decide PASS / PASS* / WARN / FAIL from {dim: argmin_lr}.

    The Phase 2 grid is {best/2, best, best*2}, so any argmin spread is a
    multiple of 2 in log-LR. Spread of 1.0 means perfect agreement; 2.0 is
    one grid step (within resolution); 4.0 is the full grid (clear failure).
    """
    if len(argmins_by_width) < 2:
        return "INSUFFICIENT", "Need at least 2 widths in Phase 2 to judge transfer"

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


def _argmin_per_width(p2_runs):
    """For each width in p2_runs, return the LR that minimized eval loss."""
    by_dim = {}
    for r in p2_runs:
        by_dim.setdefault(r["dim"], []).append(r)
    return {
        d: min(rs, key=lambda r: r["final_eval_loss"])["lr"]
        for d, rs in by_dim.items()
    }


def render(results_path, out_path):
    if not results_path.exists():
        print(f"No results at {results_path}", file=sys.stderr)
        sys.exit(1)
    all_results = json.loads(results_path.read_text())

    finished = [r for r in all_results if r.get("final_eval_loss") is not None]
    archs = sorted({r["model_type"] for r in finished})
    if not archs:
        print("No finished runs in results.json yet — is the sweep still going?",
              file=sys.stderr)
        sys.exit(1)

    n = len(archs)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)

    overall_verdict = []
    for row, mt in enumerate(archs):
        p1 = sorted([r for r in finished if r["phase"] == 1 and r["model_type"] == mt],
                    key=lambda r: r["lr"])
        p2 = [r for r in finished if r["phase"] == 2 and r["model_type"] == mt]

        # ── Panel 1: Phase 1 U-curve ──────────────────────────────────────
        ax = axes[row][0]
        if p1:
            xs = [r["lr"] for r in p1]
            ys = [r["final_eval_loss"] for r in p1]
            ax.plot(xs, ys, "o-", color="steelblue", linewidth=2, markersize=7)
            best = min(p1, key=lambda r: r["final_eval_loss"])
            ax.plot(best["lr"], best["final_eval_loss"], "*",
                    markersize=18, color="red", zorder=5)
            ax.annotate(f"best lr={best['lr']:.0e}",
                        xy=(best["lr"], best["final_eval_loss"]),
                        xytext=(8, 8), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("final eval loss")
        ax.set_title(f"{mt} — Phase 1 (base width)")
        ax.grid(True, alpha=0.3)

        # ── Panel 2: Phase 2 per-width U-curves ───────────────────────────
        ax = axes[row][1]
        argmins = {}
        if p2:
            dims = sorted({r["dim"] for r in p2})
            cmap = plt.get_cmap("viridis", max(len(dims), 2))
            for i, d in enumerate(dims):
                runs = sorted([r for r in p2 if r["dim"] == d],
                              key=lambda r: r["lr"])
                xs = [r["lr"] for r in runs]
                ys = [r["final_eval_loss"] for r in runs]
                color = cmap(i)
                ax.plot(xs, ys, "o-", color=color, linewidth=2,
                        label=f"dim={d}")
                bw = min(runs, key=lambda r: r["final_eval_loss"])
                argmins[d] = bw["lr"]
                ax.plot(bw["lr"], bw["final_eval_loss"], "*",
                        markersize=18, color=color, markeredgecolor="black",
                        markeredgewidth=1.0, zorder=5)
            ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_title(f"{mt} — Phase 2 width transfer\n(stars = argmin per width)")
        ax.grid(True, alpha=0.3)

        # ── Panel 3: argmin-LR vs width (the verdict plot) ────────────────
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
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("width (dim)")
        ax.set_ylabel("argmin LR")
        color = {"PASS": "green", "PASS*": "olive",
                 "WARN": "darkorange", "FAIL": "crimson",
                 "INSUFFICIENT": "gray"}.get(verdict, "black")
        ax.set_title(f"{mt} — verdict: {verdict}", color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)
        # Reference line: where a perfect-transfer optimum would sit
        if argmins:
            median = sorted(argmins.values())[len(argmins) // 2]
            ax.axhline(median, color="gray", linestyle="--", alpha=0.5,
                       label="perfect transfer")
            ax.legend()

    overall = ", ".join(f"{mt}: {v}" for mt, v, _ in overall_verdict)
    fig.suptitle(f"μP transfer check — {overall}", fontsize=14, fontweight="bold")
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
