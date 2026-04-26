#!/usr/bin/env python
"""
Generate two charts:
  1. Benchmark accuracy: DD 50M vs Pythia-70M
  2. Training cost to Chinchilla-optimal vs model size (10M–500M)

Usage:
    python evals/plot_charts.py
    python evals/plot_charts.py --h200-price 3.50  # override $/hr
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ═══════════════════════════════════════════════════════════════════════════
#  Chart 1: Benchmark comparison
# ═══════════════════════════════════════════════════════════════════════════

# DD 50M results (54.5M params, 1B tokens on SlimPajama)
DD_50M = {
    "HellaSwag":     0.284,
    "PIQA":          0.542,
    "ARC-Easy":      0.296,
    "ARC-Challenge": 0.196,
    "WinoGrande":    0.516,
    "BoolQ":         0.370,
    "MMLU":          0.254,
    "TruthfulQA":    0.308,
    "LAMBADA":       0.022,
}

# Pythia-70M (0-shot, from EleutherAI eval harness / published results)
# Trained on The Pile, 300B tokens — significantly overtrained vs Chinchilla
PYTHIA_70M = {
    "HellaSwag":     0.271,
    "PIQA":          0.593,
    "ARC-Easy":      0.382,
    "ARC-Challenge": 0.208,
    "WinoGrande":    0.502,
    "BoolQ":         0.571,
    "MMLU":          0.252,
    "TruthfulQA":    0.229,
    "LAMBADA":       0.325,
}

# Random baselines for reference
RANDOM = {
    "HellaSwag":     0.25,
    "PIQA":          0.50,
    "ARC-Easy":      0.25,
    "ARC-Challenge": 0.25,
    "WinoGrande":    0.50,
    "BoolQ":         0.50,
    "MMLU":          0.25,
    "TruthfulQA":    None,  # variable number of choices
    "LAMBADA":       0.00,
}


def plot_benchmark_comparison(output_path="evals/benchmark_comparison.png"):
    benchmarks = list(DD_50M.keys())
    dd_scores = [DD_50M[b] for b in benchmarks]
    pythia_scores = [PYTHIA_70M[b] for b in benchmarks]
    random_scores = [RANDOM[b] for b in benchmarks]

    x = np.arange(len(benchmarks))
    width = 0.32

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_dd = ax.bar(x - width / 2, dd_scores, width, label="DD 50M (54.5M, 1B tok)",
                     color="#4A90D9", edgecolor="white", linewidth=0.5)
    bars_py = ax.bar(x + width / 2, pythia_scores, width, label="Pythia-70M (70M, 300B tok)",
                     color="#E8833A", edgecolor="white", linewidth=0.5)

    # Random chance line per benchmark
    for i, r in enumerate(random_scores):
        if r is not None:
            ax.plot([i - 0.45, i + 0.45], [r, r], color="gray", linewidth=1,
                    linestyle="--", alpha=0.5)
    # Single legend entry for random
    ax.plot([], [], color="gray", linewidth=1, linestyle="--", alpha=0.5, label="Random chance")

    # Value labels on bars
    for bar in bars_dd:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#4A90D9", fontweight="bold")
    for bar in bars_py:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#E8833A", fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Benchmark Comparison: Double Decoder 50M vs Pythia-70M (0-shot)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 0.72)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Footnote
    fig.text(0.5, -0.02,
             "Note: Pythia-70M is 30% larger and trained on 300× more tokens (300B vs 1B). "
             "DD 50M is Chinchilla-optimal (20 tok/param).",
             ha="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Chart 2: Training cost vs model size
# ═══════════════════════════════════════════════════════════════════════════

# H200 specs
H200_BF16_TFLOPS = 990  # peak BF16 TFLOPS

def _mfu_curve(n_params):
    """Estimate MFU (model FLOPS utilization) as a function of model size.

    Small models are memory-bandwidth-bound and don't saturate GPU compute.
    Larger models achieve better utilization.

    Rough calibration:
        10M  → ~15%    (heavily bandwidth-bound)
        50M  → ~20%    (matches DD 50M training throughput estimate)
        200M → ~32%
        500M → ~40%    (approaching good utilization for H200)
    """
    # Log-linear interpolation between anchor points
    log_n = np.log10(n_params)
    # Clamp to [10M, 500M] range
    log_lo, log_hi = np.log10(10e6), np.log10(500e6)
    t = np.clip((log_n - log_lo) / (log_hi - log_lo), 0, 1)
    return 0.15 + 0.25 * t


def compute_training_cost(n_params, h200_price_per_hr):
    """Compute H200 hours and $ cost for Chinchilla-optimal training."""
    tokens = 20 * n_params                          # Chinchilla: ~20 tok/param
    flops = 6 * n_params * tokens                   # 6ND approximation
    mfu = _mfu_curve(n_params)
    effective_tflops = H200_BF16_TFLOPS * mfu       # practical throughput
    seconds = flops / (effective_tflops * 1e12)      # TFLOPS = 1e12 FLOPS/s
    hours = seconds / 3600
    cost = hours * h200_price_per_hr
    return hours, cost, tokens, mfu


def plot_training_cost(h200_price=3.50, output_path="evals/training_cost.png"):
    # 15 log-spaced points from 10M to 500M
    param_counts = np.logspace(np.log10(10e6), np.log10(500e6), 15)

    hours_list = []
    cost_list = []
    tokens_list = []
    mfu_list = []

    for n in param_counts:
        h, c, t, m = compute_training_cost(n, h200_price)
        hours_list.append(h)
        cost_list.append(c)
        tokens_list.append(t)
        mfu_list.append(m)

    hours_arr = np.array(hours_list)
    cost_arr = np.array(cost_list)

    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    # H200 hours (left y-axis)
    color_hours = "#2E86AB"
    line1, = ax1.plot(param_counts / 1e6, hours_arr, "o-", color=color_hours,
                      linewidth=2.5, markersize=6, label="H200 hours", zorder=3)
    ax1.set_xlabel("Model Parameters (millions)", fontsize=12)
    ax1.set_ylabel("H200 GPU-Hours", fontsize=12, color=color_hours)
    ax1.tick_params(axis="y", labelcolor=color_hours)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Format x-axis with clean labels
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax1.set_xticks([10, 20, 50, 100, 200, 500])
    ax1.set_xticklabels(["10M", "20M", "50M", "100M", "200M", "500M"])

    # Cost (right y-axis)
    ax2 = ax1.twinx()
    color_cost = "#A23B72"
    line2, = ax2.plot(param_counts / 1e6, cost_arr, "s--", color=color_cost,
                      linewidth=2.5, markersize=6, label=f"Cost (${h200_price:.2f}/hr)", zorder=3)
    ax2.set_ylabel(f"Cost (USD, @ ${h200_price:.2f}/hr H200)", fontsize=12, color=color_cost)
    ax2.tick_params(axis="y", labelcolor=color_cost)
    ax2.set_yscale("log")

    # Format cost axis
    def _fmt_cost(x, _):
        if x < 1:
            return f"${x:.2f}"
        elif x < 100:
            return f"${x:.1f}"
        elif x < 1000:
            return f"${x:.0f}"
        else:
            return f"${x / 1000:.1f}K"
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_cost))

    # Format hours axis
    def _fmt_hours(x, _):
        if x < 1:
            return f"{x * 60:.0f}m"
        elif x < 100:
            return f"{x:.1f}h"
        else:
            return f"{x:.0f}h"
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_hours))

    # Annotate key points
    key_sizes = [50e6, 300e6]
    for ks in key_sizes:
        h, c, t, m = compute_training_cost(ks, h200_price)
        label = f"{ks/1e6:.0f}M"
        tok_label = f"{t/1e9:.1f}B tok" if t >= 1e9 else f"{t/1e6:.0f}M tok"
        ax1.annotate(
            f"{label}\n{_fmt_hours(h, None)} · {_fmt_cost(c, None)}\n{tok_label} · MFU {m:.0%}",
            xy=(ks / 1e6, h), xytext=(15, 25),
            textcoords="offset points", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            ha="left", zorder=5,
        )

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    ax1.set_title("Training Cost to Chinchilla-Optimal (1× H200 GPU)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.2)

    # Footnote
    fig.text(0.5, -0.02,
             "Chinchilla-optimal ≈ 20 tokens/param. FLOPs ≈ 6ND. "
             f"MFU interpolated 15%–40% (small → large). H200 @ {H200_BF16_TFLOPS} BF16 TFLOPS peak.",
             ha="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # Print table for reference
    print(f"\n  {'Params':>10s}  {'Tokens':>10s}  {'MFU':>5s}  {'H200-hrs':>10s}  {'Cost':>10s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*10}  {'─'*10}")
    for i, n in enumerate(param_counts):
        t = tokens_list[i]
        tok_str = f"{t/1e9:.2f}B" if t >= 1e9 else f"{t/1e6:.0f}M"
        cost_str = f"${cost_list[i]:.2f}" if cost_list[i] < 1000 else f"${cost_list[i]/1000:.2f}K"
        hrs_str = f"{hours_list[i]:.2f}" if hours_list[i] >= 1 else f"{hours_list[i]*60:.1f}m"
        print(f"  {n/1e6:>8.1f}M  {tok_str:>10s}  {mfu_list[i]:>4.0%}  {hrs_str:>10s}  {cost_str:>10s}")


# ═══════════════════════════════════════════════════════════════════════════
#  Chart 3: Chinchilla-optimal token count vs model size
# ═══════════════════════════════════════════════════════════════════════════

def plot_chinchilla_tokens(output_path="evals/chinchilla_tokens.png"):
    param_counts = np.logspace(np.log10(10e6), np.log10(500e6), 15)
    token_counts = 20 * param_counts  # Chinchilla: ~20 tokens/param

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(param_counts / 1e6, token_counts / 1e9, "o-",
            color="#2E86AB", linewidth=2.5, markersize=7, zorder=3)

    # Fill the region to visualize the scaling
    ax.fill_between(param_counts / 1e6, token_counts / 1e9,
                     alpha=0.08, color="#2E86AB")

    ax.set_xlabel("Model Parameters (millions)", fontsize=12)
    ax.set_ylabel("Chinchilla-Optimal Token Count (billions)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Clean x-axis labels
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(["10M", "20M", "50M", "100M", "200M", "500M"])

    # Clean y-axis labels
    def _fmt_tokens(x, _):
        if x < 1:
            return f"{x * 1000:.0f}M"
        else:
            return f"{x:.1f}B"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tokens))

    # Annotate key model sizes
    annotations = [
        (50e6,  "DD 50M\n(your current run)"),
        (300e6, "DD 300M\n(your next run)"),
    ]
    for n, label in annotations:
        t = 20 * n
        ax.plot(n / 1e6, t / 1e9, "D", color="#E8833A", markersize=10, zorder=4)
        tok_str = f"{t/1e9:.1f}B" if t >= 1e9 else f"{t/1e6:.0f}M"
        ax.annotate(
            f"{label}\n{tok_str} tokens",
            xy=(n / 1e6, t / 1e9), xytext=(20, 25),
            textcoords="offset points", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            ha="left", zorder=5,
        )

    # Reference lines for well-known models
    ref_models = [
        (70e6,  300e9,  "Pythia-70M\n(300B, 4286× overtrained)"),
    ]
    for n, t, label in ref_models:
        ax.plot(n / 1e6, t / 1e9, "x", color="#A23B72", markersize=10,
                markeredgewidth=2.5, zorder=4)
        ax.annotate(
            label,
            xy=(n / 1e6, t / 1e9), xytext=(-15, -40),
            textcoords="offset points", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5e6f0",
                      edgecolor="#A23B72", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#A23B72", lw=1),
            ha="center", zorder=5,
        )

    ax.set_title("Chinchilla-Optimal Training Tokens vs Model Size",
                  fontsize=13, fontweight="bold")
    ax.grid(True, which="both", alpha=0.2)

    fig.text(0.5, -0.02,
             "Chinchilla scaling law: optimal tokens ≈ 20 × parameters. "
             "Points above the line are overtrained; below are undertrained.",
             ha="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h200-price", type=float, default=3.50,
                        help="H200 price per hour in USD (default: $3.50)")
    parser.add_argument("--output-dir", type=str, default="evals")
    args = parser.parse_args()

    print("\n── Chart 1: Benchmark Comparison ──")
    plot_benchmark_comparison(f"{args.output_dir}/benchmark_comparison.png")

    print("\n── Chart 2: Training Cost ──")
    plot_training_cost(args.h200_price, f"{args.output_dir}/training_cost.png")

    print("\n── Chart 3: Chinchilla-Optimal Tokens ──")
    plot_chinchilla_tokens(f"{args.output_dir}/chinchilla_tokens.png")
    print()


if __name__ == "__main__":
    main()
