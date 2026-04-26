#!/usr/bin/env python
"""
Plot comparison of 3 architectures across all eval categories.

Fixes addressed:
  - Intrinsic metrics with different scales get separate subplots (PPL vs BPB vs Accuracy)
  - Random guess baselines shown as dashed lines on all MC benchmarks
  - NIAH/Copy Retrieval: bars at 0.0 are explicitly shown (not invisible)
  - bpb and wikitext_bpb are deduplicated (they're identical)
  - Value labels adapt to y-scale

Usage:
    python evals/plot_comparison.py
    python evals/plot_comparison.py --results file1.json file2.json file3.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Eval metadata ─────────────────────────────────────────────────────────

EVAL_META = {
    # MC benchmarks
    "hellaswag":       {"display": "HellaSwag",        "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "piqa":            {"display": "PIQA",              "metric": "accuracy",  "higher": True,  "random": 0.50,  "category": "MC Benchmarks"},
    "arc_easy":        {"display": "ARC-Easy",          "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "arc_challenge":   {"display": "ARC-Challenge",     "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "winogrande":      {"display": "WinoGrande",        "metric": "accuracy",  "higher": True,  "random": 0.50,  "category": "MC Benchmarks"},
    "boolq":           {"display": "BoolQ",             "metric": "accuracy",  "higher": True,  "random": 0.50,  "category": "MC Benchmarks"},
    "mmlu":            {"display": "MMLU",              "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "truthfulqa":      {"display": "TruthfulQA",        "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    # Intrinsic — split into separate sub-categories by scale
    "ppl":             {"display": "PPL (SlimPajama)",   "metric": "perplexity","higher": False, "random": None,  "category": "Perplexity"},
    "wikitext_ppl":    {"display": "PPL (Wikitext-103)", "metric": "perplexity","higher": False, "random": None,  "category": "Perplexity"},
    "bpb":             {"display": "Bits-per-Byte",      "metric": "bpb",       "higher": False, "random": None,  "category": "BPB & Token Accuracy"},
    "token_accuracy":  {"display": "Token Accuracy",     "metric": "accuracy",  "higher": True,  "random": None,  "category": "BPB & Token Accuracy"},
    # LAMBADA
    "lambada":         {"display": "LAMBADA (pretrain)", "metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "LAMBADA & Probes"},
    "lambada_enc_dec": {"display": "LAMBADA (enc-dec)",  "metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "LAMBADA & Probes"},
    # Probes
    "niah":            {"display": "Needle in Haystack", "metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "LAMBADA & Probes"},
    "copy_retrieval":  {"display": "Copy Retrieval",     "metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "LAMBADA & Probes"},
    # Generation (asymmetric tasks — Elfeki's encoder-advantaging category)
    "xsum":            {"display": "XSum (ROUGE-1)",     "metric": "rouge1",    "higher": True,  "random": 0.00,  "category": "Generation (Asymmetric Tasks)"},
    "squad":           {"display": "SQuAD (F1)",         "metric": "f1",        "higher": True,  "random": 0.00,  "category": "Generation (Asymmetric Tasks)"},
    "triviaqa":        {"display": "TriviaQA (F1)",      "metric": "f1",        "higher": True,  "random": 0.00,  "category": "Generation (Asymmetric Tasks)"},
}

# wikitext_bpb intentionally excluded — identical to bpb (both use Wikitext-103)

COLORS = ["#4A90D9", "#E8833A", "#50B878"]


def _extract_score(result, meta):
    """Extract the primary score from an eval result dict."""
    if "error" in result:
        return None
    m = meta["metric"]
    if m in result:
        return result[m]
    if m == "accuracy" and "overall_accuracy" in result:
        return result["overall_accuracy"]
    return None


def load_results(path):
    with open(path) as f:
        return json.load(f)


def _plot_grouped_bars(ax, eval_names, all_results, labels, category_name,
                       show_random=True, y_pad_frac=0.15, force_zero_base=True):
    """Plot grouped bars for a list of evals on a single axes."""
    # Filter to evals that have data
    active_evals = [e for e in eval_names if any(
        e in r.get("evals", {}) for r in all_results)]
    if not active_evals:
        return

    x = np.arange(len(active_evals))
    n_models = len(all_results)
    width = 0.8 / n_models

    all_scores = []  # for y-limit calculation

    for m_idx, (results, label) in enumerate(zip(all_results, labels)):
        scores = []
        for e in active_evals:
            meta = EVAL_META[e]
            r = results.get("evals", {}).get(e, {})
            s = _extract_score(r, meta)
            scores.append(s if s is not None else 0)
            if s is not None:
                all_scores.append(s)

        offset = (m_idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=label,
                      color=COLORS[m_idx % len(COLORS)],
                      edgecolor="white", linewidth=0.5, zorder=3)

        # Value labels — adapt format to magnitude
        max_val = max(all_scores) if all_scores else 1
        for bar, score in zip(bars, scores):
            h = bar.get_height()
            if max_val > 10:
                txt = f"{score:.1f}"
            elif max_val > 1:
                txt = f"{score:.3f}"
            else:
                txt = f"{score:.3f}"
            y_offset = max_val * 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, h + y_offset,
                    txt, ha="center", va="bottom",
                    fontsize=6.5, color=COLORS[m_idx % len(COLORS)],
                    fontweight="bold")

    # Random chance baselines
    if show_random:
        drew_random = False
        for i, e in enumerate(active_evals):
            r = EVAL_META[e].get("random")
            if r is not None:
                ax.plot([i - 0.45, i + 0.45], [r, r], color="red",
                        linewidth=1.2, linestyle="--", alpha=0.5, zorder=2)
                drew_random = True
        if drew_random:
            ax.plot([], [], color="red", linewidth=1.2, linestyle="--",
                    alpha=0.5, label="Random guess")

    # Labels and formatting
    display_names = [EVAL_META[e]["display"] for e in active_evals]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=25, ha="right", fontsize=9)
    ax.set_title(category_name, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.2, zorder=0)

    # Direction indicator
    sample_meta = EVAL_META[active_evals[0]]
    # Check if mixed directions in this category
    directions = set(EVAL_META[e]["higher"] for e in active_evals)
    if len(directions) == 1:
        direction = "Higher is better" if sample_meta["higher"] else "Lower is better"
    else:
        direction = "Mixed (see labels)"
    ax.text(0.98, 0.97, direction, transform=ax.transAxes, fontsize=7,
            ha="right", va="top", style="italic", color="gray")

    # Y-axis: ensure bars are visible even at 0
    if all_scores:
        ymax = max(all_scores) * (1 + y_pad_frac)
        if force_zero_base:
            ax.set_ylim(0, ymax)
        else:
            ymin = min(all_scores) * 0.85
            ax.set_ylim(ymin, ymax)


def plot_comparison(result_files, labels, output_dir="evals"):
    all_results = [load_results(f) for f in result_files]

    # Print param counts
    print("\n  Model parameters:")
    for r, label in zip(all_results, labels):
        params = r.get("parameters", "?")
        if isinstance(params, (int, float)):
            print(f"    {label}: {params / 1e6:.1f}M")

    # Build ordered category list (preserving definition order)
    seen = {}
    for e, meta in EVAL_META.items():
        cat = meta["category"]
        if cat not in seen:
            seen[cat] = []
        seen[cat].append(e)

    # Filter to categories with data
    cats_with_data = []
    for cat, evals in seen.items():
        has_data = any(e in r.get("evals", {}) for e in evals for r in all_results)
        if has_data:
            cats_with_data.append((cat, evals))

    n_plots = len(cats_with_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4.0 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, (cat, evals) in zip(axes, cats_with_data):
        show_random = cat in ("MC Benchmarks", "LAMBADA & Probes")
        _plot_grouped_bars(ax, evals, all_results, labels, cat,
                           show_random=show_random)

    # Single legend at the top — collect from first axes that has random line
    all_handles = []
    all_labels_leg = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in all_labels_leg:
                all_handles.append(hi)
                all_labels_leg.append(li)
    fig.legend(all_handles, all_labels_leg, loc="upper center",
               ncol=len(all_labels_leg), fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, "architecture_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n  {'Eval':<25s}  {'Metric':<12s}  {'Better':<8s}  {'Random':<8s}", end="")
    for label in labels:
        print(f"  {label:<18s}", end="")
    print()
    print(f"  {'─'*25}  {'─'*12}  {'─'*8}  {'─'*8}", end="")
    for _ in labels:
        print(f"  {'─'*18}", end="")
    print()

    for cat, evals in cats_with_data:
        for e in evals:
            meta = EVAL_META[e]
            has_any = any(e in r.get("evals", {}) for r in all_results)
            if not has_any:
                continue
            direction = "Higher" if meta["higher"] else "Lower"
            rand_str = f"{meta['random']:.2f}" if meta["random"] is not None else "—"
            print(f"  {meta['display']:<25s}  {meta['metric']:<12s}  {direction:<8s}  {rand_str:<8s}", end="")
            for r in all_results:
                result = r.get("evals", {}).get(e, {})
                score = _extract_score(result, meta)
                if score is not None:
                    print(f"  {score:<18.4f}", end="")
                else:
                    print(f"  {'—':<18s}", end="")
            print()


def plot_efficiency(efficiency_path, output_dir="evals"):
    """Plot efficiency metrics as grouped bars from benchmark_efficiency.py output."""
    with open(efficiency_path) as f:
        data = json.load(f)

    labels = list(data.keys())
    n = len(labels)

    metrics = [
        ("Throughput (tok/s)", [data[l]["throughput"]["mean_tok_per_sec"] for l in labels],
         True, "tok/s"),
        ("First-Token Latency (ms)", [data[l]["first_token_latency"][
            str(max(int(k) for k in data[l]["first_token_latency"]))]["mean_ms"] for l in labels],
         False, "ms"),
        ("Peak Memory (MB)", [data[l]["peak_memory_mb"] for l in labels],
         False, "MB"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (title, values, higher_better, unit) in zip(axes, metrics):
        x = np.arange(n)
        bars = ax.bar(x, values, color=COLORS[:n], edgecolor="white", linewidth=0.5, width=0.6)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(axis="y", alpha=0.2)

        direction = "Higher is better" if higher_better else "Lower is better"
        ax.text(0.98, 0.97, direction, transform=ax.transAxes, fontsize=7,
                ha="right", va="top", style="italic", color="gray")

        # Highlight the best
        best_idx = values.index(max(values)) if higher_better else values.index(min(values))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

    fig.suptitle("Inference Efficiency Comparison (54.5M params, H200 GPU)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "efficiency_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+",
                        default=[
                            "evals/results_compare_dec_only_50m.json",
                            "evals/results_compare_dd_50m.json",
                            "evals/results_compare_enc_dec_50m.json",
                        ])
    parser.add_argument("--labels", nargs="+",
                        default=["Decoder-Only", "Double Decoder", "Std Enc-Dec"])
    parser.add_argument("--output-dir", default="evals")
    parser.add_argument("--efficiency", default=None,
                        help="Path to efficiency_results.json (generates efficiency plot)")
    args = parser.parse_args()

    assert len(args.results) == len(args.labels), "Must have same number of results and labels"

    missing = [f for f in args.results if not os.path.exists(f)]
    if missing:
        print(f"  Missing result files: {missing}")
        print(f"  Run evals first, then re-run this script.")
        return

    plot_comparison(args.results, args.labels, args.output_dir)

    # Efficiency plot
    eff_path = args.efficiency
    if eff_path is None:
        eff_path = os.path.join(args.output_dir, "efficiency_results.json")
    if os.path.exists(eff_path):
        print("\n── Efficiency Plot ──")
        plot_efficiency(eff_path, args.output_dir)


if __name__ == "__main__":
    main()
