#!/usr/bin/env python
"""
Plot comparison of 3 architectures across all eval categories.

Usage:
    python evals/plot_comparison.py
    python evals/plot_comparison.py --results file1.json file2.json file3.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Eval metadata ─────────────────────────────────────────────────────────

EVAL_META = {
    # MC benchmarks
    "hellaswag":       {"display": "HellaSwag",       "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "piqa":            {"display": "PIQA",             "metric": "accuracy",  "higher": True,  "random": 0.50,  "category": "MC Benchmarks"},
    "arc_easy":        {"display": "ARC-Easy",         "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "arc_challenge":   {"display": "ARC-Challenge",    "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "winogrande":      {"display": "WinoGrande",       "metric": "accuracy",  "higher": True,  "random": 0.50,  "category": "MC Benchmarks"},
    "boolq":           {"display": "BoolQ",            "metric": "accuracy",  "higher": True,  "random": 0.50,  "category": "MC Benchmarks"},
    "mmlu":            {"display": "MMLU",             "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    "truthfulqa":      {"display": "TruthfulQA",       "metric": "accuracy",  "higher": True,  "random": 0.25,  "category": "MC Benchmarks"},
    # Intrinsic
    "ppl":             {"display": "PPL (SlimPajama)",  "metric": "perplexity","higher": False, "random": None,  "category": "Intrinsic"},
    "wikitext_ppl":    {"display": "PPL (Wikitext)",    "metric": "perplexity","higher": False, "random": None,  "category": "Intrinsic"},
    "bpb":             {"display": "BPB (Wikitext)",    "metric": "bpb",       "higher": False, "random": None,  "category": "Intrinsic"},
    "wikitext_bpb":    {"display": "BPB (Wikitext)",    "metric": "bpb",       "higher": False, "random": None,  "category": "Intrinsic"},
    "token_accuracy":  {"display": "Token Accuracy",    "metric": "accuracy",  "higher": True,  "random": None,  "category": "Intrinsic"},
    # LAMBADA variants
    "lambada":         {"display": "LAMBADA (pretrain)","metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "LAMBADA"},
    "lambada_enc_dec": {"display": "LAMBADA (enc-dec)", "metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "LAMBADA"},
    # Probes
    "niah":            {"display": "Needle in Haystack","metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "Probes"},
    "copy_retrieval":  {"display": "Copy Retrieval",    "metric": "accuracy",  "higher": True,  "random": 0.00,  "category": "Probes"},
    # Generation
    "xsum":            {"display": "XSum",              "metric": "rouge1",    "higher": True,  "random": 0.00,  "category": "Generation"},
    "squad":           {"display": "SQuAD",             "metric": "f1",        "higher": True,  "random": 0.00,  "category": "Generation"},
    "triviaqa":        {"display": "TriviaQA",          "metric": "f1",        "higher": True,  "random": 0.00,  "category": "Generation"},
    "humaneval":       {"display": "HumanEval",         "metric": "pass_at_1", "higher": True,  "random": 0.00,  "category": "Generation"},
}


def _extract_score(result, meta):
    """Extract the primary score from an eval result dict."""
    if "error" in result:
        return None
    m = meta["metric"]
    if m in result:
        return result[m]
    # Fallback lookups
    if m == "accuracy" and "overall_accuracy" in result:
        return result["overall_accuracy"]
    if m == "rouge1" and "rouge1" in result:
        return result["rouge1"]
    return None


def load_results(path):
    with open(path) as f:
        return json.load(f)


COLORS = ["#4A90D9", "#E8833A", "#50B878"]
MODEL_MARKERS = ["o", "s", "D"]


def plot_category(ax, evals_in_cat, all_results, labels, category_name):
    """Plot a single category as grouped bars."""
    eval_names = [e for e in evals_in_cat if any(
        e in r.get("evals", {}) for r in all_results)]
    if not eval_names:
        return

    x = np.arange(len(eval_names))
    n_models = len(all_results)
    width = 0.8 / n_models

    for m_idx, (results, label) in enumerate(zip(all_results, labels)):
        scores = []
        for e in eval_names:
            meta = EVAL_META[e]
            r = results.get("evals", {}).get(e, {})
            s = _extract_score(r, meta)
            scores.append(s if s is not None else 0)

        offset = (m_idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=label,
                      color=COLORS[m_idx % len(COLORS)],
                      edgecolor="white", linewidth=0.5)

        # Value labels
        for bar, score in zip(bars, scores):
            if score > 0:
                h = bar.get_height()
                fontsize = 6.5 if len(eval_names) > 6 else 7.5
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{score:.3f}", ha="center", va="bottom",
                        fontsize=fontsize, color=COLORS[m_idx % len(COLORS)],
                        fontweight="bold", rotation=0)

    # Random chance lines
    for i, e in enumerate(eval_names):
        r = EVAL_META[e].get("random")
        if r is not None and r > 0:
            ax.plot([i - 0.45, i + 0.45], [r, r], color="gray",
                    linewidth=1, linestyle="--", alpha=0.4)

    display_names = [EVAL_META[e]["display"] for e in eval_names]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
    ax.set_title(category_name, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    # Add "higher/lower is better" indicator
    sample_meta = EVAL_META[eval_names[0]]
    direction = "Higher is better" if sample_meta["higher"] else "Lower is better"
    ax.text(0.98, 0.97, direction, transform=ax.transAxes, fontsize=7,
            ha="right", va="top", style="italic", color="gray")


def plot_comparison(result_files, labels, output_dir="evals"):
    all_results = [load_results(f) for f in result_files]

    # Print param counts
    print("\n  Model parameters:")
    for r, label in zip(all_results, labels):
        params = r.get("parameters", "?")
        if isinstance(params, (int, float)):
            print(f"    {label}: {params / 1e6:.1f}M")

    # Group evals by category
    categories = {}
    for e, meta in EVAL_META.items():
        cat = meta["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(e)

    # Create figure with subplots for each category
    cats_with_data = []
    for cat, evals in categories.items():
        has_data = any(e in r.get("evals", {}) for e in evals for r in all_results)
        if has_data:
            cats_with_data.append((cat, evals))

    n_cats = len(cats_with_data)
    fig, axes = plt.subplots(n_cats, 1, figsize=(14, 4.5 * n_cats))
    if n_cats == 1:
        axes = [axes]

    for ax, (cat, evals) in zip(axes, cats_with_data):
        plot_category(ax, evals, all_results, labels, cat)

    # Single legend at the top
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=len(labels),
               fontsize=11, bbox_to_anchor=(0.5, 1.02))

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
    args = parser.parse_args()

    assert len(args.results) == len(args.labels), "Must have same number of results and labels"

    # Check that result files exist
    missing = [f for f in args.results if not os.path.exists(f)]
    if missing:
        print(f"  Missing result files: {missing}")
        print(f"  Run evals first, then re-run this script.")
        return

    plot_comparison(args.results, args.labels, args.output_dir)


if __name__ == "__main__":
    main()
