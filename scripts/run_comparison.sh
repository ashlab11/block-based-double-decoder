#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 50M Architecture Comparison: Decoder-Only vs Double Decoder vs Std Enc-Dec
#
# Clone → run this script → come back to find all results and plots.
#
# What it does:
#   1. Installs dependencies (1_setup.sh)
#   2. Downloads & packs 6B tokens from SlimPajama (2_data.sh)
#   3. Trains 3 models at ~54M params on 1B tokens each (~20-30 min per model)
#   4. Evaluates all 3 models on 20+ benchmarks
#   5. Generates comparison plots and summary table
#
# Hardware: 1x H200 (or H100/A100 — auto batch size adapts)
# Total time: ~2-3 hours
# Total cost: ~$7-10 at $3.50/hr
#
# Usage:
#   bash scripts/run_comparison.sh
#   bash scripts/run_comparison.sh --skip-setup    # skip setup if already done
#   bash scripts/run_comparison.sh --skip-data      # skip data download
#   bash scripts/run_comparison.sh --eval-only      # skip training, just eval+plot
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_SETUP=false
SKIP_DATA=false
EVAL_ONLY=false
MAX_EVAL_EXAMPLES=500

for arg in "$@"; do
    case "$arg" in
        --skip-setup) SKIP_SETUP=true ;;
        --skip-data)  SKIP_DATA=true ;;
        --eval-only)  EVAL_ONLY=true ;;
    esac
done

CONFIGS=(
    "runs/compare_dec_only_50m"
    "runs/compare_dd_50m"
    "runs/compare_enc_dec_50m"
)
NAMES=(
    "compare_dec_only_50m"
    "compare_dd_50m"
    "compare_enc_dec_50m"
)
LABELS=(
    "Decoder-Only"
    "Double Decoder"
    "Std Enc-Dec"
)

EVAL_LIST="ppl,bpb,token_accuracy,positional_accuracy,wikitext_ppl,wikitext_bpb,hellaswag,piqa,arc_easy,arc_challenge,winogrande,boolq,mmlu,truthfulqa,lambada,lambada_enc_dec,glue,superglue,niah,copy_retrieval"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  50M Architecture Comparison"
echo "  Models: ${LABELS[*]}"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Setup ─────────────────────────────────────────────────────────
if [ "$SKIP_SETUP" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "── Step 1/5: Setup ──"
    bash scripts/1_setup.sh
else
    echo ""
    echo "── Step 1/5: Setup (skipped) ──"
fi

# ── Step 2: Data ──────────────────────────────────────────────────────────
if [ "$SKIP_DATA" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "── Step 2/5: Data ──"
    bash scripts/2_data.sh
else
    echo ""
    echo "── Step 2/5: Data (skipped) ──"
fi

# ── Step 3: Train ─────────────────────────────────────────────────────────
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "── Step 3/5: Training 3 models ──"
    for i in "${!CONFIGS[@]}"; do
        config="${CONFIGS[$i]}"
        name="${NAMES[$i]}"
        label="${LABELS[$i]}"
        checkpoint="checkpoints/${name}.pt"

        if [ -f "$checkpoint" ]; then
            echo ""
            echo "  [${label}] Checkpoint exists: ${checkpoint} — skipping training."
            continue
        fi

        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "  Training: ${label} (${config})"
        echo "────────────────────────────────────────────────────────────"
        torchrun --nproc_per_node=1 training/pretrain.py \
            --config-name="$config"
    done
else
    echo ""
    echo "── Step 3/5: Training (skipped — eval-only mode) ──"
fi

# ── Step 4: Evaluate ─────────────────────────────────────────────────────
echo ""
echo "── Step 4/5: Evaluating all models ──"
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    label="${LABELS[$i]}"
    checkpoint="checkpoints/${name}.pt"
    results="evals/results_${name}.json"

    if [ ! -f "$checkpoint" ]; then
        echo "  [${label}] No checkpoint at ${checkpoint} — skipping eval."
        continue
    fi

    if [ -f "$results" ]; then
        echo "  [${label}] Results already exist: ${results} — skipping eval."
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Evaluating: ${label}"
    echo "────────────────────────────────────────────────────────────"
    python evals/run_evals.py \
        --checkpoint "$checkpoint" \
        --evals "$EVAL_LIST" \
        --max-examples "$MAX_EVAL_EXAMPLES" \
        --output "$results"
done

# ── Step 5: Plot ──────────────────────────────────────────────────────────
echo ""
echo "── Step 5/5: Generating comparison plots ──"

# Collect result files and labels for models that have results
RESULT_FILES=()
RESULT_LABELS=()
for i in "${!NAMES[@]}"; do
    results="evals/results_${NAMES[$i]}.json"
    if [ -f "$results" ]; then
        RESULT_FILES+=("$results")
        RESULT_LABELS+=("${LABELS[$i]}")
    fi
done

if [ ${#RESULT_FILES[@]} -ge 2 ]; then
    python evals/plot_comparison.py \
        --results "${RESULT_FILES[@]}" \
        --labels "${RESULT_LABELS[@]}"
else
    echo "  Need at least 2 result files to plot comparison."
    echo "  Found: ${RESULT_FILES[*]:-none}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Comparison complete!"
echo ""
echo "  Results:"
for i in "${!NAMES[@]}"; do
    results="evals/results_${NAMES[$i]}.json"
    [ -f "$results" ] && echo "    ${LABELS[$i]}: ${results}"
done
echo ""
echo "  Plot: evals/architecture_comparison.png"
echo "═══════════════════════════════════════════════════════════════"
