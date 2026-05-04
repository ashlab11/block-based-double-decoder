#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: SFT + Post-Training Evaluation + Efficiency Benchmarks
#
# Prerequisite: run_comparison.sh must have completed (pretrained checkpoints exist)
#
# What it does:
#   1. Downloads UltraChat SFT data (~50M tokens)
#   2. SFTs all 3 pretrained models on UltraChat
#   3. Evaluates SFT'd models on encoder-advantaging tasks (SQuAD, XSum, TriviaQA)
#      plus all MC benchmarks and LAMBADA enc-dec
#   4. Runs inference efficiency benchmarks (first-token latency, throughput, memory)
#   5. Generates comparison plots (pretrain vs SFT)
#
# Hardware: 1x H200 (or H100/A100)
# Total time: ~1-2 hours
# Total cost: ~$4-7 at $3.50/hr
#
# Usage:
#   bash scripts/9_run_sft_and_eval.sh
#   bash scripts/9_run_sft_and_eval.sh --skip-data
#   bash scripts/9_run_sft_and_eval.sh --eval-only
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

SKIP_DATA=false
EVAL_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-data)  SKIP_DATA=true ;;
        --eval-only)  EVAL_ONLY=true ;;
    esac
done

SFT_CONFIGS=(
    "runs/sft_dec_only_50m"
    "runs/sft_dd_50m"
    "runs/sft_enc_dec_50m"
)
PRETRAIN_CKPTS=(
    "checkpoints/compare_dec_only_50m.pt"
    "checkpoints/compare_dd_50m.pt"
    "checkpoints/compare_enc_dec_50m.pt"
)
SFT_NAMES=(
    "sft_dec_only_50m"
    "sft_dd_50m"
    "sft_enc_dec_50m"
)
LABELS=(
    "Decoder-Only"
    "Double Decoder"
    "Std Enc-Dec"
)

# Encoder-advantaging evals: tasks with clear input→output asymmetry
# Plus MC benchmarks for comparison and LAMBADA enc-dec
EVAL_LIST="ppl,bpb,token_accuracy,wikitext_ppl,hellaswag,piqa,arc_easy,arc_challenge,winogrande,boolq,mmlu,truthfulqa,lambada,lambada_enc_dec,xsum,squad,triviaqa,copy_retrieval"
MAX_EVAL_EXAMPLES=500

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 2: SFT + Post-Training Evaluation"
echo "  Models: ${LABELS[*]}"
echo "═══════════════════════════════════════════════════════════════"

# ── Verify pretrained checkpoints exist ───────────────────────────────────
echo ""
echo "── Checking pretrained checkpoints ──"
for i in "${!PRETRAIN_CKPTS[@]}"; do
    ckpt="${PRETRAIN_CKPTS[$i]}"
    label="${LABELS[$i]}"
    if [ ! -f "$ckpt" ]; then
        echo "  MISSING: ${ckpt} (${label})"
        echo "  Run scripts/run_comparison.sh first to pretrain models."
        exit 1
    fi
    echo "  ✓ ${label}: ${ckpt}"
done

# ── Step 1: SFT Data ─────────────────────────────────────────────────────
if [ "$SKIP_DATA" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "── Step 1/4: Downloading UltraChat SFT data ──"
    if [ -f "data/SFT/ultrachat.jsonl" ]; then
        echo "  SFT data already exists, skipping."
    else
        python data/retrieval_scripts/ultrachat.py
    fi
else
    echo ""
    echo "── Step 1/4: SFT Data (skipped) ──"
fi

# ── Step 2: SFT Training ─────────────────────────────────────────────────
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "── Step 2/4: SFT training on UltraChat ──"
    for i in "${!SFT_CONFIGS[@]}"; do
        config="${SFT_CONFIGS[$i]}"
        name="${SFT_NAMES[$i]}"
        label="${LABELS[$i]}"
        checkpoint="checkpoints/${name}.pt"

        if [ -f "$checkpoint" ]; then
            echo ""
            echo "  [${label}] SFT checkpoint exists: ${checkpoint} — skipping."
            continue
        fi

        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "  SFT Training: ${label} (${config})"
        echo "────────────────────────────────────────────────────────────"
        torchrun --nproc_per_node=1 training/sft.py \
            --config-name="$config"
    done
else
    echo ""
    echo "── Step 2/4: SFT Training (skipped — eval-only mode) ──"
fi

# ── Step 3: Evaluate SFT'd models ────────────────────────────────────────
echo ""
echo "── Step 3/4: Evaluating SFT'd models ──"
for i in "${!SFT_NAMES[@]}"; do
    name="${SFT_NAMES[$i]}"
    label="${LABELS[$i]}"
    checkpoint="checkpoints/${name}.pt"
    results="evals/results_${name}.json"

    if [ ! -f "$checkpoint" ]; then
        echo "  [${label}] No SFT checkpoint at ${checkpoint} — skipping eval."
        continue
    fi

    if [ -f "$results" ]; then
        echo "  [${label}] Results already exist: ${results} — skipping eval."
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Evaluating: ${label} (SFT)"
    echo "────────────────────────────────────────────────────────────"
    python evals/run_evals.py \
        --checkpoint "$checkpoint" \
        --evals "$EVAL_LIST" \
        --max-examples "$MAX_EVAL_EXAMPLES" \
        --output "$results"
done

# ── Step 4: Efficiency benchmarks ────────────────────────────────────────
echo ""
echo "── Step 4/4: Inference efficiency benchmarks ──"

# Use SFT checkpoints if available, fall back to pretrained
BENCH_CKPTS=()
BENCH_LABELS=()
for i in "${!SFT_NAMES[@]}"; do
    sft_ckpt="checkpoints/${SFT_NAMES[$i]}.pt"
    pt_ckpt="${PRETRAIN_CKPTS[$i]}"
    if [ -f "$sft_ckpt" ]; then
        BENCH_CKPTS+=("$sft_ckpt")
    elif [ -f "$pt_ckpt" ]; then
        BENCH_CKPTS+=("$pt_ckpt")
    else
        continue
    fi
    BENCH_LABELS+=("${LABELS[$i]}")
done

if [ ${#BENCH_CKPTS[@]} -ge 2 ]; then
    python evals/benchmark_efficiency.py \
        --checkpoints "${BENCH_CKPTS[@]}" \
        --labels "${BENCH_LABELS[@]}" \
        --output evals/efficiency_results.json
else
    echo "  Need at least 2 checkpoints for efficiency benchmarks."
fi

# ── Generate comparison plots ────────────────────────────────────────────
echo ""
echo "── Generating SFT comparison plots ──"

SFT_RESULT_FILES=()
SFT_RESULT_LABELS=()
for i in "${!SFT_NAMES[@]}"; do
    results="evals/results_${SFT_NAMES[$i]}.json"
    if [ -f "$results" ]; then
        SFT_RESULT_FILES+=("$results")
        SFT_RESULT_LABELS+=("${LABELS[$i]} (SFT)")
    fi
done

if [ ${#SFT_RESULT_FILES[@]} -ge 2 ]; then
    python evals/plot_comparison.py \
        --results "${SFT_RESULT_FILES[@]}" \
        --labels "${SFT_RESULT_LABELS[@]}" \
        --output-dir evals
    # Rename to avoid overwriting pretrain comparison
    mv evals/architecture_comparison.png evals/sft_comparison.png 2>/dev/null || true
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 2 complete!"
echo ""
echo "  SFT Results:"
for i in "${!SFT_NAMES[@]}"; do
    results="evals/results_${SFT_NAMES[$i]}.json"
    [ -f "$results" ] && echo "    ${LABELS[$i]}: ${results}"
done
echo ""
echo "  Efficiency: evals/efficiency_results.json"
echo "  SFT Plot:   evals/sft_comparison.png"
echo "═══════════════════════════════════════════════════════════════"
