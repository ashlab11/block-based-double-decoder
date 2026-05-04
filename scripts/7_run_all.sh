#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline: Pretrain → Eval → SFT → Eval → Efficiency → Plots
#
# Runs everything end-to-end:
#   Phase 1: Setup, data download, pretrain 3 architectures, eval, plot
#   Phase 2: SFT data, SFT all 3, eval SFT'd models, efficiency benchmarks, plots
#   Final:   Collect all outputs into evals/ for easy SCP
#
# Hardware: 1x H200 (or H100/A100)
# Total time: ~3-5 hours
# Total cost: ~$10-17 at $3.50/hr
#
# Usage:
#   bash scripts/7_run_all.sh                    # full run from scratch
#   bash scripts/7_run_all.sh --skip-setup       # skip pip install
#   bash scripts/7_run_all.sh --skip-data        # skip data download
#   bash scripts/7_run_all.sh --phase2-only      # skip pretrain, run SFT+eval only
#   bash scripts/7_run_all.sh --eval-only        # skip all training, just eval+plot
#
# After completion, SCP all results:
#   scp -P <port> -i ~/.ssh/id_ed25519 -r root@<host>:block-based-double-decoder/evals/ ~/Desktop/evals/
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

SKIP_SETUP=false
SKIP_DATA=false
PHASE2_ONLY=false
EVAL_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-setup)   SKIP_SETUP=true ;;
        --skip-data)    SKIP_DATA=true ;;
        --phase2-only)  PHASE2_ONLY=true ;;
        --eval-only)    EVAL_ONLY=true ;;
    esac
done

START_TIME=$(date +%s)

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Full Pipeline: Pretrain → SFT → Eval → Efficiency → Plots  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "  Started: $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: Pretrain + Eval
# ═══════════════════════════════════════════════════════════════════════════

if [ "$PHASE2_ONLY" = false ]; then
    echo "┌───────────────────────────────────────────────────────────────┐"
    echo "│  PHASE 1: Pretrain 3 Architectures + Eval                    │"
    echo "└───────────────────────────────────────────────────────────────┘"

    PHASE1_ARGS=()
    [ "$SKIP_SETUP" = true ] && PHASE1_ARGS+=("--skip-setup")
    [ "$SKIP_DATA" = true ]  && PHASE1_ARGS+=("--skip-data")
    [ "$EVAL_ONLY" = true ]  && PHASE1_ARGS+=("--eval-only")

    bash scripts/run_comparison.sh "${PHASE1_ARGS[@]}"

    # Rename pretrain plot to distinguish from SFT plot
    if [ -f evals/architecture_comparison.png ]; then
        cp evals/architecture_comparison.png evals/pretrain_comparison.png
        echo "  Saved pretrain plot: evals/pretrain_comparison.png"
    fi

    PHASE1_TIME=$(date +%s)
    echo ""
    echo "  Phase 1 completed in $(( (PHASE1_TIME - START_TIME) / 60 )) minutes."
    echo ""
else
    echo "── Phase 1: Pretrain (skipped — phase2-only mode) ──"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: SFT + Eval + Efficiency
# ═══════════════════════════════════════════════════════════════════════════

echo "┌───────────────────────────────────────────────────────────────┐"
echo "│  PHASE 2: SFT + Post-Training Eval + Efficiency Benchmarks   │"
echo "└───────────────────────────────────────────────────────────────┘"

PHASE2_ARGS=()
[ "$SKIP_DATA" = true ]  && PHASE2_ARGS+=("--skip-data")
[ "$EVAL_ONLY" = true ]  && PHASE2_ARGS+=("--eval-only")

bash scripts/run_sft_and_eval.sh "${PHASE2_ARGS[@]}"

PHASE2_TIME=$(date +%s)
echo ""
echo "  Phase 2 completed in $(( (PHASE2_TIME - ${PHASE1_TIME:-$START_TIME}) / 60 )) minutes."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
#  FINAL: Generate all plots (re-run to ensure latest data)
# ═══════════════════════════════════════════════════════════════════════════

echo "┌───────────────────────────────────────────────────────────────┐"
echo "│  FINAL: Generating All Plots                                  │"
echo "└───────────────────────────────────────────────────────────────┘"

# Pretrain comparison plot
PT_FILES=()
PT_LABELS=()
for name_label in "compare_dec_only_50m:Decoder-Only" "compare_dd_50m:Double Decoder" "compare_enc_dec_50m:Std Enc-Dec"; do
    name="${name_label%%:*}"
    label="${name_label#*:}"
    results="evals/results_${name}.json"
    if [ -f "$results" ]; then
        PT_FILES+=("$results")
        PT_LABELS+=("$label")
    fi
done

if [ ${#PT_FILES[@]} -ge 2 ]; then
    echo "  Generating pretrain comparison plot..."
    python evals/plot_comparison.py \
        --results "${PT_FILES[@]}" \
        --labels "${PT_LABELS[@]}" \
        --output-dir evals
    mv evals/architecture_comparison.png evals/pretrain_comparison.png 2>/dev/null || true
fi

# SFT comparison plot (with efficiency)
SFT_FILES=()
SFT_LABELS=()
for name_label in "sft_dec_only_50m:Decoder-Only (SFT)" "sft_dd_50m:Double Decoder (SFT)" "sft_enc_dec_50m:Std Enc-Dec (SFT)"; do
    name="${name_label%%:*}"
    label="${name_label#*:}"
    results="evals/results_${name}.json"
    if [ -f "$results" ]; then
        SFT_FILES+=("$results")
        SFT_LABELS+=("$label")
    fi
done

if [ ${#SFT_FILES[@]} -ge 2 ]; then
    echo "  Generating SFT comparison plot..."
    python evals/plot_comparison.py \
        --results "${SFT_FILES[@]}" \
        --labels "${SFT_LABELS[@]}" \
        --output-dir evals \
        --efficiency evals/efficiency_results.json
    mv evals/architecture_comparison.png evals/sft_comparison.png 2>/dev/null || true
fi

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline Complete!                                          ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Total time: ${TOTAL_MIN} minutes"
echo "║  Finished:   $(date)"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Output files in evals/:                                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Pretrain Results:"
for f in evals/results_compare_*.json; do
    [ -f "$f" ] && echo "    $f"
done
echo ""
echo "  SFT Results:"
for f in evals/results_sft_*.json; do
    [ -f "$f" ] && echo "    $f"
done
echo ""
echo "  Efficiency:"
[ -f evals/efficiency_results.json ] && echo "    evals/efficiency_results.json"
echo ""
echo "  Plots:"
[ -f evals/pretrain_comparison.png ]    && echo "    evals/pretrain_comparison.png"
[ -f evals/sft_comparison.png ]         && echo "    evals/sft_comparison.png"
[ -f evals/efficiency_comparison.png ]  && echo "    evals/efficiency_comparison.png"
echo ""
echo "  To download all results to your local machine:"
echo "    scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:block-based-double-decoder/evals/ ~/Desktop/evals/"
echo ""
