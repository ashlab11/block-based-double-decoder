#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 1B Double Decoder — Quick Reference
#
# Run each step separately to control what executes:
#
#   bash scripts/1_setup.sh       # Install deps, verify wandb    (~2-5 min,  ~$0.10)
#   bash scripts/2_data.sh        # Tokenizer + 20B token download (~2-5 hrs, ~$5-12)
#   bash scripts/3_preflight.sh   # 12 sanity checks               (~10-15 min, ~$0.50)
#   bash scripts/4_micro_run.sh   # 1B model dry run, 50 steps     (~5-10 min, ~$0.50)
#   bash scripts/5_train.sh       # ⚠️  FULL RUN                   (~20-24 hrs, ~$250)
#
# To resume after preemption:
#   bash scripts/5_train.sh --resume checkpoints/dd_1b_20btok_<step>.pt
# ─────────────────────────────────────────────────────────────────────────────
echo "Run each step individually — see scripts/ directory."
echo "  bash scripts/1_setup.sh"
echo "  bash scripts/2_data.sh"
echo "  bash scripts/3_preflight.sh"
echo "  bash scripts/4_micro_run.sh"
echo "  bash scripts/5_train.sh       # ⚠️  THE EXPENSIVE ONE (~\$250)"
