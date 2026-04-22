#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 5: FULL 1B TRAINING RUN — 20B tokens
#
# ⚠️  THIS IS THE EXPENSIVE STEP (~$250)
#
# Wall clock: ~20-24 hours on 4x H100
# Devices:    4x H100 (or set NUM_GPUS)
# Cost:       ~$250 (RunPod on-demand) / ~$180 (spot)
#
# The run logs to wandb and saves checkpoints every 5000 steps.
# If preempted, resume with:
#   bash scripts/5_train.sh --resume checkpoints/dd_1b_20btok_<step>.pt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

NUM_GPUS=${NUM_GPUS:-4}

# Parse optional --resume flag
RESUME_ARGS=""
if [ "${1:-}" = "--resume" ] && [ -n "${2:-}" ]; then
    RESUME_ARGS="resume_from=${2}"
    echo "Resuming from checkpoint: ${2}"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 5: FULL TRAINING — 1B params, 20B tokens"
echo "  GPUs: ${NUM_GPUS}"
echo "  Effective batch: $((16 * 4 * NUM_GPUS)) sequences/step"
echo "  Estimated time: ~20-24 hours"
echo "  Estimated cost: ~\$250"
echo "═══════════════════════════════════════════════════════════════"
echo ""

mkdir -p checkpoints

torchrun --nproc_per_node=${NUM_GPUS} training/pretrain.py \
    --config-name=runs/pretrain_1b \
    ${RESUME_ARGS}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/dd_1b_20btok.pt"
echo "  wandb: https://wandb.ai/benjamin_bradley/block-based-double-decoder"
echo "═══════════════════════════════════════════════════════════════"
