#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 5: FULL 300M TRAINING RUN — 6B tokens
#
# ⚠️  THIS IS THE EXPENSIVE STEP (~$25-50)
#
# Wall clock: ~8-16 hours on 1x H100
# Devices:    1x H100 (or set NUM_GPUS)
# Cost:       ~$25-50 (RunPod on-demand)
#
# The run logs to wandb and saves checkpoints every 2000 steps.
# If preempted, resume with:
#   bash scripts/5_train.sh --resume checkpoints/dd_300m_6btok_<step>.pt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

NUM_GPUS=${NUM_GPUS:-1}

# Parse optional --resume flag
RESUME_ARGS=""
if [ "${1:-}" = "--resume" ] && [ -n "${2:-}" ]; then
    RESUME_ARGS="resume_from=${2}"
    echo "Resuming from checkpoint: ${2}"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 5: FULL TRAINING — 300M params, 6B tokens"
echo "  GPUs: ${NUM_GPUS}"
echo "  Effective batch: $((16 * 16 * NUM_GPUS)) sequences/step"
echo "  Estimated time: ~8-16 hours"
echo "  Estimated cost: ~\$25-50"
echo "═══════════════════════════════════════════════════════════════"
echo ""

mkdir -p checkpoints

CONFIG_NAME=runs/pretrain_300m \
    torchrun --nproc_per_node=${NUM_GPUS} training/pretrain.py \
    ${RESUME_ARGS}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/dd_300m_6btok.pt"
echo "  wandb: https://wandb.ai/benjamin_bradley/block-based-double-decoder"
echo "═══════════════════════════════════════════════════════════════"
