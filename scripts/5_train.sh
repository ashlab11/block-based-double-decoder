#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 5: FULL 50M TRAINING RUN — 1B tokens (Chinchilla-optimal)
#
# Wall clock: ~1-2 hours on 1x H100
# Devices:    1x H100 (or set NUM_GPUS)
# Cost:       ~$3-6 (RunPod on-demand)
#
# Batch size is auto-detected to maximize GPU utilization.
# If preempted, resume with:
#   bash scripts/5_train.sh --resume checkpoints/dd_50m_1btok_<step>.pt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_uv.sh

NUM_GPUS=${NUM_GPUS:-1}

# Parse optional --resume flag
RESUME_ARGS=""
if [ "${1:-}" = "--resume" ] && [ -n "${2:-}" ]; then
    RESUME_ARGS="resume_from=${2}"
    echo "Resuming from checkpoint: ${2}"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 5: FULL TRAINING — 50M params, 1B tokens"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size: auto (will profile GPU memory)"
echo "  Estimated time: ~1-2 hours"
echo "  Estimated cost: ~\$3-6"
echo "═══════════════════════════════════════════════════════════════"
echo ""

mkdir -p checkpoints

uv_run torchrun --nproc_per_node=${NUM_GPUS} training/pretrain.py \
    --config-name=runs/pretrain_50m \
    ${RESUME_ARGS}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/dd_50m_1btok.pt"
echo "  wandb: https://wandb.ai/block-based-double-decoders/block-based-double-decoder"
echo "═══════════════════════════════════════════════════════════════"
