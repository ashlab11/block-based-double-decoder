#!/usr/bin/env bash
#SBATCH --job-name=dd-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --output=logs/%x-%j.out
# ─────────────────────────────────────────────────────────────────────────────
# Step 5: FULL 50M TRAINING RUN — 1B tokens (Chinchilla-optimal)
#
# Wall clock: ~1-2 hours on 1x H100
# Devices:    1x H100 (or set NUM_GPUS / SLURM_GPUS_ON_NODE)
# Cost:       ~$3-6 (RunPod on-demand)
#
# Batch size is auto-detected to maximize GPU utilization.
#
# RunPod usage:   bash scripts/5_train.sh
#                 bash scripts/5_train.sh --resume checkpoints/dd_50m_1btok_<step>.pt
# SLURM usage:    sbatch scripts/5_train.sh
#                 sbatch scripts/5_train.sh --resume checkpoints/dd_50m_1btok_<step>.pt
#
# Storage location: defaults to repo-relative ./checkpoints. Override with:
#   export CKPT_DIR=$SCRATCH/dd/checkpoints
# (handled by symlink; configs keep using the relative path).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

: "${CKPT_DIR:=checkpoints}"
mkdir -p "$CKPT_DIR"
if [ "$CKPT_DIR" != "checkpoints" ] && [ ! -e "checkpoints" ]; then
    ln -s "$CKPT_DIR" "checkpoints"
fi

# Auto-detect GPU count from SLURM env, falling back to NUM_GPUS or 1.
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NUM_GPUS=${NUM_GPUS:-$SLURM_GPUS_ON_NODE}
else
    NUM_GPUS=${NUM_GPUS:-1}
fi

# Parse optional --resume flag
RESUME_ARGS=""
if [ "${1:-}" = "--resume" ] && [ -n "${2:-}" ]; then
    RESUME_ARGS="resume_from=${2}"
    echo "Resuming from checkpoint: ${2}"
fi

# SLURM preemption / time-limit handler: resubmit ourselves before SIGTERM
# arrives so the next allocation picks up from the latest checkpoint.
# pretrain.py writes checkpoints/dd_50m_1btok_step<N>.pt periodically.
on_signal() {
    echo "Caught SIGUSR1 — resubmitting job before preemption."
    LATEST=$(ls -t "$CKPT_DIR"/dd_50m_1btok_*.pt 2>/dev/null | head -n1 || true)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        if [ -n "$LATEST" ]; then
            sbatch "$0" --resume "$LATEST"
        else
            sbatch "$0"
        fi
    fi
    exit 0
}
trap on_signal USR1

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 5: FULL TRAINING — 50M params, 1B tokens"
echo "  GPUs: ${NUM_GPUS}"
[ -n "${SLURM_JOB_ID:-}" ] && echo "  SLURM job: ${SLURM_JOB_ID}"
echo "  Checkpoint dir: ${CKPT_DIR}"
echo "  Batch size: auto (will profile GPU memory)"
echo "  Estimated time: ~1-2 hours"
echo "  Estimated cost: ~\$3-6"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ -n "${SLURM_JOB_ID:-}" ]; then
    LAUNCHER=(srun torchrun --nproc_per_node=${NUM_GPUS})
else
    LAUNCHER=(torchrun --nproc_per_node=${NUM_GPUS})
fi

"${LAUNCHER[@]}" training/pretrain.py \
    --config-name=runs/pretrain_50m \
    ${RESUME_ARGS} &
TRAIN_PID=$!
wait $TRAIN_PID

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/dd_50m_1btok.pt"
echo "  wandb: https://wandb.ai/block-based-double-decoders/block-based-double-decoder"
echo "═══════════════════════════════════════════════════════════════"
