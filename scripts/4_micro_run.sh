#!/usr/bin/env bash
#SBATCH --job-name=dd-micro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Micro-scale dry run (50M model, 50 steps)
#
# Trains the REAL 50M model end-to-end on a tiny slice of data.
# Validates: wandb logging, torch.compile, checkpoint save/load, eval loop.
#
# Wall clock: ~1 minute
# Devices:    1x H100 (or set NUM_GPUS)
# Cost:       ~$0.05
#
# After this completes, check your wandb dashboard:
#   Entity:  block-based-double-decoders
#   Project: block-based-double-decoder
#   Run:     dd_50m_micro_dryrun
#
# RunPod usage:   bash scripts/4_micro_run.sh
# SLURM usage:    sbatch scripts/4_micro_run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Auto-detect GPU count from SLURM env, falling back to NUM_GPUS or 1.
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NUM_GPUS=${NUM_GPUS:-$SLURM_GPUS_ON_NODE}
else
    NUM_GPUS=${NUM_GPUS:-1}
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 4: Micro Dry Run (50M model, 50 steps, ${NUM_GPUS} GPUs)"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p checkpoints

if [ -n "${SLURM_JOB_ID:-}" ]; then
    LAUNCHER=(srun torchrun --nproc_per_node=${NUM_GPUS})
else
    LAUNCHER=(torchrun --nproc_per_node=${NUM_GPUS})
fi

"${LAUNCHER[@]}" training/pretrain.py \
    --config-name=runs/pretrain_50m_micro

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Micro dry run complete ✓"
echo ""
echo "  Check wandb dashboard before proceeding:"
echo "    https://wandb.ai/block-based-double-decoders/block-based-double-decoder"
echo "    Run: dd_50m_micro_dryrun"
echo ""
echo "  Verify:"
echo "    - Loss starts near 10.4 and decreases"
echo "    - Grad norms are stable (no spikes/collapse)"
echo "    - No NaN values"
echo ""
echo "  If everything looks good:"
echo "    bash scripts/5_train.sh"
echo "═══════════════════════════════════════════════════════════════"
