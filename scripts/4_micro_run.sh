#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Micro-scale dry run (300M model, 50 steps)
#
# Trains the REAL 300M model end-to-end on a tiny slice of data.
# Validates: wandb logging, DDP, gradient checkpointing, torch.compile,
#            checkpoint save/load, eval loop.
#
# Wall clock: ~5-10 minutes
# Devices:    1x H100 (or set NUM_GPUS)
# Cost:       ~$0.50
#
# After this completes, check your wandb dashboard:
#   Entity:  benjamin_bradley
#   Project: block-based-double-decoder
#   Run:     dd_300m_micro_dryrun
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

NUM_GPUS=${NUM_GPUS:-1}

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 4: Micro Dry Run (300M model, 50 steps, ${NUM_GPUS} GPUs)"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p checkpoints

torchrun --nproc_per_node=${NUM_GPUS} training/pretrain.py \
    --config-name=runs/pretrain_300m_micro

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Micro dry run complete ✓"
echo ""
echo "  Check wandb dashboard before proceeding:"
echo "    https://wandb.ai/benjamin_bradley/block-based-double-decoder"
echo "    Run: dd_300m_micro_dryrun"
echo ""
echo "  Verify:"
echo "    - Loss starts near 10.4 and decreases"
echo "    - Grad norms are stable (no spikes/collapse)"
echo "    - No NaN values"
echo ""
echo "  If everything looks good:"
echo "    bash scripts/5_train.sh"
echo "═══════════════════════════════════════════════════════════════"
