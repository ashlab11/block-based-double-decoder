#!/usr/bin/env bash
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
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

NUM_GPUS=${NUM_GPUS:-1}

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 4: Micro Dry Run (50M model, 50 steps, ${NUM_GPUS} GPUs)"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p checkpoints

torchrun --nproc_per_node=${NUM_GPUS} training/pretrain.py \
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
