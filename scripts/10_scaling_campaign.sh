#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Full scaling campaign: 3 archs (dd/sed/dec) × 2 sizes (5M, 25M)
#                         × 5 token budgets (100M, 500M, 1B, 3B, 6B) = 30 cells
#
#  Settings:
#    - Optimal LR per arch from configs/mup_tuned.json (auto-loaded)
#    - Optimal WD per arch from configs/wd_tuned.json  (auto-loaded)
#    - Effective batch = 32 sequences × 2048 = 65,536 tokens/step
#      (batch_size=16 × grad_accum=2)
#    - Checkpoints saved at 50%, 90%, 100% of training
#    - Local copies: checkpoints/scaling_campaign/
#    - HF Hub copies: $HF_REPO  (set below or via env)
#    - Live wandb logging
#
#  Required env vars:
#    HF_REPO        — e.g. "your-username/bbdd-scaling-checkpoints"
#    WANDB_ENTITY   — your wandb team or username
#
#  Required CLI prep (run once):
#    pip install huggingface_hub wandb
#    huggingface-cli login          # token from huggingface.co/settings/tokens
#    wandb login                    # key from wandb.ai/authorize
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

: "${HF_REPO:?must set HF_REPO, e.g. export HF_REPO=your-username/bbdd-scaling-checkpoints}"
: "${WANDB_ENTITY:?must set WANDB_ENTITY, e.g. export WANDB_ENTITY=block-based-double-decoders}"

WANDB_PROJECT="${WANDB_PROJECT:-dd-scaling-campaign}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/scaling_campaign}"

python scripts/parallel_scaling.py \
    --arch-set large --only-arch 5M,25M \
    --token-set large \
    --batch-size 16 --grad-accum 2 \
    --save-checkpoints \
    --checkpoint-fractions 0.5,0.9,1.0 \
    --hf-repo "$HF_REPO" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --output-dir "$OUTPUT_DIR" \
    --eval-suite paper \
    --mid-eval-points 5
