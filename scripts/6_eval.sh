#!/bin/bash
#SBATCH --job-name=dd-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Run all 22 evals on a trained model checkpoint.
#
# RunPod usage:
#   bash scripts/6_eval.sh checkpoints/model.pt
#   bash scripts/6_eval.sh checkpoints/model.pt 50   # cap at 50 examples per eval (fast debug)
#
# SLURM usage:
#   sbatch scripts/6_eval.sh checkpoints/model.pt
#   sbatch scripts/6_eval.sh checkpoints/model.pt 50

set -euo pipefail
mkdir -p logs

CHECKPOINT="${1:?Usage: bash scripts/6_eval.sh <checkpoint.pt> [max_examples]}"
MAX_EXAMPLES="${2:-}"

EXTRA_ARGS=""
if [ -n "$MAX_EXAMPLES" ]; then
    EXTRA_ARGS="--max-examples $MAX_EXAMPLES"
fi

python evals/run_evals.py \
    --checkpoint "$CHECKPOINT" \
    --evals all \
    $EXTRA_ARGS
