#!/bin/bash
# Run all 22 evals on a trained model checkpoint.
#
# Usage:
#   bash scripts/6_eval.sh checkpoints/model.pt
#   bash scripts/6_eval.sh checkpoints/model.pt 50   # cap at 50 examples per eval (fast debug)

set -euo pipefail

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
