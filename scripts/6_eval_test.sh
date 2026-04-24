#!/bin/bash
# Smoke-test all 22 evals with 1 example each to verify the code works.
#
# Usage:
#   bash scripts/6_eval_test.sh checkpoints/model.pt

set -euo pipefail

CHECKPOINT="${1:?Usage: bash scripts/6_eval_test.sh <checkpoint.pt>}"

python evals/run_evals.py \
    --checkpoint "$CHECKPOINT" \
    --evals all \
    --max-examples 1 \
    --output evals/results_smoke_test.json
