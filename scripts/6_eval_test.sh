#!/bin/bash
# Smoke-test all 22 evals with 1 example each to verify the code works.
#
# Usage:
#   bash scripts/6_eval_test.sh checkpoints/model.pt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_uv.sh

CHECKPOINT="${1:?Usage: bash scripts/6_eval_test.sh <checkpoint.pt>}"

uv_run python evals/run_evals.py \
    --checkpoint "$CHECKPOINT" \
    --evals all \
    --max-examples 1 \
    --output evals/results_smoke_test.json
