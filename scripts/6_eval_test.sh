#!/bin/bash
#SBATCH --job-name=dd-eval-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# Smoke-test all 22 evals with 1 example each to verify the code works.
#
# RunPod usage:   bash   scripts/6_eval_test.sh checkpoints/model.pt
# SLURM usage:    sbatch scripts/6_eval_test.sh checkpoints/model.pt

set -euo pipefail
mkdir -p logs

CHECKPOINT="${1:?Usage: bash scripts/6_eval_test.sh <checkpoint.pt>}"

python evals/run_evals.py \
    --checkpoint "$CHECKPOINT" \
    --evals all \
    --max-examples 1 \
    --output evals/results_smoke_test.json
