#!/usr/bin/env bash
#SBATCH --job-name=dd-preflight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Preflight sanity checks (11 automated checks for 50M model)
#
# Wall clock: ~3-5 minutes
#   - Most checks: seconds each
#   - micro_train (100 steps): ~1-2 min
#   - memory_profile: ~30s
#   - compile_compat: ~1 min
# Devices:    1+ GPU (DDP smoke test needs 2+)
# Cost:       ~$0.50 (minimal GPU time)
#
# RunPod usage:   bash scripts/3_preflight.sh
# SLURM usage:    sbatch scripts/3_preflight.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 3: Preflight Sanity Checks"
echo "═══════════════════════════════════════════════════════════════"

bash tests/preflight.sh

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All preflight checks passed ✓"
echo "  Next: bash scripts/4_micro_run.sh"
echo "═══════════════════════════════════════════════════════════════"
