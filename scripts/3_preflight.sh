#!/usr/bin/env bash
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
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
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
