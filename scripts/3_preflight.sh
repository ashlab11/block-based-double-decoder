#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Preflight sanity checks (12 automated checks)
#
# Wall clock: ~10-15 minutes
#   - Most checks: seconds each
#   - micro_train (100 steps): ~5-8 min
#   - memory_profile: ~1 min
#   - compile_compat: ~2 min
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
