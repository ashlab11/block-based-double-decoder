#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Preflight checks for 50M Double Decoder training run
#
# Runs all sanity checks sequentially, stopping on first failure.
# The DDP smoke test requires at least 2 GPUs.
#
# Usage:
#   bash tests/preflight.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_uv.sh

echo "═══════════════════════════════════════════════════════════════"
echo "  PREFLIGHT CHECKS — 50M Double Decoder"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Run all single-GPU checks
echo "── Running single-GPU checks ──"
uv_run python tests/sanity_checks.py --check param_count
uv_run python tests/sanity_checks.py --check forward_pass
uv_run python tests/sanity_checks.py --check gradient_flow
uv_run python tests/sanity_checks.py --check micro_train --steps 100
uv_run python tests/sanity_checks.py --check memory_profile
uv_run python tests/sanity_checks.py --check data_pipeline
uv_run python tests/sanity_checks.py --check tokenizer
uv_run python tests/sanity_checks.py --check block_masks
uv_run python tests/sanity_checks.py --check combo_attn_stability
uv_run python tests/sanity_checks.py --check config_validation
uv_run python tests/sanity_checks.py --check compile_compat

# DDP smoke test (requires >= 2 GPUs)
NUM_GPUS=$(uv_run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS" -ge 2 ]; then
    echo ""
    echo "── Running DDP smoke test (${NUM_GPUS} GPUs) ──"
    uv_run torchrun --nproc_per_node=2 tests/sanity_checks.py --check ddp_smoke
else
    echo ""
    echo "── Skipping DDP smoke test (only ${NUM_GPUS} GPU available) ──"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL PREFLIGHT CHECKS PASSED ✓"
echo "═══════════════════════════════════════════════════════════════"
