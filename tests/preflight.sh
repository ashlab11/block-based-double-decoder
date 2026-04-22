#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Preflight checks for 1B Double Decoder training run
#
# Runs all sanity checks sequentially, stopping on first failure.
# The DDP smoke test requires at least 2 GPUs.
#
# Usage:
#   bash tests/preflight.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "═══════════════════════════════════════════════════════════════"
echo "  PREFLIGHT CHECKS — 1B Double Decoder"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Run all single-GPU checks
echo "── Running single-GPU checks ──"
python tests/sanity_checks.py --check param_count
python tests/sanity_checks.py --check forward_pass
python tests/sanity_checks.py --check gradient_flow
python tests/sanity_checks.py --check micro_train --steps 100
python tests/sanity_checks.py --check memory_profile
python tests/sanity_checks.py --check data_pipeline
python tests/sanity_checks.py --check tokenizer
python tests/sanity_checks.py --check block_masks
python tests/sanity_checks.py --check combo_attn_stability
python tests/sanity_checks.py --check config_validation
python tests/sanity_checks.py --check compile_compat

# DDP smoke test (requires >= 2 GPUs)
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS" -ge 2 ]; then
    echo ""
    echo "── Running DDP smoke test (${NUM_GPUS} GPUs) ──"
    torchrun --nproc_per_node=2 tests/sanity_checks.py --check ddp_smoke
else
    echo ""
    echo "── Skipping DDP smoke test (only ${NUM_GPUS} GPU available) ──"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL PREFLIGHT CHECKS PASSED ✓"
echo "═══════════════════════════════════════════════════════════════"
