#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install dependencies and verify environment
#
# Wall clock: ~2-5 minutes
# Devices:    CPU only (no GPU needed)
# Cost:       ~$0.10 (minimal RunPod time)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 1: Setup & Dependencies"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "── Installing core dependencies ──"
echo "  (1/3) Installing PyTorch + core packages..."
pip install torch==2.6.0 torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer 2>&1 | tail -5
echo "  ✓ Core packages done"

echo ""
echo "  (2/3) Installing flash-attn (compiles CUDA kernels — may take 10-20 min)..."
pip install flash-attn --no-build-isolation 2>&1 | \
    grep -E '(Building|Compiling|Installing|error|ERROR|Successfully|already satisfied)' || true
echo "  ✓ flash-attn done"

echo ""
echo "  (3/3) Verifying all packages installed..."
pip install torch==2.6.0 torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer flash-attn 2>&1 | tail -3
echo "  ✓ All packages verified"

echo ""
echo "── Verifying GPU ──"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"

echo ""
echo "── Verifying wandb login ──"
if ! python -c "import wandb; wandb.login(verify=True)" 2>/dev/null; then
    echo "  ✗ wandb not logged in. Run: wandb login"
    echo "  Make sure you have access to project 'block-based-double-decoder'"
    echo "  under entity 'benjamin_bradley'"
    exit 1
fi
echo "  ✓ wandb login verified"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete ✓"
echo "  Next: bash scripts/2_data.sh"
echo "═══════════════════════════════════════════════════════════════"
