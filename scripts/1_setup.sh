#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install dependencies and verify environment
#
# Wall clock: ~10-20 minutes (flash-attn compilation dominates)
# Devices:    CPU only (no GPU needed)
# Cost:       ~$0.10-0.50 (minimal RunPod time)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 1: Setup & Dependencies"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "── (1/3) Installing core packages (including PyTorch 2.6+) ──"
pip install 'torch>=2.6.0' --index-url https://download.pytorch.org/whl/cu124
pip install torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer

echo ""
echo "── (2/3) Installing flash-attn (compiles CUDA kernels — 10-20 min) ──"
pip install flash-attn --no-build-isolation

echo ""
echo "── (3/3) Verifying environment ──"
echo ""
echo "  Packages:"
python -c "
import torch, flash_attn, wandb, transformers, datasets
print(f'    PyTorch:      {torch.__version__}')
print(f'    flash-attn:   {flash_attn.__version__}')
print(f'    transformers: {transformers.__version__}')
print(f'    datasets:     {datasets.__version__}')
print(f'    wandb:        {wandb.__version__}')
"

echo ""
echo "  GPU:"
python -c "
import torch
print(f'    CUDA:      {torch.cuda.is_available()}')
print(f'    GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
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
