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
pip install --progress-bar on torch==2.6.0 torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer
echo "  ✓ Core packages done"

echo ""
echo "  (2/3) Installing flash-attn (compiles CUDA kernels — may take 10-20 min)..."
echo "        You will see compiler output scrolling — this is normal."
pip install -v flash-attn --no-build-isolation 2>&1 | \
    while IFS= read -r line; do
        case "$line" in
            *running\ build_ext*) echo "    ▶ Starting CUDA kernel compilation..." ;;
            *ninja*) echo "    ▶ Compiling with ninja: $line" | head -c 120 ;;
            *nvcc*) echo "    ▶ [nvcc] $(echo "$line" | grep -oP '[^/]+\.cu' | head -1)" ;;
            *Successfully\ installed*) echo "    $line" ;;
            *already\ satisfied*) echo "    $line" ;;
            *error*|*ERROR*|*Error*) echo "    ⚠ $line" ;;
            *building*|*Building*) echo "    ▶ $line" ;;
        esac
    done
echo "  ✓ flash-attn done"

echo ""
echo "  (3/3) Verifying all packages installed..."
pip install torch==2.6.0 torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer flash-attn
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
