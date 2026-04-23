#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install dependencies and verify environment
#
# Wall clock: ~10-20 minutes (flash-attn compilation dominates)
# Devices:    GPU needed for verification steps
# Cost:       ~$0.10-0.50 (minimal RunPod time)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 1: Setup & Dependencies"
echo "═══════════════════════════════════════════════════════════════"

# ── Detect CUDA version for correct PyTorch wheel ────────────────────────────
CUDA_VERSION=$(python -c "
import subprocess, re
out = subprocess.check_output(['nvcc', '--version'], text=True)
m = re.search(r'release (\d+)\.(\d+)', out)
if m: print(f'cu{m.group(1)}{m.group(2)}')
else: print('cu124')
" 2>/dev/null || echo "cu124")
echo "  Detected CUDA: ${CUDA_VERSION}"

# ── (1/4) Install PyTorch ────────────────────────────────────────────────────
echo ""
echo "── (1/4) Installing PyTorch ──"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VERSION}"
echo "  Using index: ${TORCH_INDEX}"

# Try latest stable first, fall back to 2.6.0 if wheel not found
if pip install 'torch>=2.7.0' --index-url "${TORCH_INDEX}" 2>/dev/null; then
    echo "  ✓ Installed PyTorch $(python -c 'import torch; print(torch.__version__)')"
else
    echo "  No torch>=2.7.0 found for ${CUDA_VERSION}, trying 2.6.0..."
    pip install 'torch==2.6.0' --index-url "${TORCH_INDEX}"
    echo "  ✓ Installed PyTorch $(python -c 'import torch; print(torch.__version__)')"
fi

# ── (2/4) Install training stack ─────────────────────────────────────────────
echo ""
echo "── (2/4) Installing training stack ──"
pip install torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer

# ── (3/4) Install flash-attn ─────────────────────────────────────────────────
echo ""
echo "── (3/4) Installing flash-attn (compiles CUDA kernels — may take 10-20 min) ──"
pip install flash-attn --no-build-isolation

# ── (4/4) Full environment verification ──────────────────────────────────────
echo ""
echo "── (4/4) Verifying environment ──"

echo ""
echo "  Packages:"
python -c "
import torch, flash_attn, wandb, transformers, datasets, torchtune
print(f'    PyTorch:      {torch.__version__}')
print(f'    CUDA build:   {torch.version.cuda}')
print(f'    flash-attn:   {flash_attn.__version__}')
print(f'    torchtune:    {torchtune.__version__}')
print(f'    transformers: {transformers.__version__}')
print(f'    datasets:     {datasets.__version__}')
print(f'    wandb:        {wandb.__version__}')
"

echo ""
echo "  GPU:"
python -c "
import torch, sys
print(f'    CUDA available: {torch.cuda.is_available()}')
print(f'    GPU count:      {torch.cuda.device_count()}')
if not torch.cuda.is_available():
    print('  ✗ FATAL: No GPU detected. Cannot proceed.')
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
"

echo ""
echo "  torch.compile:"
python -c "
import torch, sys

# Quick compile smoke test — catches inductor bugs before they waste time later
class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
    def forward(self, x):
        return self.linear(x).sum()

model = TinyModel().cuda()
compiled = torch.compile(model, fullgraph=False, dynamic=False)
x = torch.randn(2, 64, device='cuda')
loss = compiled(x)
loss.backward()
print('    ✓ torch.compile works (inductor backend)')
"

echo ""
echo "  wandb login:"
if ! python -c "import wandb; wandb.login(verify=True)" 2>/dev/null; then
    echo "  ✗ wandb not logged in. Run: wandb login"
    echo "  Make sure you have access to project 'block-based-double-decoder'"
    echo "  under entity 'benjamin_bradley'"
    exit 1
fi
echo "    ✓ wandb login verified"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete ✓"
echo "  Next: bash scripts/2_data.sh"
echo "═══════════════════════════════════════════════════════════════"
