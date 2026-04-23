#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install dependencies and verify environment
#
# Pinned versions (tested on RunPod H100 SXM with CUDA 12.4):
#   torch==2.6.0+cu124   flash-attn==2.8.3   torchtune==0.6.0
#
# Wall clock: ~10-20 minutes (flash-attn compilation dominates)
# Devices:    GPU needed for verification steps
# Cost:       ~$0.10-0.50 (minimal RunPod time)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 1: Setup & Dependencies"
echo "═══════════════════════════════════════════════════════════════"

# ── (1/4) Install PyTorch (pinned to cu124 for CUDA 12.4 pods) ──────────────
echo ""
echo "── (1/4) Installing PyTorch 2.6.0+cu124 ──"
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
echo "  ✓ Installed PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "  ✓ CUDA build: $(python -c 'import torch; print(torch.version.cuda)')"

# ── (2/4) Install training stack ─────────────────────────────────────────────
echo ""
echo "── (2/4) Installing training stack ──"
pip install torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer

# ── (3/4) Install flash-attn (must be AFTER torch, compiled against it) ──────
echo ""
echo "── (3/4) Installing flash-attn (compiles CUDA kernels — may take 10-20 min) ──"
echo "  torch.version.cuda = $(python -c 'import torch; print(torch.version.cuda)')"
echo "  System CUDA: $(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo 'nvcc not found')"
# --no-deps: prevent flash-attn from pulling a different torch from PyPI
# --no-cache-dir: prevent reuse of wheels compiled against a different torch
# --force-reinstall: always recompile even if version matches
pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --no-deps

# ── (4/4) Full environment verification ──────────────────────────────────────
echo ""
echo "── (4/4) Verifying environment ──"

echo ""
echo "  Packages:"
python -c "
import torch, flash_attn, wandb, transformers, datasets
print(f'    PyTorch:      {torch.__version__}')
print(f'    CUDA build:   {torch.version.cuda}')
print(f'    flash-attn:   {flash_attn.__version__}')
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
import torch._inductor.config as _inductor_config
_inductor_config.pattern_matcher = False  # workaround for 2.6.0 quantization bug

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
    echo "  under entity 'block-based-double-decoders'"
    exit 1
fi
echo "    ✓ wandb login verified"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete ✓"
echo "  Next: bash scripts/2_data.sh"
echo "═══════════════════════════════════════════════════════════════"
