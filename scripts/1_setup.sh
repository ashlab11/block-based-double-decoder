#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install dependencies and verify environment with uv
#
# Default wheel selection:
#   TORCH_CUDA_EXTRA=cu128
#
# Override if needed:
#   TORCH_CUDA_EXTRA=cu118 bash scripts/1_setup.sh
#   TORCH_CUDA_EXTRA=cu126 bash scripts/1_setup.sh
#
# Note: the PyTorch wheel CUDA runtime does not need to match the local
# toolkit exactly. A host with CUDA 12.9 can still use the cu128 wheels.
#
# Wall clock: ~2-5 minutes unless optional extras need source builds
# Devices:    GPU needed for verification steps
# Cost:       ~$0.10-0.50 (minimal RunPod time)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_uv.sh

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 1: Setup & Dependencies"
echo "═══════════════════════════════════════════════════════════════"

# ── (1/4) Sync project environment ───────────────────────────────────────────
echo ""
echo "── (1/4) Syncing uv environment ──"
echo "  PyTorch extra: ${TORCH_CUDA_EXTRA}"
echo "  Optional extras: ${UV_OPTIONAL_EXTRAS:-none}"
uv_sync_project
echo "  ✓ Environment synced into .venv"

# ── (2/4) Package verification ───────────────────────────────────────────────
echo ""
echo "── (2/4) Verifying packages ──"

echo ""
echo "  Packages:"
uv_run python -c "
import torch, wandb, transformers, datasets
import flash_attn
print(f'    PyTorch:      {torch.__version__}')
print(f'    CUDA build:   {torch.version.cuda}')
print(f'    flash-attn:   {flash_attn.__version__}')
print(f'    transformers: {transformers.__version__}')
print(f'    datasets:     {datasets.__version__}')
print(f'    wandb:        {wandb.__version__}')
"

# ── (3/4) GPU verification ───────────────────────────────────────────────────
echo ""
echo "── (3/4) Verifying GPU ──"
echo "  System CUDA toolkit: $(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo 'nvcc not found')"
echo "  GPU:"
uv_run python -c "
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

# ── (4/4) Runtime verification ───────────────────────────────────────────────
echo ""
echo "── (4/4) Verifying runtime ──"
echo "  torch.compile:"
uv_run python -c "
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
if ! uv_run python -c "import wandb; wandb.login(verify=True)" 2>/dev/null; then
    echo "  ✗ wandb not logged in. Run: uv run wandb login"
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
