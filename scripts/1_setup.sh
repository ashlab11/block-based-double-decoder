#!/usr/bin/env bash
#SBATCH --job-name=dd-setup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install dependencies and verify environment
#
# Pinned versions (tested on RunPod H100 SXM with CUDA 12.4):
#   torch==2.6.0+cu124   flash-attn==2.8.3   torchtune==0.6.0
#
# Wall clock: ~10-20 minutes (flash-attn compilation dominates)
# Devices:    GPU needed for verification steps
# Cost:       ~$0.10-0.50 (minimal RunPod time)
#
# RunPod usage:   bash scripts/1_setup.sh
# SLURM usage:    sbatch scripts/1_setup.sh   (after activating conda env)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs

# ── Environment detection ────────────────────────────────────────────────────
IN_ALLOCATION=0; [ -n "${SLURM_JOB_ID:-}" ] && IN_ALLOCATION=1
ON_SLURM_CLUSTER=0; command -v sbatch >/dev/null 2>&1 && ON_SLURM_CLUSTER=1
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    HAS_GPU=1
fi

# ── SLURM-only preconditions (no-op on RunPod) ──────────────────────────────
if [ "$HAS_GPU" = 0 ]; then
    if [ "$ON_SLURM_CLUSTER" = 1 ]; then
        echo "✗ No GPU visible. You're likely on a SLURM login node."
        echo "  Submit setup as a job:"
        echo "    sbatch scripts/1_setup.sh"
        echo "  or grab an interactive GPU shell:"
        echo "    srun --gres=gpu:1 --pty bash scripts/1_setup.sh"
    else
        echo "✗ No GPU detected. Setup requires CUDA for verification."
    fi
    exit 1
fi

if [ "$ON_SLURM_CLUSTER" = 1 ] && [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "✗ No active Python env. On SLURM, activate one before running:"
    echo "    conda create -n dd python=3.11 -y && conda activate dd"
    echo "    sbatch scripts/1_setup.sh"
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load cuda/12.4 2>/dev/null || true
fi

if [ "$ON_SLURM_CLUSTER" = 1 ] && [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "✗ WANDB_API_KEY not set. Either:"
    echo "    export WANDB_API_KEY=...    (preferably in ~/.bashrc)"
    echo "  or skip wandb entirely:"
    echo "    export WANDB_MODE=offline"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 1: Setup & Dependencies"
[ "$IN_ALLOCATION" = 1 ] && echo "  Running under SLURM job ${SLURM_JOB_ID}"
echo "═══════════════════════════════════════════════════════════════"

# ── (1/4) Install PyTorch (auto-select cu128 for Blackwell, cu124 otherwise) ─
# Blackwell (sm_100, sm_120) has no kernels in cu124 wheels of torch 2.6 and
# fails with 'no kernel image is available for execution on the device'.
echo ""
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1)
COMPUTE_CAP_INT=$(awk -v c="$COMPUTE_CAP" 'BEGIN{split(c,a,"."); printf "%d", a[1]*10 + a[2]}')

if [ "$COMPUTE_CAP_INT" -ge 100 ]; then
    TORCH_SPEC="torch>=2.7"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    echo "── (1/4) Detected Blackwell (compute cap $COMPUTE_CAP) — installing PyTorch with cu128 wheels ──"
else
    TORCH_SPEC="torch==2.6.0"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "── (1/4) Installing PyTorch 2.6.0+cu124 (compute cap $COMPUTE_CAP) ──"
fi

pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
echo "  ✓ Installed PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "  ✓ CUDA build: $(python -c 'import torch; print(torch.version.cuda)')"

# ── (2/4) Install training stack ─────────────────────────────────────────────
# Pin torch here too so pip doesn't silently swap our build for a different
# one pulled transitively via torchdata→torch.
echo ""
echo "── (2/4) Installing training stack ──"
pip install torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer \
    "$TORCH_SPEC" --index-url "$TORCH_INDEX" \
    --extra-index-url https://pypi.org/simple/

# ── (3/4) Install flash-attn (must be AFTER torch, compiled against it) ──────
echo ""
echo "── (3/4) Installing flash-attn (compiles CUDA kernels — may take 10-20 min) ──"
echo "  torch.version.cuda = $(python -c 'import torch; print(torch.version.cuda)')"
echo "  System CUDA: $(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo 'nvcc not found')"
# flash-attn is optional — only used in inference mode (ComboAttention
# without block masks).  Training always uses flex_attention instead.
# The source build often hits ABI mismatches on RunPod images, so treat
# installation as best-effort.
python -c "from flash_attn import flash_attn_func; print('  ✓ flash-attn already works')" 2>/dev/null || {
    echo "  Pre-installed flash-attn broken or missing — attempting rebuild..."
    pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --no-deps 2>&1 | tail -5
    python -c "from flash_attn import flash_attn_func; print('  ✓ flash-attn rebuilt successfully')" 2>/dev/null || \
        echo "  ⚠ flash-attn unavailable (will use flex_attention fallback during training)"
}

# ── (4/4) Full environment verification ──────────────────────────────────────
echo ""
echo "── (4/4) Verifying environment ──"

echo ""
echo "  Packages:"
python -c "
import torch, wandb, transformers, datasets
print(f'    PyTorch:      {torch.__version__}')
print(f'    CUDA build:   {torch.version.cuda}')
try:
    import flash_attn
    print(f'    flash-attn:   {flash_attn.__version__}')
except ImportError:
    print(f'    flash-attn:   UNAVAILABLE (training will use flex_attention)')
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
if [ "${WANDB_MODE:-}" = "offline" ]; then
    echo "    ✓ WANDB_MODE=offline — skipping login check"
elif [ -n "${WANDB_API_KEY:-}" ]; then
    if python -c "import wandb; wandb.login(verify=True)" 2>/dev/null; then
        echo "    ✓ wandb login verified via WANDB_API_KEY"
    else
        echo "    ✗ WANDB_API_KEY set but verification failed. Check the key."
        exit 1
    fi
elif [ "$ON_SLURM_CLUSTER" = 0 ]; then
    # Interactive RunPod path: prompt the user once.
    if ! python -c "import wandb; wandb.login(verify=True)" 2>/dev/null; then
        echo "  ✗ wandb not logged in. Run: wandb login"
        echo "  Make sure you have access to project 'block-based-double-decoder'"
        echo "  under entity 'block-based-double-decoders'"
        exit 1
    fi
    echo "    ✓ wandb login verified"
else
    echo "    ✗ Unreachable: SLURM path without WANDB_API_KEY should have exited above."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete ✓"
echo "  Next: bash scripts/2_data.sh"
echo "═══════════════════════════════════════════════════════════════"
