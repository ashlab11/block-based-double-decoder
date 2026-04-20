#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run 15M Double Decoder pretraining on 50M tokens (H100)
#
# Usage on RunPod:
#   git clone https://github.com/ashlab11/block-based-double-decoder.git
#   cd block-based-double-decoder
#   git checkout ben
#   bash run_experiment.sh
#
# Estimated time on H100: ~15-30 minutes total (data + train)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  Block-Based Double Decoder — 15M params, 50M tokens"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Install dependencies ─────────────────────────────────────────────
echo ""
echo "── Step 1: Installing dependencies ──"
pip install -q torch torchtune torchao transformers datasets hydra-core omegaconf matplotlib tqdm flash-attn --no-build-isolation 2>/dev/null || true
pip install -q torch torchtune torchao transformers datasets hydra-core omegaconf matplotlib tqdm

# Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Step 2: Download 50M tokens from SlimPajama ─────────────────────────────
echo ""
echo "── Step 2: Downloading 50M tokens from SlimPajama ──"
if [ -f "data/Pretrain/slimpajama.jsonl" ]; then
    echo "  Data already downloaded, skipping."
else
    python data/retrieval_scripts/slimpajama.py --tokens 50000000
fi

# ── Step 3: Pack dataset into training-ready JSONL ───────────────────────────
echo ""
echo "── Step 3: Packing dataset ──"
if [ -f "data/Pretrain/slimpajama_packed.jsonl" ]; then
    echo "  Packed data already exists, skipping."
else
    python data/retrieval_scripts/pack_dataset.py
fi

# Count lines to estimate training steps
TRAIN_LINES=$(wc -l < data/Pretrain/slimpajama_packed.jsonl)
echo "  Packed sequences: $TRAIN_LINES"

# ── Step 4: Train ────────────────────────────────────────────────────────────
echo ""
echo "── Step 4: Training ──"
echo "  Config: dim=256, enc=6, dec=3 (~15M params)"
echo "  Data: 50M tokens, seq_len=2048"
echo ""

mkdir -p checkpoints

torchrun --nproc_per_node=1 training/pretrain.py

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/dd_15m_50mtok.pt"
echo "═══════════════════════════════════════════════════════════════"
