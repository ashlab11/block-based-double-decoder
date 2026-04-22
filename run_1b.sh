#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run 1B Double Decoder pretraining on 20B tokens (4x H100)
#
# This script handles the full pipeline:
#   1. Install dependencies
#   2. Verify wandb login
#   3. Build 32K tokenizer (on larger corpus)
#   4. Download 20B tokens from SlimPajama-627B
#   5. Pack dataset
#   6. Run preflight sanity checks
#   7. Run micro-scale dry run (1B model, 50 steps)
#   8. Launch full training
#
# Usage on RunPod:
#   git clone https://github.com/ashlab11/block-based-double-decoder.git
#   cd block-based-double-decoder
#   git checkout ben
#   bash run_1b.sh
#
# Estimated time: ~2-6 hours data prep + ~24 hours training on 4x H100
# Estimated cost: ~$250 on RunPod
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export HYDRA_FULL_ERROR=1
export HF_HUB_ENABLE_HF_TRANSFER=1

NUM_GPUS=${NUM_GPUS:-4}

echo "═══════════════════════════════════════════════════════════════"
echo "  Block-Based Double Decoder — 1B params, 20B tokens"
echo "  GPUs: ${NUM_GPUS}"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Install dependencies ─────────────────────────────────────────────
echo ""
echo "── Step 1: Installing dependencies ──"
pip install -q torch==2.6.0 torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer \
    flash-attn --no-build-isolation 2>/dev/null || true
pip install -q torch==2.6.0 torchtune==0.6.0 torchao==0.6.1 transformers datasets \
    hydra-core omegaconf matplotlib tqdm wandb hf_transfer

# Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"

# ── Step 2: Verify wandb login ───────────────────────────────────────────────
echo ""
echo "── Step 2: Verifying wandb login ──"
if ! python -c "import wandb; wandb.login(verify=True)" 2>/dev/null; then
    echo "  wandb not logged in. Please run: wandb login"
    echo "  Make sure you have access to the 'block-based-double-decoder' project"
    echo "  under entity 'benjamin_bradley'"
    exit 1
fi
echo "  wandb login verified ✓"

# ── Step 3: Build 32K tokenizer ──────────────────────────────────────────────
echo ""
echo "── Step 3: Building 32K tokenizer ──"
if [ -f "tokenizer/tokenizer_32k.json" ]; then
    echo "  32K tokenizer already exists, skipping."
else
    echo "  Step 3a: Downloading tokenizer corpus (500M tokens)..."
    if [ -f "data/Pretrain/tokenizer_corpus.jsonl" ]; then
        echo "  Tokenizer corpus already downloaded, skipping."
    else
        python data/retrieval_scripts/tokenizer_corpus.py --tokens 500000000
    fi
    echo "  Step 3b: Training 32K BPE tokenizer..."
    python tokenizer/hf_tokenizer.py \
        --vocab-size 32768 \
        --corpus data/Pretrain/tokenizer_corpus.jsonl \
        --output tokenizer/tokenizer_32k.json
fi

# ── Step 4: Download 20B tokens from SlimPajama ──────────────────────────────
echo ""
echo "── Step 4: Downloading 20B tokens from SlimPajama-627B ──"
if [ -f "data/Pretrain/slimpajama_20b.jsonl" ]; then
    echo "  Data already downloaded, skipping."
else
    python data/retrieval_scripts/slimpajama.py \
        --tokens 20000000000 \
        --output-prefix slimpajama_20b
fi

# ── Step 5: Pack dataset ─────────────────────────────────────────────────────
echo ""
echo "── Step 5: Packing dataset ──"
if [ -f "data/Pretrain/slimpajama_20b_packed.jsonl" ]; then
    echo "  Packed data already exists, skipping."
else
    python data/retrieval_scripts/pack_dataset.py \
        --tokenizer tokenizer/tokenizer_32k.json \
        --input data/Pretrain/slimpajama_20b.jsonl \
        --output data/Pretrain/slimpajama_20b_packed.jsonl \
        --eval-input data/Pretrain/slimpajama_20b_eval.jsonl \
        --eval-output data/Pretrain/slimpajama_20b_eval_packed.jsonl
fi

TRAIN_LINES=$(wc -l < data/Pretrain/slimpajama_20b_packed.jsonl)
echo "  Packed sequences: $TRAIN_LINES"

# ── Step 6: Preflight sanity checks ─────────────────────────────────────────
echo ""
echo "── Step 6: Running preflight sanity checks ──"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
bash tests/preflight.sh

# ── Step 7: Micro-scale dry run ──────────────────────────────────────────────
echo ""
echo "── Step 7: Micro-scale dry run (1B model, 50 steps) ──"
echo "  This validates the full pipeline before the real run."
echo "  Expected: ~5-10 minutes"
echo ""
mkdir -p checkpoints
torchrun --nproc_per_node=${NUM_GPUS} training/pretrain.py \
    --config-name=runs/pretrain_1b_micro

echo ""
echo "  ✓ Micro dry run complete. Check wandb dashboard:"
echo "    Entity: benjamin_bradley"
echo "    Project: block-based-double-decoder"
echo "    Run: dd_1b_micro_dryrun"
echo ""
echo "  Review the dashboard before proceeding."
echo "  Press Enter to start the full training run, or Ctrl+C to abort."
read -r

# ── Step 8: Full training ────────────────────────────────────────────────────
echo ""
echo "── Step 8: Full training ──"
echo "  Config: dim=1536, enc=24, dec=12 (~1.07B params)"
echo "  Data: 20B tokens, seq_len=2048"
echo "  Effective batch: 16 * 4 * ${NUM_GPUS} GPUs = $((16 * 4 * NUM_GPUS)) sequences"
echo ""

torchrun --nproc_per_node=${NUM_GPUS} training/pretrain.py \
    --config-name=runs/pretrain_1b

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/dd_1b_20btok.pt"
echo "═══════════════════════════════════════════════════════════════"
