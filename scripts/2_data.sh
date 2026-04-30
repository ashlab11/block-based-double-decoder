#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build tokenizer + download & pack 6B tokens
#
# Wall clock: ~1-2 hours total
#   - Tokenizer corpus download: ~15-30 min
#   - Tokenizer training: ~10-20 min
#   - 6B token download: ~30-60 min
#   - Packing: ~15-30 min
# Devices:    CPU + network (no GPU needed)
# Cost:       ~$2-5 (RunPod instance time during download)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_uv.sh

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 2: Data Pipeline (tokenizer + 6B tokens)"
echo "═══════════════════════════════════════════════════════════════"

# ── 2a: Build 32K tokenizer ──────────────────────────────────────────────────
echo ""
echo "── 2a: Building 32K tokenizer ──"
if [ -f "tokenizer/tokenizer_32k.json" ]; then
    echo "  32K tokenizer already exists, skipping."
else
    if [ -f "data/Pretrain/tokenizer_corpus.jsonl" ]; then
        echo "  Tokenizer corpus already downloaded, skipping download."
    else
        echo "  Downloading tokenizer corpus (500M tokens)..."
        uv_run python data/retrieval_scripts/tokenizer_corpus.py --tokens 500000000
    fi
    echo "  Training 32K BPE tokenizer..."
    uv_run python tokenizer/hf_tokenizer.py \
        --vocab-size 32768 \
        --corpus data/Pretrain/tokenizer_corpus.jsonl \
        --output tokenizer/tokenizer_32k.json

    # Clean up tokenizer corpus — it's been consumed and won't be needed again
    echo "  Removing tokenizer corpus to free disk space..."
    rm -f data/Pretrain/tokenizer_corpus.jsonl
fi

# ── 2b: Download 6B tokens ──────────────────────────────────────────────────
echo ""
echo "── 2b: Downloading 6B tokens from DKYoon/SlimPajama-6B ──"
if [ -f "data/Pretrain/slimpajama_6b.jsonl" ] || [ -f "data/Pretrain/slimpajama_6b_packed.jsonl" ]; then
    echo "  Data already downloaded (or already packed), skipping."
else
    uv_run python data/retrieval_scripts/slimpajama.py \
        --tokens 6000000000 \
        --output-prefix slimpajama_6b
fi

# ── 2c: Pack dataset ────────────────────────────────────────────────────────
echo ""
echo "── 2c: Packing dataset ──"
if [ -f "data/Pretrain/slimpajama_6b_packed.jsonl" ]; then
    echo "  Packed data already exists, skipping."
else
    uv_run python data/retrieval_scripts/pack_dataset.py \
        --tokenizer tokenizer/tokenizer_32k.json \
        --input data/Pretrain/slimpajama_6b.jsonl \
        --output data/Pretrain/slimpajama_6b_packed.jsonl \
        --eval-input data/Pretrain/slimpajama_6b_eval.jsonl \
        --eval-output data/Pretrain/slimpajama_6b_eval_packed.jsonl

    # Clean up raw JSONL files — the packed versions are all we need from here
    echo "  Removing raw JSONL files to free disk space..."
    rm -f data/Pretrain/slimpajama_6b.jsonl data/Pretrain/slimpajama_6b_eval.jsonl
fi

TRAIN_LINES=$(wc -l < data/Pretrain/slimpajama_6b_packed.jsonl)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Data pipeline complete ✓"
echo "  Packed sequences: $TRAIN_LINES"
echo "  Next: bash scripts/3_preflight.sh"
echo "═══════════════════════════════════════════════════════════════"
