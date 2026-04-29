#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Download pre-packed data (fast) or build from scratch (slow)
#
# Fast path (~2-5 min): download pre-packed data + tokenizer from HuggingFace
# Slow path (~1-2 hours): build tokenizer, download raw text, tokenize & pack
#
# The fast path is tried first. If the HF dataset doesn't exist or download
# fails, falls back to the slow path automatically.
#
# Devices:    CPU + network (no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1

# HuggingFace dataset repo for pre-packed data
HF_DATASET_REPO="ashlab11/dd-packed-data"

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 2: Data Pipeline"
echo "═══════════════════════════════════════════════════════════════"

# ── Fast path: download pre-packed data from HuggingFace ────────────────────
NEED_SLOW_PATH=false

if [ -f "data/Pretrain/slimpajama_6b_packed.jsonl" ] && \
   [ -f "data/Pretrain/slimpajama_6b_eval_packed.jsonl" ] && \
   [ -f "tokenizer/tokenizer_32k.json" ]; then
    echo ""
    echo "  All data files already exist, skipping."
else
    echo ""
    echo "── Trying fast path: download pre-packed data from HuggingFace ──"
    echo "  Repo: ${HF_DATASET_REPO}"

    mkdir -p data/Pretrain tokenizer

    if huggingface-cli download "${HF_DATASET_REPO}" \
        slimpajama_6b_packed.jsonl \
        slimpajama_6b_eval_packed.jsonl \
        tokenizer_32k.json \
        --repo-type dataset \
        --local-dir /tmp/dd-packed-download \
        --quiet 2>/dev/null; then

        # Move files to their expected locations
        mv -n /tmp/dd-packed-download/slimpajama_6b_packed.jsonl data/Pretrain/ 2>/dev/null || true
        mv -n /tmp/dd-packed-download/slimpajama_6b_eval_packed.jsonl data/Pretrain/ 2>/dev/null || true
        mv -n /tmp/dd-packed-download/tokenizer_32k.json tokenizer/ 2>/dev/null || true
        rm -rf /tmp/dd-packed-download

        # Verify all files arrived
        if [ -f "data/Pretrain/slimpajama_6b_packed.jsonl" ] && \
           [ -f "data/Pretrain/slimpajama_6b_eval_packed.jsonl" ] && \
           [ -f "tokenizer/tokenizer_32k.json" ]; then
            echo "  ✓ Downloaded pre-packed data successfully (fast path)"
        else
            echo "  ⚠ Download incomplete, falling back to slow path"
            NEED_SLOW_PATH=true
        fi
    else
        echo "  ⚠ HF download failed, falling back to slow path"
        NEED_SLOW_PATH=true
    fi
fi

# ── Slow path: build everything from scratch ────────────────────────────────
if [ "$NEED_SLOW_PATH" = true ]; then
    echo ""
    echo "── Slow path: building tokenizer + downloading & packing 6B tokens ──"
    echo "  This will take ~1-2 hours."

    # ── 2a: Build 32K tokenizer ─────────────────────────────────────────────
    echo ""
    echo "── 2a: Building 32K tokenizer ──"
    if [ -f "tokenizer/tokenizer_32k.json" ]; then
        echo "  32K tokenizer already exists, skipping."
    else
        if [ -f "data/Pretrain/tokenizer_corpus.jsonl" ]; then
            echo "  Tokenizer corpus already downloaded, skipping download."
        else
            echo "  Downloading tokenizer corpus (500M tokens)..."
            python data/retrieval_scripts/tokenizer_corpus.py --tokens 500000000
        fi
        echo "  Training 32K BPE tokenizer..."
        python tokenizer/hf_tokenizer.py \
            --vocab-size 32768 \
            --corpus data/Pretrain/tokenizer_corpus.jsonl \
            --output tokenizer/tokenizer_32k.json

        echo "  Removing tokenizer corpus to free disk space..."
        rm -f data/Pretrain/tokenizer_corpus.jsonl
    fi

    # ── 2b: Download 6B tokens ──────────────────────────────────────────────
    echo ""
    echo "── 2b: Downloading 6B tokens from DKYoon/SlimPajama-6B ──"
    if [ -f "data/Pretrain/slimpajama_6b.jsonl" ] || [ -f "data/Pretrain/slimpajama_6b_packed.jsonl" ]; then
        echo "  Data already downloaded (or already packed), skipping."
    else
        python data/retrieval_scripts/slimpajama.py \
            --tokens 6000000000 \
            --output-prefix slimpajama_6b
    fi

    # ── 2c: Pack dataset ────────────────────────────────────────────────────
    echo ""
    echo "── 2c: Packing dataset ──"
    if [ -f "data/Pretrain/slimpajama_6b_packed.jsonl" ]; then
        echo "  Packed data already exists, skipping."
    else
        python data/retrieval_scripts/pack_dataset.py \
            --tokenizer tokenizer/tokenizer_32k.json \
            --input data/Pretrain/slimpajama_6b.jsonl \
            --output data/Pretrain/slimpajama_6b_packed.jsonl \
            --eval-input data/Pretrain/slimpajama_6b_eval.jsonl \
            --eval-output data/Pretrain/slimpajama_6b_eval_packed.jsonl

        echo "  Removing raw JSONL files to free disk space..."
        rm -f data/Pretrain/slimpajama_6b.jsonl data/Pretrain/slimpajama_6b_eval.jsonl
    fi
fi

TRAIN_LINES=$(wc -l < data/Pretrain/slimpajama_6b_packed.jsonl)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Data pipeline complete ✓"
echo "  Packed sequences: $TRAIN_LINES"
echo "  Next: bash scripts/3_preflight.sh"
echo "═══════════════════════════════════════════════════════════════"
