#!/usr/bin/env bash
#SBATCH --job-name=dd-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
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
#
# RunPod usage:   bash scripts/2_data.sh
# SLURM usage:    sbatch scripts/2_data.sh   (after sourcing slurm.env to point
#                                             DATA_DIR at scratch storage)
#
# Storage location: defaults to repo-relative ./data/Pretrain. Override with:
#   export DATA_DIR=$SCRATCH/dd/data/Pretrain
#   export TOKENIZER_DIR=$SCRATCH/dd/tokenizer
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

: "${DATA_DIR:=data/Pretrain}"
: "${TOKENIZER_DIR:=tokenizer}"
mkdir -p "$DATA_DIR" "$TOKENIZER_DIR"

# Symlink scratch dirs back into the repo-relative paths so the rest of
# the pipeline (Hydra configs, pretrain.py) keeps using relative paths
# unchanged. No-op when DATA_DIR is already the default.
if [ "$DATA_DIR" != "data/Pretrain" ] && [ ! -e "data/Pretrain" ]; then
    mkdir -p data
    ln -s "$DATA_DIR" "data/Pretrain"
fi
if [ "$TOKENIZER_DIR" != "tokenizer" ] && [ ! -e "tokenizer" ]; then
    ln -s "$TOKENIZER_DIR" "tokenizer"
fi

# HuggingFace dataset repo for pre-packed data
HF_DATASET_REPO="bpbradle/slimpajama-6b-packed"

echo "═══════════════════════════════════════════════════════════════"
echo "  Step 2: Data Pipeline"
echo "═══════════════════════════════════════════════════════════════"

# ── Fast path: download pre-packed data from HuggingFace ────────────────────
# Set FORCE_SLOW_PATH=1 to skip the HF download. Required after any change
# to tokenizer/hf_tokenizer.py's special_tokens list (e.g. the sentinel
# additions in commit 5c78ad4) — the HF pre-pack is keyed to the older
# tokenizer and using it produces off-by-15 token IDs across the entire
# vocabulary.
NEED_SLOW_PATH=false

if [ "${FORCE_SLOW_PATH:-0}" = "1" ]; then
    echo ""
    echo "── FORCE_SLOW_PATH=1 set: skipping HF fast path, rebuilding from source ──"
    NEED_SLOW_PATH=true
elif [ -f "data/Pretrain/slimpajama_6b_packed.jsonl" ] && \
   [ -f "data/Pretrain/slimpajama_6b_eval_packed.jsonl" ] && \
   [ -f "tokenizer/tokenizer_32k.json" ]; then
    echo ""
    echo "  All data files already exist, skipping."
else
    echo ""
    echo "── Trying fast path: download pre-packed data from HuggingFace ──"
    echo "  Repo: ${HF_DATASET_REPO}"

    mkdir -p data/Pretrain tokenizer

    if hf download "${HF_DATASET_REPO}" \
        data/Pretrain/slimpajama_6b_packed.jsonl \
        data/Pretrain/slimpajama_6b_eval_packed.jsonl \
        tokenizer/tokenizer_32k.json \
        --repo-type dataset \
        --local-dir . 2>/dev/null; then

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

# ── 2c.5: prefixLM SFT + eval (best-effort download from HF) ─────────────────
# These files are produced by data/retrieval_scripts/slimpajama_prefixlm.py and
# uploaded to the same HF repo as the pretrain pack. On a fresh pod we try the
# fast path; if the files aren't on HF yet, just print a hint — they aren't a
# hard prerequisite for pretrain runs and the user can generate them later.
echo ""
echo "── 2c.5: prefixLM SFT + eval data (fast path from HF) ──"
if [ -f "data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl" ] && \
   [ -f "data/Pretrain/slimpajama_prefixlm_eval_packed.jsonl" ]; then
    echo "  prefixLM packed data already exists locally, skipping."
elif hf download "${HF_DATASET_REPO}" \
        data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl \
        data/Pretrain/slimpajama_prefixlm_eval_packed.jsonl \
        --repo-type dataset \
        --local-dir . 2>/dev/null; then
    echo "  ✓ Downloaded prefixLM packed data from ${HF_DATASET_REPO}"
else
    echo "  (prefixLM packed data not on HF yet — generate with:"
    echo "    python data/retrieval_scripts/slimpajama_prefixlm.py --pack)"
fi

# ── 2d: SFT data (UltraChat) ────────────────────────────────────────────────
# DISABLED: switched to prefixLM SFT on packed SlimPajama (step 2c.5 above).
# UltraChat is no longer used by any active workflow — both --run-sft and the
# new post_hoc_prefixlm.py pull from data/Pretrain/slimpajama_prefixlm_*_packed.jsonl.
# Re-enable this block if you ever want chat-style SFT data again.
#
# echo ""
# echo "── 2d: SFT data (UltraChat 50M tokens) ──"
# mkdir -p data/SFT
# if [ -f "data/SFT/ultrachat.jsonl" ] && [ -f "data/SFT/ultrachat_eval.jsonl" ]; then
#     echo "  SFT data already exists, skipping."
# else
#     echo "  Tokenizing UltraChat with tokenizer/tokenizer_32k.json..."
#     python data/retrieval_scripts/ultrachat.py \
#         --tokenizer tokenizer/tokenizer_32k.json \
#         --target-tokens 50000000 \
#         --out-dir data/SFT
# fi

TRAIN_LINES=$(wc -l < data/Pretrain/slimpajama_6b_packed.jsonl)
SFT_LINES=$(wc -l < data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl 2>/dev/null || echo "0")
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Data pipeline complete ✓"
echo "  Packed pretrain sequences: $TRAIN_LINES"
echo "  Packed prefixLM SFT sequences: $SFT_LINES"
echo "  Next: python scripts/parallel_scaling.py ..."
echo "═══════════════════════════════════════════════════════════════"
