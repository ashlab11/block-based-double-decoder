#!/usr/bin/env python
# ------------------------------------------------------------------

import argparse
from datasets import load_dataset
from pathlib import Path
import json
import sys
import re
import unicodedata
import numpy as np
from transformers import PreTrainedTokenizerFast


# -------- settings -------------------------------------------------
parser = argparse.ArgumentParser(description="Retrieve and tokenize UltraChat for SFT.")
parser.add_argument("--tokenizer", default="tokenizer/tokenizer_32k.json",
                    help="Tokenizer file (must match the one used for pretraining)")
parser.add_argument("--target-tokens", type=int, default=50_000_000)
parser.add_argument("--out-dir", default="data/SFT")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
target_tokens = args.target_tokens
# Maximum total tokens for a single training pair (includes +1 BOS and +1 EOS added later)
SEQ_LEN = 2048
# Batch size for on-the-fly tokenization
BATCH_SIZE = 64

# -------- main loop ------------------------------------------------
print(f"▶ Retrieving {target_tokens // 1_000_000}M tokens from Ultrachat")
print(f"  Tokenizer: {args.tokenizer}")

try:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

tok_so_far    = 0
output_file   = OUT_DIR / "ultrachat.jsonl"
test_file     = OUT_DIR / "ultrachat_eval.jsonl"

# Initialize tokenizer once. Default is tokenizer_32k.json (matches what
# parallel_scaling.py and the pretraining configs use). The legacy
# tokenizer.json has a different (smaller) vocab and would silently produce
# wrong token IDs.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=args.tokenizer
)

print(f"Writing to {output_file}")

with output_file.open("w", encoding="utf-8") as f, test_file.open("w") as tgt:
    try:
        # Buffers for batched tokenization
        buf_inputs = []
        buf_outputs = []
        buf_files = []  # parallel to know where to write each pair

        def process_buffer():
            if not buf_inputs:
                return False, 0  # not done, no tokens added
            tokens_added = 0
            tokenized_inputs = tokenizer(buf_inputs, add_special_tokens=False)
            tokenized_outputs = tokenizer(buf_outputs, add_special_tokens=False)
            for ids_inp, ids_out, fh in zip(tokenized_inputs["input_ids"], tokenized_outputs["input_ids"], buf_files):
                # Include BOS and EOS that collator adds later
                total_len = (len(ids_inp) + 1) + (len(ids_out) + 1)
                if total_len <= SEQ_LEN and tok_so_far + tokens_added < target_tokens:
                    fh.write(json.dumps({
                        "input_ids": ids_inp,
                        "output_ids": ids_out,
                    }, ensure_ascii=False) + "\n")
                    # ONLY count the out tokens, since that's all we're predicting
                    tokens_added += len(ids_out)
                    if tok_so_far + tokens_added >= target_tokens:
                        buf_inputs.clear()
                        buf_outputs.clear()
                        buf_files.clear()
                        return True, tokens_added  # done, tokens added
            buf_inputs.clear()
            buf_outputs.clear()
            buf_files.clear()
            return False, tokens_added

        for ex in ds:
            file = tgt if np.random.random() < 0.01 else f
            if tok_so_far >= target_tokens:
                break
            messages = ex["messages"]
            context = ""
            for message in messages:
                if message['role'] == 'user':
                    if context:
                        context += " "
                    context += f"<user> {message['content']}"
                else:
                    # Collect pair for batch tokenization
                    buf_inputs.append(context)
                    buf_outputs.append("<assistant> " + message['content'])
                    buf_files.append(file)
                    # Update running context
                    context += " " + f"<assistant> {message['content']}"
                    # Process batch if ready
                    if len(buf_inputs) >= BATCH_SIZE:
                        done, added = process_buffer()
                        tok_so_far += added
                        if done:
                            break
            # Early exit if token budget reached during inner loop
            if tok_so_far >= target_tokens:
                break

        # Process any remaining buffered pairs
        if tok_so_far < target_tokens:
            done, added = process_buffer()
            tok_so_far += added

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        print(f"\nProcessed tokens (approx incl. BOS/EOS): {tok_so_far:,}")

if tok_so_far >= target_tokens:
    print(f"\nSuccess! {tok_so_far:,} tokens written to {output_file}")
else:
    print(f"\nWarning: Only {tok_so_far:,} tokens written (target was {target_tokens:,})")
