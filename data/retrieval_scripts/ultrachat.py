#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SEQ_LEN = 2048
BATCH_SIZE = 64


def main():
    from datasets import load_dataset
    from transformers import PreTrainedTokenizerFast

    parser = argparse.ArgumentParser(description="Retrieve and tokenize UltraChat for SFT.")
    parser.add_argument("--out-dir", default="data/SFT")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_32k.json",
                        help="Tokenizer file (must match the one used for pretraining)")
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "ultrachat.jsonl"
    test_file = out_dir / "ultrachat_eval.jsonl"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)

    print(f"▶ Retrieving {args.target_tokens // 1_000_000}M tokens from Ultrachat")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Writing to {output_file}")

    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        sys.exit(1)

    tok_so_far = 0
    with output_file.open("w", encoding="utf-8") as f, test_file.open("w", encoding="utf-8") as tgt:
        buf_inputs, buf_outputs, buf_files = [], [], []

        def process_buffer():
            nonlocal tok_so_far
            if not buf_inputs:
                return False
            ins = tokenizer(buf_inputs, add_special_tokens=False)["input_ids"]
            outs = tokenizer(buf_outputs, add_special_tokens=False)["input_ids"]
            for ids_inp, ids_out, fh in zip(ins, outs, buf_files):
                if (len(ids_inp) + len(ids_out) + 2) <= SEQ_LEN and tok_so_far < args.target_tokens:
                    fh.write(json.dumps({"input_ids": ids_inp, "output_ids": ids_out}, ensure_ascii=False) + "\n")
                    tok_so_far += len(ids_out)
                    if tok_so_far >= args.target_tokens:
                        break
            buf_inputs.clear()
            buf_outputs.clear()
            buf_files.clear()
            return tok_so_far >= args.target_tokens

        try:
            for ex in ds:
                if tok_so_far >= args.target_tokens:
                    break
                target = tgt if np.random.random() < 0.01 else f
                context = ""
                for message in ex["messages"]:
                    if message["role"] == "user":
                        context = f"{context} <user> {message['content']}".strip()
                    else:
                        buf_inputs.append(context)
                        buf_outputs.append(f"<assistant> {message['content']}")
                        buf_files.append(target)
                        context = f"{context} <assistant> {message['content']}".strip()
                        if len(buf_inputs) >= BATCH_SIZE and process_buffer():
                            break
            process_buffer()
        finally:
            print(f"\nProcessed tokens (approx incl. BOS/EOS): {tok_so_far:,}")

    if tok_so_far >= args.target_tokens:
        print(f"\nSuccess! {tok_so_far:,} tokens written to {output_file}")
    else:
        print(f"\nWarning: Only {tok_so_far:,} tokens written (target was {args.target_tokens:,})")


if __name__ == "__main__":
    main()
