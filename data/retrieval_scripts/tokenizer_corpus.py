#!/usr/bin/env python
"""Stream text from SlimPajama-627B to build a tokenizer training corpus.

Usage:
    python data/retrieval_scripts/tokenizer_corpus.py --tokens 500000000
"""

from datasets import load_dataset
from pathlib import Path
import json
import argparse
import unicodedata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=500_000_000,
                        help="Target token count for corpus (default: 500M)")
    parser.add_argument("--output", type=str, default="data/Pretrain/tokenizer_corpus.jsonl")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"▶ Streaming ~{args.tokens // 1_000_000}M tokens from cerebras/SlimPajama-627B for tokenizer training")

    ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

    tok_so_far = 0
    prev_tokens = 0

    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            if tok_so_far >= args.tokens:
                break
            text = unicodedata.normalize("NFKC", ex["text"])
            n_tok = len(text) // 4
            if n_tok < 10:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            tok_so_far += n_tok

            if tok_so_far - prev_tokens >= 10_000_000:
                print(f"  ✓ {tok_so_far:,} tokens written")
                prev_tokens = tok_so_far

    print(f"Done: {tok_so_far:,} tokens written to {out_path}")

if __name__ == "__main__":
    main()
