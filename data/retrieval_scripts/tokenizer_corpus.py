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
import time


def _fmt_time(seconds):
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=500_000_000,
                        help="Target token count for corpus (default: 500M)")
    parser.add_argument("--output", type=str, default="data/Pretrain/tokenizer_corpus.jsonl")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"▶ Streaming ~{args.tokens // 1_000_000}M tokens from DKYoon/SlimPajama-6B for tokenizer training")

    ds = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)

    tok_so_far = 0
    docs_written = 0
    start_time = time.time()
    last_report_time = start_time

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
            docs_written += 1

            now = time.time()
            if now - last_report_time >= 10:  # report every 10 seconds
                elapsed = now - start_time
                pct = 100 * tok_so_far / args.tokens
                toks_per_sec = tok_so_far / max(elapsed, 1)
                remaining = (args.tokens - tok_so_far) / max(toks_per_sec, 1)
                print(f"  [{pct:5.1f}%] {tok_so_far:>13,} / {args.tokens:,} tokens | "
                      f"{toks_per_sec:,.0f} tok/s | "
                      f"elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(remaining)} | "
                      f"{docs_written:,} docs")
                last_report_time = now

    elapsed = time.time() - start_time
    print(f"\nDone: {tok_so_far:,} tokens in {docs_written:,} docs written to {out_path}")
    print(f"Total time: {_fmt_time(elapsed)} ({tok_so_far / max(elapsed, 1):,.0f} tok/s avg)")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
