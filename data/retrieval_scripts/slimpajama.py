#!/usr/bin/env python
"""Download tokens from SlimPajama with domain-proportional sampling.

For small runs (<=6B tokens), uses DKYoon/SlimPajama-6B.
For larger runs, uses cerebras/SlimPajama-627B.

Usage:
    python data/retrieval_scripts/slimpajama.py --tokens 50000000
    python data/retrieval_scripts/slimpajama.py --tokens 20000000000 --output-prefix slimpajama_20b
"""

from datasets import load_dataset
from pathlib import Path
import json
import sys
import unicodedata
import numpy as np
import argparse
from tqdm import tqdm


def n_tokens(txt: str) -> int:
    return len(txt) // 4  # ~4 chars per token on average


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50_000_000,
                        help="Target token count (default: 50M)")
    parser.add_argument("--output-prefix", type=str, default="slimpajama",
                        help="Output file prefix (default: slimpajama)")
    parser.add_argument("--output-dir", type=str, default="data/Pretrain")
    args = parser.parse_args()

    target_tokens = args.tokens
    tok_m = target_tokens // 1_000_000
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Domain-proportional sampling
    token_counts = {
        "RedPajamaCommonCrawl": {"count": 0.5 * target_tokens, "so_far": 0},
        "RedPajamaC4":          {"count": 0.2 * target_tokens, "so_far": 0},
        "RedPajamaGithub":      {"count": 0.1 * target_tokens, "so_far": 0},
        "RedPajamaArXiv":       {"count": 0.1 * target_tokens, "so_far": 0},
        "RedPajamaBook":        {"count": 0.1 * target_tokens, "so_far": 0},
    }

    # Use the small dataset for <=6B tokens, full dataset for larger
    if target_tokens <= 6_000_000_000:
        dataset_name = "DKYoon/SlimPajama-6B"
    else:
        dataset_name = "cerebras/SlimPajama-627B"

    print(f"▶ Retrieving {tok_m}M tokens from {dataset_name}")

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    output_file = out_dir / f"{args.output_prefix}.jsonl"
    eval_file = out_dir / f"{args.output_prefix}_eval.jsonl"

    print(f"Writing to {output_file}")

    tok_so_far = 0
    pbar = tqdm(total=target_tokens, unit="tok", unit_scale=True, desc="Downloading")

    with output_file.open("w", encoding="utf-8") as f, eval_file.open("w", encoding="utf-8") as tgt:
        for ex in ds:
            if tok_so_far >= target_tokens:
                break

            # cerebras/SlimPajama-627B uses ex["meta"] as a string, not a dict
            meta = ex.get("meta", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            set_name = meta.get("redpajama_set_name")

            if set_name not in token_counts or token_counts[set_name]["so_far"] >= token_counts[set_name]["count"]:
                continue

            text = unicodedata.normalize("NFKC", ex["text"])
            n_tok = n_tokens(text)

            # 0.1% holdout for eval
            if np.random.random() < 0.001:
                tgt.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                continue

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            tok_so_far += n_tok
            token_counts[set_name]["so_far"] += n_tok
            pbar.update(n_tok)

    pbar.close()
    print(f"\nDone: {tok_so_far:,} tokens written")
    print("Domain breakdown:")
    for name, info in token_counts.items():
        pct = 100 * info["so_far"] / max(1, tok_so_far)
        print(f"  {name}: {info['so_far']:,} tokens ({pct:.1f}%)")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
