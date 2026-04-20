#!/usr/bin/env python
# Retrieve 500M lines from SlimPajama CommonCrawl
# ------------------------------------------------------------------

from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import json
import sys
import re
import unicodedata
import subprocess
import numpy as np


def n_tokens(txt:str) -> int:
    return len(txt) // 4 # 4 chars per token, on average

# -------- settings -------------------------------------------------
OUT_DIR = Path("data/Pretrain")
OUT_DIR.mkdir(exist_ok=True)

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--tokens", type=int, default=50_000_000, help="Target token count (default: 50M)")
_args, _ = _parser.parse_known_args()
target_tokens = _args.tokens
tok_m = target_tokens // 1_000_000

token_counts = {
    "RedPajamaCommonCrawl": {"count": 0.5 * target_tokens, "so_far": 0},
    "RedPajamaC4": {"count": 0.2 * target_tokens, "so_far": 0},
    "RedPajamaGithub": {"count": 0.1 * target_tokens, "so_far": 0},
    "RedPajamaArXiv": {"count": 0.1 * target_tokens, "so_far": 0},
    "RedPajamaBook": {"count": 0.1 * target_tokens, "so_far": 0}
}
# -------- main loop ------------------------------------------------
print(f"▶ Retrieving {tok_m}M tokens from SlimPajama")

try:
    ds = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

tok_so_far = 0
output_file = OUT_DIR / f"slimpajama.jsonl"
test_file = OUT_DIR / f"slimpajama_eval.jsonl"

print(f"Writing base to {output_file}")

#Base dataset
with output_file.open("w", encoding="utf-8") as f, test_file.open("w") as tgt:
    prev_tokens = 0
    for ex in ds:  
        #Break when we've reached all of our tokens! 
        if tok_so_far >= target_tokens:
            break
        set_name = ex.get('meta', {}).get('redpajama_set_name')
        
        # Take enough examples from each! 
        if set_name not in token_counts.keys() or token_counts[set_name]["so_far"] >= token_counts[set_name]["count"]:
            continue
        text = unicodedata.normalize("NFKC", ex["text"]) #Normalizing text
        if np.random.random() < 0.001:
            tgt.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            continue
        
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        tok_so_far += n_tokens(text)
        token_counts[set_name]["so_far"] += n_tokens(text)
        
        if tok_so_far - prev_tokens >= 1_000_000:
            print(f"  ✓ {tok_so_far:,} tokens written")
            prev_tokens = tok_so_far

print(f"Done: {tok_so_far:,} tokens written")
import os
os._exit(0)

