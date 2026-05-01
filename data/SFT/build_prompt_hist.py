#!/usr/bin/env python
"""Build the SFT prompt-length histogram used by DD's prompt_style boundary.

Reads data/SFT/ultrachat.jsonl (built by data/retrieval_scripts/ultrachat.py),
extracts the length of `input_ids` (the user-side prompt prefix) for each
example, and writes a 1D int64 numpy array of those lengths to
data/SFT/prompt_length_hist.npy.

The DDPretrainCollator's `boundary_strategy="prompt_style"` mode samples
from this array, so pretraining sees boundary positions whose distribution
matches the SFT prompt/response split — defusing the "you over-randomized"
reviewer concern.

Usage:
    python data/SFT/build_prompt_hist.py
    python data/SFT/build_prompt_hist.py --input data/SFT/ultrachat.jsonl
                                         --output data/SFT/prompt_length_hist.npy
                                         --max-len 2047
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/SFT/ultrachat.jsonl",
                        help="UltraChat jsonl with {input_ids, output_ids} per line")
    parser.add_argument("--output", default="data/SFT/prompt_length_hist.npy",
                        help="Where to write the int64 array of prompt lengths")
    parser.add_argument("--max-len", type=int, default=2047,
                        help="Drop examples whose input length exceeds this "
                             "(typically seq_len - 1 to leave room for at "
                             "least one response token)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: {inp} not found. Build the SFT data first:")
        print(f"  python data/retrieval_scripts/ultrachat.py "
              f"--tokenizer tokenizer/tokenizer_32k.json")
        sys.exit(1)

    lengths = []
    n_skipped_long = 0
    n_total = 0
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            ex = json.loads(line)
            # input_ids is the prompt-side context (user turn[s]); its length
            # is the position where the assistant response starts. Adding +1
            # would include the BOS the SFT collator prepends, but we want
            # the boundary *before* the response so leave as-is.
            L = len(ex.get("input_ids", []))
            if L < 2:
                continue  # too short to be a useful split position
            if L > args.max_len:
                n_skipped_long += 1
                continue
            lengths.append(L)

    if not lengths:
        print(f"ERROR: no usable examples in {inp} "
              f"(scanned {n_total}, skipped {n_skipped_long} > max-len)")
        sys.exit(1)

    arr = np.array(lengths, dtype=np.int64)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)

    # Print summary so user can sanity-check the distribution before running
    # any DD training with prompt_style — a degenerate hist (all same value,
    # all very short, etc.) would silently make pretraining trivial.
    print(f"Wrote {arr.size:,} prompt lengths to {out}")
    print(f"  min={arr.min()}  max={arr.max()}  "
          f"mean={arr.mean():.1f}  median={int(np.median(arr))}")
    pcts = [10, 25, 50, 75, 90, 95, 99]
    pct_str = "  ".join(f"p{p}={int(np.percentile(arr, p))}" for p in pcts)
    print(f"  {pct_str}")
    if n_skipped_long:
        print(f"  ({n_skipped_long:,} examples dropped for exceeding "
              f"max-len={args.max_len})")


if __name__ == "__main__":
    main()
