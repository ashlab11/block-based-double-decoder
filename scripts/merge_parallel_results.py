#!/usr/bin/env python3
"""Merge per-run JSONs from checkpoints/parallel_scaling/ into one grid file.

Each tier (L40 / H100 / B200) of the large sweep writes per-run results to
the same folder using filenames {model_type}_{arch}_{tok}tok_results.json.
This script scans those files and rebuilds the unified
scaling_grid_results.json that downstream tools (scripts/scaling_laws.py
collect, plot scripts) expect.

Usage:
    python scripts/merge_parallel_results.py
    python scripts/merge_parallel_results.py --in-dir checkpoints/parallel_scaling
    python scripts/merge_parallel_results.py --out other.json
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "checkpoints" / "parallel_scaling"

# Match {model_type}_{arch_label}_{tok_label}tok_results.json
# arch_label can be "5M", "0.6M", "150M", etc.; tok_label likewise.
RE_RUN = re.compile(r"^(dd|sed|dec)_([\d.]+M)_([\d.]+[MB])tok_results\.json$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default=str(DEFAULT_DIR))
    p.add_argument("--out", type=str, default=None,
                   help="Output path (default: <in-dir>/scaling_grid_results.json)")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else in_dir / "scaling_grid_results.json"

    if not in_dir.exists():
        raise SystemExit(f"Input dir does not exist: {in_dir}")

    grid = {}
    n_files = 0
    for fpath in sorted(in_dir.glob("*_results.json")):
        m = RE_RUN.match(fpath.name)
        if not m:
            continue  # skip aggregates / unrelated files
        model_type, arch_label, tok_label = m.group(1), m.group(2), m.group(3)
        with open(fpath) as f:
            data = json.load(f)
        display = f"{model_type}_{arch_label}"
        grid.setdefault(tok_label, {})[display] = data
        n_files += 1

    with open(out_path, "w") as f:
        json.dump(grid, f, indent=2)

    print(f"Merged {n_files} per-run JSONs from {in_dir}")
    print(f"  Token budgets: {sorted(grid.keys())}")
    print(f"  Per-budget runs: " +
          ", ".join(f"{tk}={len(v)}" for tk, v in sorted(grid.items())))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
