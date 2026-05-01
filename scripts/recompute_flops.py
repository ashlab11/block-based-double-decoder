#!/usr/bin/env python3
"""Backfill arch-aware FLOP fields onto existing per-run JSONs.

Pre-#1 runs only logged `non_emb_params` and `tokens_seen`, with the implicit
assumption that compute = 6 * N * T regardless of architecture. That's wrong
by 1.33× for DD (encoder pass + combo cross-attn) and 0.94× for SED (sparse
decoder output). This script rewrites those JSONs in place to add:

    flops_arch              — architecture-aware FLOP estimate
    flops_naive_6NT         — the old-style 6NT estimate (kept for comparison)
    flop_arch_multiplier    — k_arch s.t. flops_arch = k_arch * 6 * N * T

`flops_naive_6NT` is preserved so plots that previously used `6 * N * T` can
verify their earlier numbers exactly. Idempotent — running twice is a no-op.

Usage:
    # Default: walks checkpoints/parallel_scaling/*.json
    python scripts/recompute_flops.py

    # Custom directory or single file:
    python scripts/recompute_flops.py --dir checkpoints/scaling
    python scripts/recompute_flops.py --file checkpoints/parallel_scaling/dd_5M_100Mtok_results.json

    # Dry-run: print what would change, don't write
    python scripts/recompute_flops.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.scaling import compute_flops_arch, arch_flop_multiplier


def _patch_one(path: Path, dry_run=False):
    """Mutate a single per-run JSON. Returns one of:
        "patched"  — fields added/updated
        "skipped"  — already has correct fields
        "noinfer"  — couldn't infer model_type (file from a different format)
        "error"    — couldn't read/parse the file
    """
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  [error] {path.name}: {e}")
        return "error"

    # Per-run JSONs at the top level have model_type + non_emb_params + tokens_seen.
    # Combined parallel_*tok_*_results.json files are dicts keyed by model name.
    if isinstance(data, dict) and "model_type" in data and "non_emb_params" in data:
        runs = [(None, data)]
    elif isinstance(data, dict):
        runs = [(k, v) for k, v in data.items() if isinstance(v, dict)
                and "model_type" in v and "non_emb_params" in v]
    else:
        return "noinfer"

    changed = False
    for key, run in runs:
        mt = run.get("model_type")
        ne = run.get("non_emb_params")
        toks = run.get("tokens_seen")
        if mt is None or ne is None or toks is None:
            continue
        # Derive enc/dec from logged hparams; fall back to defaults if missing
        h = run.get("hparams", {})
        enc = h.get("num_encoder_layers", 8)
        dec = h.get("num_decoder_layers", 4)

        new_arch = compute_flops_arch(mt, ne, toks, enc=enc, dec=dec)
        new_naive = 6 * ne * toks
        new_mult = arch_flop_multiplier(mt, enc=enc, dec=dec)

        if (run.get("flops_arch") == new_arch
                and run.get("flops_naive_6NT") == new_naive
                and abs(run.get("flop_arch_multiplier", 0) - new_mult) < 1e-9):
            continue  # already up-to-date

        run["flops_arch"] = new_arch
        run["flops_naive_6NT"] = new_naive
        run["flop_arch_multiplier"] = new_mult
        changed = True

    if not changed:
        return "skipped"
    if not dry_run:
        path.write_text(json.dumps(data, indent=2))
    return "patched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="checkpoints/parallel_scaling",
                        help="Directory to walk recursively for *.json files")
    parser.add_argument("--file", default=None,
                        help="Single JSON file to patch (overrides --dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions but do not write any file")
    args = parser.parse_args()

    if args.file:
        targets = [Path(args.file)]
    else:
        d = Path(args.dir)
        if not d.exists():
            print(f"Directory {d} does not exist; nothing to do.")
            return
        targets = sorted(d.rglob("*.json"))

    if not targets:
        print(f"No JSON files found.")
        return

    counts = {"patched": 0, "skipped": 0, "noinfer": 0, "error": 0}
    for p in targets:
        result = _patch_one(p, dry_run=args.dry_run)
        counts[result] += 1
        if result == "patched":
            print(f"  [{'DRY' if args.dry_run else 'patched'}] {p}")

    print(f"\nDone. patched={counts['patched']}  skipped={counts['skipped']}  "
          f"noinfer={counts['noinfer']}  error={counts['error']}  "
          f"(scanned {len(targets)} files)")


if __name__ == "__main__":
    main()
