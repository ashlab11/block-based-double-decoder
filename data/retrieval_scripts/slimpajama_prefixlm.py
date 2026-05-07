#!/usr/bin/env python3
"""Pull SlimPajama prefixLM corpora from gmongaras/SlimPajama-627B_Reupload.

Pulls 400M tokens from the `validation` split → SFT-train, and 100M tokens
from `test` → held-out eval. Both splits are disjoint from `train` by
construction at upload time, and `DKYoon/SlimPajama-6B` (the source of our
6B pretrain corpus) sampled from `train`. So:

    cerebras/SlimPajama-627B
        ├── train       → DKYoon sampled here → our pretrain
        ├── validation  → SFT-train (this script)         ← disjoint by
        └── test        → held-out eval (this script)     ← construction

Disjointness is verified at the end via a 10K-doc spot-check: stream that
many docs from DKYoon and check each against the union of SFT+eval hashes.
0 overlaps confirms the chain.

Outputs (compatible with data/retrieval_scripts/pack_dataset.py):
    data/Pretrain/slimpajama_prefixlm_sft.jsonl            (raw text)
    data/Pretrain/slimpajama_prefixlm_eval.jsonl           (raw text)
    data/Pretrain/slimpajama_prefixlm_sft_packed.jsonl     (token IDs, with --pack)
    data/Pretrain/slimpajama_prefixlm_eval_packed.jsonl    (token IDs, with --pack)

With `--hf-repo` set (default: bpbradle/slimpajama-6b-packed — same repo
2_data.sh pulls pretrain packs from), the *_packed.jsonl files get uploaded
to HF Hub at the end. Subsequent pod clones can then pull them via the
same fast path as the pretrain data instead of regenerating.

If the local *_packed.jsonl files already exist, this script no-ops on
the relevant phase (idempotent).

Wall-clock on a fast pod with HF_HUB_ENABLE_HF_TRANSFER=1: ~5-15 min for
download + dedup; pack adds ~1-2 hr (single-threaded BPE).

Why no parallel sharding: empirically tested IterableDataset.shard() at
4-way and 8-way; both made throughput strictly worse (per-worker setup
overhead + HF connection contention dominate). Single fast stream wins.

Usage:
    # Default: download + spot-check, print pack command
    python data/retrieval_scripts/slimpajama_prefixlm.py

    # Full pipeline including pack + HF upload
    python data/retrieval_scripts/slimpajama_prefixlm.py --pack

    # Custom token budgets
    python data/retrieval_scripts/slimpajama_prefixlm.py \
        --sft-tokens 500000000 --eval-tokens 100000000 --pack

    # Skip HF upload entirely
    python data/retrieval_scripts/slimpajama_prefixlm.py --pack --hf-repo ""

    # Smoke test (50M SFT + 10M eval, ~3 min on a fast pod)
    python data/retrieval_scripts/slimpajama_prefixlm.py \
        --sft-tokens 50000000 --eval-tokens 10000000 --pack
"""

import argparse
import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# Default to fast download — required for >2x speedup on HF transfers.
# Setting BEFORE importing datasets ensures the env var is picked up.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import xxhash  # noqa: E402
from datasets import load_dataset  # noqa: E402

CHARS_PER_TOKEN = 4  # matches slimpajama.py heuristic; tokenizer-agnostic estimate

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ────────────────────────────────────────────────────────────────

def norm_hash(text: str) -> int:
    """64-bit hash of NFKC-normalized text. Same normalization as slimpajama.py
    so a doc that round-trips through both pipelines produces an identical hash."""
    return xxhash.xxh3_64(unicodedata.normalize("NFKC", text).encode("utf-8")).intdigest()


def fmt_n(n: float) -> str:
    if n < 1e3:  return f"{n:.0f}"
    if n < 1e6:  return f"{n/1e3:.1f}K"
    if n < 1e9:  return f"{n/1e6:.1f}M"
    return f"{n/1e9:.2f}B"


def fmt_time(s: float) -> str:
    if s < 60:    return f"{s:.0f}s"
    if s < 3600:  return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


# ── Phase 1+2: stream + write ──────────────────────────────────────────────

def stream_and_write(source, split, target_tokens, out_path,
                     existing_hashes=None, log_every_tokens=20_000_000):
    """Stream documents from `source/split` until target_tokens accumulated;
    write JSONL with one normalized {"text": ...} per line.

    Skips intra-split duplicates (same text seen twice in this stream) and
    cross-split duplicates (text already in `existing_hashes`, defensive
    against the uploader having put a doc into both validation and test).

    Returns: dict with keys {tokens, docs, elapsed_s, hashes, domains, dup_skipped, cross_skipped}.
    """
    print(f"\n┌─ pulling {fmt_n(target_tokens)} tokens: {source} [{split}]", flush=True)
    print(f"│   → {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(source, split=split, streaming=True)
    seen = set()
    domains = {}
    n_docs = n_dup = n_cross = 0
    n_chars = 0
    last_log = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            text = ex["text"]
            if not text or len(text) < 4:
                continue

            # Normalize once, then hash + write the same normalized form so
            # downstream pipeline sees what we hashed.
            norm = unicodedata.normalize("NFKC", text)
            h = xxhash.xxh3_64(norm.encode("utf-8")).intdigest()

            if h in seen:
                n_dup += 1
                continue
            if existing_hashes is not None and h in existing_hashes:
                n_cross += 1
                continue
            seen.add(h)

            f.write(json.dumps({"text": norm}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += len(norm)

            # Domain tracking
            meta = ex.get("meta", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            d = meta.get("redpajama_set_name", "?")
            domains[d] = domains.get(d, 0) + 1

            tok_so_far = n_chars // CHARS_PER_TOKEN
            if tok_so_far >= last_log + log_every_tokens:
                last_log = tok_so_far
                pct = 100 * tok_so_far / target_tokens
                tps = tok_so_far / max(time.time() - t0, 1e-3)
                eta = (target_tokens - tok_so_far) / max(tps, 1)
                print(f"│   [{pct:5.1f}%] {fmt_n(tok_so_far):>6} tok | "
                      f"{n_docs:,} docs | {tps:>9,.0f} tok/s | ETA {fmt_time(eta)}",
                      flush=True)

            if tok_so_far >= target_tokens:
                break

    elapsed = time.time() - t0
    tok_total = n_chars // CHARS_PER_TOKEN
    print(f"│   ✓ {fmt_n(tok_total)} tokens in {fmt_time(elapsed)} "
          f"({tok_total/max(elapsed,1e-3):,.0f} tok/s)", flush=True)
    if n_dup:
        print(f"│   skipped {n_dup:,} intra-split duplicates")
    if n_cross:
        print(f"│   skipped {n_cross:,} cross-split duplicates")
    print(f"│   domains: " + ", ".join(
        f"{k.replace('RedPajama','')}: {v}"
        for k, v in sorted(domains.items(), key=lambda kv: -kv[1])))
    print(f"└──────────────────────────────────────────────────────────────────────")

    return {"tokens": tok_total, "docs": n_docs, "elapsed_s": elapsed,
            "hashes": seen, "domains": domains,
            "dup_skipped": n_dup, "cross_skipped": n_cross}


# ── Phase 3: spot-check disjointness against pretrain source ───────────────

def spot_check(source, split, sample_n, target_hashes):
    """Stream sample_n docs from a known-pretrain-source dataset, count how
    many appear in target_hashes. Returns (overlap_count, sample_n, elapsed_s,
    overlap_examples)."""
    print(f"\n┌─ spot-check: {sample_n:,} docs from {source} [{split}]", flush=True)
    print(f"│   testing against {len(target_hashes):,} candidate hashes "
          f"(SFT ∪ eval)", flush=True)
    ds = load_dataset(source, split=split, streaming=True)
    overlaps = []
    n = 0
    t0 = time.time()
    for ex in ds:
        if n >= sample_n:
            break
        if norm_hash(ex["text"]) in target_hashes:
            overlaps.append(ex["text"][:120])
        n += 1
        if n % 2000 == 0:
            print(f"│   [{n:>5,}/{sample_n:,}] {time.time()-t0:.1f}s, "
                  f"{len(overlaps)} overlaps so far", flush=True)
    elapsed = time.time() - t0
    rate = 100 * len(overlaps) / max(n, 1)
    print(f"│   ✓ {n:,} docs scanned in {fmt_time(elapsed)} | "
          f"{len(overlaps)} overlaps ({rate:.4f}%)", flush=True)
    print(f"└──────────────────────────────────────────────────────────────────────")
    return len(overlaps), n, elapsed, overlaps


# ── Optional: invoke pack_dataset.py to chain into the existing pipeline ───

def upload_to_hf(paths, repo_id, repo_type="dataset", path_in_repo_prefix="data/Pretrain"):
    """Upload each path to HF Hub at <path_in_repo_prefix>/<basename>.

    Mirrors 2_data.sh's pull pattern: pretrain pulls
    `data/Pretrain/slimpajama_6b_packed.jsonl` from the dataset repo, so we
    upload to the same `data/Pretrain/` prefix to keep the layout symmetric.
    """
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import HfHubHTTPError

    print(f"\n┌─ uploading to HF Hub: {repo_id} (repo_type={repo_type})", flush=True)
    api = HfApi()

    # Create repo if it doesn't exist (idempotent — no-op if it does).
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    except HfHubHTTPError as e:
        print(f"│   ⚠ create_repo failed (continuing — repo may already exist): {e}",
              flush=True)

    for p in paths:
        if not p.exists():
            print(f"│   ✗ skipping {p} — file does not exist", flush=True)
            continue
        size_mb = p.stat().st_size / 1e6
        path_in_repo = f"{path_in_repo_prefix}/{p.name}"
        print(f"│   ↑ {p.name} ({size_mb:.1f} MB) → {repo_id}:{path_in_repo}", flush=True)
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Add {p.name} (slimpajama_prefixlm.py)",
            )
            print(f"│     ✓ uploaded in {fmt_time(time.time() - t0)} "
                  f"({size_mb / max(time.time() - t0, 1e-3):.1f} MB/s)", flush=True)
        except Exception as e:
            print(f"│     ✗ upload FAILED: {e}", flush=True)
            print(f"│     (you can retry manually: hf upload {repo_id} {p} {path_in_repo} --repo-type {repo_type})",
                  flush=True)
    print(f"└──────────────────────────────────────────────────────────────────────")


def run_pack(sft_jsonl, eval_jsonl, tokenizer):
    print(f"\n┌─ packing into fixed-length sequences via pack_dataset.py", flush=True)
    sft_packed = sft_jsonl.with_name(sft_jsonl.stem + "_packed.jsonl")
    eval_packed = eval_jsonl.with_name(eval_jsonl.stem + "_packed.jsonl")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data" / "retrieval_scripts" / "pack_dataset.py"),
        "--tokenizer", str(tokenizer),
        "--input", str(sft_jsonl),
        "--output", str(sft_packed),
        "--eval-input", str(eval_jsonl),
        "--eval-output", str(eval_packed),
    ]
    print(f"│   {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"│   ✓ packed in {fmt_time(time.time()-t0)}")
    print(f"│     → {sft_packed}")
    print(f"│     → {eval_packed}")
    print(f"└──────────────────────────────────────────────────────────────────────")
    return sft_packed, eval_packed


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="gmongaras/SlimPajama-627B_Reupload",
                   help="HF dataset to pull from")
    p.add_argument("--sft-split", default="validation")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--sft-tokens", type=int, default=400_000_000,
                   help="Target token count for SFT-train (default: 400M)")
    p.add_argument("--eval-tokens", type=int, default=100_000_000,
                   help="Target token count for eval (default: 100M)")
    p.add_argument("--out-dir", default="data/Pretrain")
    p.add_argument("--sft-filename", default="slimpajama_prefixlm_sft.jsonl")
    p.add_argument("--eval-filename", default="slimpajama_prefixlm_eval.jsonl")
    p.add_argument("--spotcheck-dataset", default="DKYoon/SlimPajama-6B",
                   help="Dataset to scan for the disjointness spot-check (= source of our pretrain)")
    p.add_argument("--spotcheck-split", default="train")
    p.add_argument("--spotcheck-n", type=int, default=10_000,
                   help="How many docs to sample for the spot-check (0 = skip).")
    p.add_argument("--max-overlap", type=int, default=0,
                   help="Abort with non-zero exit if spot-check finds more than this many overlaps.")
    p.add_argument("--pack", action="store_true",
                   help="After download, run pack_dataset.py to produce the *_packed.jsonl files.")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer_32k.json",
                   help="Tokenizer for --pack")
    p.add_argument("--hf-repo", default="bpbradle/slimpajama-6b-packed",
                   help="HF dataset repo to upload the *_packed.jsonl files to "
                        "(default matches 2_data.sh's pretrain repo so subsequent "
                        "clones can pull via the same fast path). Pass --hf-repo '' to "
                        "skip the upload entirely.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    sft_path = out_dir / args.sft_filename
    eval_path = out_dir / args.eval_filename
    sft_packed_path = sft_path.with_name(sft_path.stem + "_packed.jsonl")
    eval_packed_path = eval_path.with_name(eval_path.stem + "_packed.jsonl")

    print(f"\n=== SlimPajama prefixLM data download ===")
    print(f"  source:     {args.source}")
    print(f"  SFT-train:  {fmt_n(args.sft_tokens)} tokens from [{args.sft_split}] → {sft_path}")
    print(f"  eval:       {fmt_n(args.eval_tokens)} tokens from [{args.eval_split}] → {eval_path}")
    if args.spotcheck_n > 0:
        print(f"  spot-check: {args.spotcheck_n:,} docs from {args.spotcheck_dataset} [{args.spotcheck_split}]")
    if args.hf_repo:
        print(f"  hf upload:  {args.hf_repo} (after --pack)")
    print(f"  HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}")

    t_total = time.time()

    # Idempotency: skip the download phases entirely if both raw files (or
    # packed files, which supersede them) already exist locally.
    have_raw = sft_path.exists() and eval_path.exists()
    have_packed = sft_packed_path.exists() and eval_packed_path.exists()

    if have_packed:
        print(f"\n✓ both *_packed.jsonl already exist locally — skipping download/dedup/pack")
        print(f"    {sft_packed_path}")
        print(f"    {eval_packed_path}")
        if args.hf_repo:
            upload_to_hf([sft_packed_path, eval_packed_path], args.hf_repo)
        print(f"\n=== Done in {fmt_time(time.time() - t_total)} ===")
        return

    if have_raw:
        print(f"\n✓ raw text JSONLs already exist locally — skipping download/dedup")
        print(f"    {sft_path}")
        print(f"    {eval_path}")
        sft = ev = None  # signals "skipped"
    else:
        # Phase 1: SFT-train (largest pull)
        sft = stream_and_write(args.source, args.sft_split, args.sft_tokens, sft_path)

        # Phase 2: eval, with cross-split dedup against SFT (defensive — should be ~0)
        ev = stream_and_write(args.source, args.eval_split, args.eval_tokens, eval_path,
                              existing_hashes=sft["hashes"])

    # Phase 3: spot-check disjointness from pretrain source
    if args.spotcheck_n > 0 and sft is not None:
        combined = sft["hashes"] | ev["hashes"]
        n_over, n_samp, _, examples = spot_check(
            args.spotcheck_dataset, args.spotcheck_split,
            args.spotcheck_n, combined)
        if n_over > args.max_overlap:
            print(f"\n✗ FAIL: spot-check found {n_over} overlaps in {n_samp} samples "
                  f"({100*n_over/n_samp:.4f}%)")
            print(f"  Disjointness assumption violated. First 3 overlapping docs:")
            for s in examples[:3]:
                print(f"    {s!r}")
            print(f"\n  This means {args.source}/{args.sft_split} (or /{args.eval_split}) is")
            print(f"  NOT held out from {args.spotcheck_dataset}. You may need a different source")
            print(f"  or a stricter dedup pass.")
            sys.exit(1)
        else:
            print(f"\n✓ disjointness verified: {n_over}/{n_samp} overlaps "
                  f"(threshold: ≤{args.max_overlap})")

    # Optional: pack now
    if args.pack:
        run_pack(sft_path, eval_path, Path(args.tokenizer))

        # Upload packed files to HF Hub (default: same repo 2_data.sh pulls from)
        if args.hf_repo:
            upload_to_hf([sft_packed_path, eval_packed_path], args.hf_repo)

    # Summary
    print(f"\n=== Done in {fmt_time(time.time() - t_total)} ===")
    if sft is not None:
        print(f"  SFT-train:  {fmt_n(sft['tokens'])} tokens, {sft['docs']:,} docs → {sft_path}")
        print(f"  eval:       {fmt_n(ev['tokens'])} tokens, {ev['docs']:,} docs → {eval_path}")
    if args.pack and sft_packed_path.exists():
        print(f"  packed:     {sft_packed_path}")
        print(f"              {eval_packed_path}")
    if not args.pack:
        print(f"\nNext step — pack into fixed-length sequences:")
        print(f"  python data/retrieval_scripts/pack_dataset.py \\")
        print(f"    --tokenizer {args.tokenizer} \\")
        print(f"    --input {sft_path} --output {sft_packed_path} \\")
        print(f"    --eval-input {eval_path} --eval-output {eval_packed_path}")
        print(f"\n  (or re-run this script with --pack to do both + upload to HF)")
    elif args.hf_repo:
        print(f"  uploaded to: {args.hf_repo}:data/Pretrain/")
        print(f"\nFuture pods can fetch via 2_data.sh (after the same patch lands)")
        print(f"or manually with:")
        print(f"  hf download {args.hf_repo} \\")
        print(f"    data/Pretrain/{sft_packed_path.name} \\")
        print(f"    data/Pretrain/{eval_packed_path.name} \\")
        print(f"    --repo-type dataset --local-dir .")


if __name__ == "__main__":
    main()
