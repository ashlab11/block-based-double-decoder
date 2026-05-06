#!/usr/bin/env python3
"""
Pre-shutdown audit: verify every local training artifact persists on HF Hub.

Walks ``--search-dir`` (default ``checkpoints/``) for ``.pt`` weights and
``.json`` results files, queries HF Hub for the same file set, and reports:

  ✓ matched           — present on both, sizes equal (safe)
  ✗ MISSING from HF   — local-only; would be lost when the pod terminates
  ⚠ size mismatch     — HF copy doesn't match local (corrupt or partial upload)

With ``--sync``, uploads any missing/mismatched files via
``HfApi.upload_folder()`` (one atomic commit on HF, content-hash dedup so
re-running is free) and re-runs the audit to confirm.

Exits 0 only when every local artifact has a matching HF copy. Exits 1
if anything is at risk and ``--sync`` was not passed (or if --sync ran
and didn't fully resolve).

Usage:
    # Audit only (read-only, no uploads)
    python scripts/verify_persistence.py --hf-repo "$HF_REPO"

    # Audit + sync — recommended right before pod shutdown
    python scripts/verify_persistence.py --hf-repo "$HF_REPO" --sync

    # Audit a specific subdirectory only
    python scripts/verify_persistence.py --hf-repo "$HF_REPO" \\
        --search-dir checkpoints/ben_sweep_dd_50M_6B
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def fmt_mb(b):
    return f"{b/1e6:.1f} MB" if b is not None else "?"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--search-dir", type=str, default="checkpoints/",
                   help="Local directory to audit recursively (default: checkpoints/)")
    p.add_argument("--hf-repo", type=str, required=True,
                   help="HF Hub repo id (e.g. 'bpbradle/scaling-runs')")
    p.add_argument("--patterns", type=str, default="*.pt,*.json",
                   help="Comma-separated glob patterns for files to audit. "
                        "Default '*.pt,*.json' covers weights + result JSONs.")
    p.add_argument("--hf-prefix", type=str, default=None,
                   help="Path prefix inside the HF repo. Defaults to the "
                        "search-dir's basename + '/' (e.g. 'checkpoints/').")
    p.add_argument("--sync", action="store_true",
                   help="Upload any missing/mismatched files via "
                        "upload_folder. Idempotent — HF dedupes by content hash.")
    p.add_argument("--allow-hf-only", action="store_true",
                   help="Don't print HF-only files (cuts noise on big repos).")

    args = p.parse_args()

    search = Path(args.search_dir).resolve()
    if not search.exists():
        print(f"ERROR: --search-dir {search} does not exist")
        sys.exit(1)
    if not search.is_dir():
        print(f"ERROR: --search-dir {search} is not a directory")
        sys.exit(1)

    if args.hf_prefix is None:
        args.hf_prefix = search.name + "/"
    args.hf_prefix = args.hf_prefix.rstrip("/") + "/"

    api = HfApi()

    # ── Inventory local ────────────────────────────────────────────────────
    patterns = [pat.strip() for pat in args.patterns.split(",") if pat.strip()]
    local = {}  # path-in-repo -> (local_path, size_bytes)
    for pat in patterns:
        for fp in search.rglob(pat):
            if fp.is_file():
                rel = fp.relative_to(search)
                rel_in_repo = f"{args.hf_prefix}{rel}"
                local[rel_in_repo] = (fp, fp.stat().st_size)

    if not local:
        print(f"[local] no files matching {patterns} under {search}")
        print("Nothing to verify. (Did you point --search-dir at the right place?)")
        return

    total_local_bytes = sum(sz for _, sz in local.values())
    print(f"[local] {len(local)} files matching {patterns} under {search}")
    print(f"[local] total size: {fmt_mb(total_local_bytes)}")

    # ── Inventory HF ───────────────────────────────────────────────────────
    print(f"[hf] querying {args.hf_repo} (this can take 5-10s for large repos)...")
    try:
        info = api.repo_info(args.hf_repo, repo_type="model",
                             files_metadata=True)
    except Exception as e:
        print(f"ERROR: querying HF repo {args.hf_repo}: {e}")
        print("  - Did you run `huggingface-cli login` (or set HF_TOKEN)?")
        print("  - Is HF_REPO set to the right repo id?")
        sys.exit(1)

    hf = {s.rfilename: s.size for s in info.siblings if s.size is not None}
    print(f"[hf] repo has {len(hf)} total files")

    # ── Compare ────────────────────────────────────────────────────────────
    matched, missing, size_mismatch = [], [], []
    for rel, (lp, lsize) in sorted(local.items()):
        if rel in hf:
            if hf[rel] == lsize:
                matched.append(rel)
            else:
                size_mismatch.append((rel, lp, lsize, hf[rel]))
        else:
            missing.append((rel, lp, lsize))

    # HF-only (informational): files under our prefix that aren't local. Could
    # be from a previous session, an aborted run, or a different machine.
    local_keys = set(local.keys())
    hf_only = sorted(rel for rel in hf
                     if rel.startswith(args.hf_prefix) and rel not in local_keys)

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  AUDIT RESULTS")
    print(f"  local: {search}")
    print(f"  hf:    {args.hf_repo} (prefix: {args.hf_prefix!r})")
    print("=" * 72)
    print(f"  ✓ matched (safe — on both, sizes equal):  {len(matched):>5d}")
    print(f"  ✗ MISSING from HF (LOST on shutdown!):    {len(missing):>5d}")
    print(f"  ⚠ size mismatch (corrupt — re-upload):    {len(size_mismatch):>5d}")
    print(f"  ℹ HF-only (under prefix, no local copy):  {len(hf_only):>5d}")
    print()

    if missing:
        miss_bytes = sum(sz for _, _, sz in missing)
        print(f"FILES MISSING FROM HF ({len(missing)} files, {fmt_mb(miss_bytes)} total):")
        for rel, lp, sz in missing:
            print(f"  {fmt_mb(sz):>10s}  {rel}")
        print()

    if size_mismatch:
        print(f"SIZE MISMATCH ({len(size_mismatch)} files — HF copy is corrupt or partial):")
        for rel, lp, lsize, hsize in size_mismatch:
            print(f"  local={fmt_mb(lsize):>10s}  hf={fmt_mb(hsize):>10s}  {rel}")
        print()

    if hf_only and not args.allow_hf_only:
        print(f"HF-ONLY ({len(hf_only)} files on HF without local counterpart "
              f"— informational, not at risk):")
        for rel in hf_only[:10]:
            print(f"  {rel}")
        if len(hf_only) > 10:
            print(f"  ... and {len(hf_only) - 10} more  (re-run with --allow-hf-only to suppress)")
        print()

    # ── Sync ───────────────────────────────────────────────────────────────
    needs_upload = bool(missing or size_mismatch)
    if args.sync and needs_upload:
        n = len(missing) + len(size_mismatch)
        upload_bytes = (sum(sz for _, _, sz in missing)
                        + sum(lsize for _, _, lsize, _ in size_mismatch))
        print(f"[sync] uploading {n} files ({fmt_mb(upload_bytes)})...")
        # upload_folder() pushes the whole tree in one commit on HF and
        # dedupes by content hash, so the matched files are no-ops. Pattern
        # filtering keeps unrelated files (e.g. .log) out of the repo.
        try:
            api.upload_folder(
                folder_path=str(search),
                path_in_repo=args.hf_prefix.rstrip("/"),
                repo_id=args.hf_repo,
                repo_type="model",
                commit_message=f"sync {n} artifacts from {search.name}",
                allow_patterns=patterns,
            )
            print(f"[sync] upload_folder() returned cleanly.")
        except Exception as e:
            print(f"[sync] FAILED: {e}")
            sys.exit(1)

        # Re-query HF to confirm everything landed (don't trust the upload
        # response — verify by reading back the listing).
        print(f"[sync] re-querying HF to verify...")
        info2 = api.repo_info(args.hf_repo, repo_type="model",
                              files_metadata=True)
        hf2 = {s.rfilename: s.size for s in info2.siblings if s.size is not None}

        still_missing = [r for r, _, _ in missing if r not in hf2]
        still_mismatch = [(r, ls) for r, _, ls, _ in size_mismatch
                          if hf2.get(r) != ls]

        if not still_missing and not still_mismatch:
            print(f"[sync] ✓ verified — all {n} files now on HF.")
            print()
            print("✓ SAFE TO SHUT DOWN POD.")
            return
        else:
            print(f"[sync] ⚠ post-upload check found gaps:")
            for r in still_missing:
                print(f"  STILL MISSING: {r}")
            for r, ls in still_mismatch:
                print(f"  STILL MISMATCH: {r} (local={fmt_mb(ls)}, hf={fmt_mb(hf2.get(r))})")
            sys.exit(1)
    elif args.sync:
        print("[sync] nothing to upload — already in sync.")
        print()
        print("✓ SAFE TO SHUT DOWN POD.")
    elif needs_upload:
        print("⚠ NOT SAFE TO SHUT DOWN — local-only files would be lost.")
        print("  Re-run with --sync to upload them, or accept the loss.")
        sys.exit(1)
    else:
        print("✓ SAFE TO SHUT DOWN POD — every local artifact has a matching HF copy.")


if __name__ == "__main__":
    main()
