#!/usr/bin/env python
"""Stream from gmongaras/SlimPajama-627B_Reupload (parquet mirror of cerebras/SlimPajama-627B),
tokenize inline with the local 32K BPE, pack into fixed-length sequences, and upload the
packed JSONL files to the bpbradle/slimpajama-6b-packed HF dataset.

Why this exists
---------------
slimpajama.py uses len(text)//4 as a token-count heuristic during streaming, which over-counts
on heterogeneous corpora and stops the loop before reaching the real BPE token target. By
tokenizing inline with the same Tokenizer used for training, the target count becomes exact.

Usage
-----
    # smoke test — 50M tokens (~5 min)
    python data/retrieval_scripts/build_and_upload_slimpajama_6b.py \
        --tokens 50000000 --no-upload

    # full 6.2B build (~3-6 h on CPU; network-bound)
    python data/retrieval_scripts/build_and_upload_slimpajama_6b.py \
        --tokens 6200000000

    # only upload existing files (skip rebuild)
    python data/retrieval_scripts/build_and_upload_slimpajama_6b.py --upload-only

The default source is gmongaras/SlimPajama-627B_Reupload (a parquet reupload of the original
cerebras/SlimPajama-627B which was taken down). Override with --source if needed; any repo
with the {text, meta:{redpajama_set_name:...}} schema will work.
"""
import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np


def _fmt(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _serialize_ids(ids):
    return '{"input_ids":[' + ','.join(map(str, ids)) + ']}\n'


def iter_with_recovery(source, base_seed, max_resets=20, log=print):
    """Yield records from a streaming HF dataset, recreating the iterator on transient
    httpx/network errors. The huggingface_hub HfFileSystem can close its httpx client
    after certain network blips and never recover; the only fix is to rebuild the
    whole datasets pipeline. Each reset uses a different shuffle seed so we don't
    re-iterate the same shards in the same order.
    """
    from datasets import load_dataset
    resets = 0
    while True:
        seed = base_seed + resets
        try:
            ds = load_dataset(source, split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            for ex in ds:
                yield ex
            return  # clean end-of-stream
        except (RuntimeError, ConnectionError, OSError) as e:
            msg = str(e)[:200]
            if resets >= max_resets:
                log(f"  ! recovery exhausted after {resets} resets: {type(e).__name__}: {msg}")
                raise
            resets += 1
            log(f"  ! stream broke ({type(e).__name__}: {msg}) — reopening "
                f"dataset (reset {resets}/{max_resets}, new seed={base_seed + resets})")
            time.sleep(min(30, 2 ** resets))
            continue
        except Exception as e:
            # httpx and other libs raise their own exception types — try to catch by name
            name = type(e).__name__
            if any(s in name for s in ("HTTP", "Httpx", "Read", "Connect", "Timeout", "Closed")):
                if resets >= max_resets:
                    log(f"  ! recovery exhausted after {resets} resets: {name}: {str(e)[:200]}")
                    raise
                resets += 1
                log(f"  ! stream broke ({name}: {str(e)[:200]}) — reopening "
                    f"dataset (reset {resets}/{max_resets}, new seed={base_seed + resets})")
                time.sleep(min(30, 2 ** resets))
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=6_200_000_000,
                    help="Target real BPE token count for the train file (default 6.2B)")
    ap.add_argument("--eval-frac", type=float, default=0.001,
                    help="Fraction of docs held out for eval (default 0.1%%)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--source", type=str, default="gmongaras/SlimPajama-627B_Reupload")
    ap.add_argument("--tokenizer", type=str,
                    default="tokenizer/tokenizer_32k.json")
    ap.add_argument("--out-dir", type=str, default="data/Pretrain")
    ap.add_argument("--train-name", type=str, default="slimpajama_6b_packed.jsonl")
    ap.add_argument("--eval-name", type=str, default="slimpajama_6b_eval_packed.jsonl")
    ap.add_argument("--target-repo", type=str, default="bpbradle/slimpajama-6b-packed")
    ap.add_argument("--no-upload", action="store_true",
                    help="Skip the HF upload step (use for smoke tests)")
    ap.add_argument("--upload-only", action="store_true",
                    help="Skip rebuild; just upload existing files")
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="Docs per encode_batch call (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / args.train_name
    eval_path = out_dir / args.eval_name

    # Lazy imports so --upload-only doesn't pull tokenizers/datasets if not needed
    from huggingface_hub import HfApi

    if not args.upload_only:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)
        bos_id = tokenizer.token_to_id("<s>")
        eos_id = tokenizer.token_to_id("</s>")
        if bos_id is None or eos_id is None:
            sys.exit(f"Tokenizer {args.tokenizer} missing <s> or </s>")
        print(f"Tokenizer: vocab={tokenizer.get_vocab_size()}  <s>={bos_id}  </s>={eos_id}")

        # Domain-proportional sampling targets — same proportions as slimpajama.py
        target = args.tokens
        domain_targets = {
            "RedPajamaCommonCrawl": int(0.5 * target),
            "RedPajamaC4":          int(0.2 * target),
            "RedPajamaGithub":      int(0.1 * target),
            "RedPajamaArXiv":       int(0.1 * target),
            "RedPajamaBook":        int(0.1 * target),
        }
        domain_so_far = {k: 0 for k in domain_targets}
        eval_target = int(target * args.eval_frac * 1.5)  # a touch of headroom

        print(f"Source: {args.source}")
        print(f"Target: {target:,} train tokens (eval ~{eval_target:,} tokens)")
        for k, v in domain_targets.items():
            print(f"  {k}: {v:,}")

        # Resilient streaming iterator — see iter_with_recovery for rationale.
        ds_iter = iter_with_recovery(args.source, base_seed=args.seed)

        train_seqs = 0
        eval_seqs = 0
        train_tokens = 0
        eval_tokens = 0
        docs_seen = 0
        docs_used_train = 0
        docs_used_eval = 0
        train_current = [bos_id]
        eval_current = [bos_id]
        train_buf = []
        eval_buf = []
        write_buf_size = 5000
        start = time.time()
        last_report = start
        all_full = False

        def flush(buf, fh):
            if buf:
                fh.writelines(buf)
                buf.clear()

        def pack_into(ids, current, seqs_count, tokens_count, buf, ctx_len):
            # Append `ids` to `current`, emitting fixed-length sequences as they fill up.
            # Returns (current, seqs_count, tokens_count).
            i = 0
            n = len(ids)
            while i < n:
                space_for_payload = ctx_len - len(current)
                take = min(space_for_payload, n - i)
                current.extend(ids[i:i + take])
                i += take
                if len(current) == ctx_len:
                    buf.append(_serialize_ids(current))
                    seqs_count += 1
                    tokens_count += ctx_len
                    current = [bos_id]
            # Insert sentinel </s><s> if there's room before next doc
            space = ctx_len - len(current)
            if space >= 2:
                current.extend([eos_id, bos_id])
            elif space == 1:
                current.append(eos_id)
                buf.append(_serialize_ids(current))
                seqs_count += 1
                tokens_count += ctx_len
                current = [bos_id]
            return current, seqs_count, tokens_count

        f_train = train_path.open("w", encoding="utf-8")
        f_eval = eval_path.open("w", encoding="utf-8")
        try:
            batch_texts = []
            batch_set_names = []
            for ex in ds_iter:
                docs_seen += 1
                meta = ex.get("meta", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                set_name = meta.get("redpajama_set_name") if isinstance(meta, dict) else None
                if set_name not in domain_targets:
                    continue
                if domain_so_far[set_name] >= domain_targets[set_name]:
                    # Check if all domains are full → exit
                    if all(domain_so_far[k] >= domain_targets[k] for k in domain_targets):
                        all_full = True
                        break
                    continue

                text = ex.get("text", "")
                if not text:
                    continue
                text = unicodedata.normalize("NFKC", text)
                batch_texts.append(text)
                batch_set_names.append(set_name)

                if len(batch_texts) >= args.batch_size:
                    encoded = tokenizer.encode_batch(batch_texts)
                    for enc, sname in zip(encoded, batch_set_names):
                        ids = enc.ids
                        n = len(ids)
                        # Hold out a small fraction of docs for eval
                        if eval_tokens < eval_target and np.random.random() < args.eval_frac:
                            eval_current, eval_seqs, eval_tokens = pack_into(
                                ids, eval_current, eval_seqs, eval_tokens,
                                eval_buf, args.seq_len)
                            docs_used_eval += 1
                            if len(eval_buf) >= write_buf_size:
                                flush(eval_buf, f_eval)
                            continue
                        # Accept toward this domain (counted before packing — exact token cost)
                        if domain_so_far[sname] + n > domain_targets[sname]:
                            # Truncate this doc's contribution so we don't blow past the domain cap
                            allowed = max(0, domain_targets[sname] - domain_so_far[sname])
                            if allowed == 0:
                                continue
                            ids = ids[:allowed]
                            n = allowed
                        train_current, train_seqs, train_tokens = pack_into(
                            ids, train_current, train_seqs, train_tokens,
                            train_buf, args.seq_len)
                        domain_so_far[sname] += n
                        docs_used_train += 1
                        if len(train_buf) >= write_buf_size:
                            flush(train_buf, f_train)
                    batch_texts.clear()
                    batch_set_names.clear()

                    now = time.time()
                    if now - last_report >= 30:
                        elapsed = now - start
                        pct = 100 * train_tokens / target
                        rate = train_tokens / max(elapsed, 1)
                        eta = (target - train_tokens) / max(rate, 1)
                        print(f"  [{pct:5.1f}%] train {train_tokens:,} tok ({train_seqs:,} seq) | "
                              f"eval {eval_tokens:,} tok | docs seen={docs_seen:,} "
                              f"used={docs_used_train:,} | rate {rate:,.0f} tok/s | "
                              f"elapsed {_fmt(elapsed)} | eta {_fmt(eta)}")
                        print("    domains: " + ", ".join(
                            f"{k.replace('RedPajama','')}: "
                            f"{100 * domain_so_far[k] / max(1, domain_targets[k]):.0f}%"
                            for k in domain_targets))
                        last_report = now

                    if all(domain_so_far[k] >= domain_targets[k] for k in domain_targets):
                        all_full = True
                        break

            # Drain residual batch
            if batch_texts and not all_full:
                encoded = tokenizer.encode_batch(batch_texts)
                for enc, sname in zip(encoded, batch_set_names):
                    ids = enc.ids
                    n = len(ids)
                    if eval_tokens < eval_target and np.random.random() < args.eval_frac:
                        eval_current, eval_seqs, eval_tokens = pack_into(
                            ids, eval_current, eval_seqs, eval_tokens,
                            eval_buf, args.seq_len)
                        docs_used_eval += 1
                        continue
                    if domain_so_far[sname] >= domain_targets[sname]:
                        continue
                    allowed = domain_targets[sname] - domain_so_far[sname]
                    if n > allowed:
                        ids = ids[:allowed]
                        n = allowed
                    train_current, train_seqs, train_tokens = pack_into(
                        ids, train_current, train_seqs, train_tokens,
                        train_buf, args.seq_len)
                    domain_so_far[sname] += n
                    docs_used_train += 1

        finally:
            # Flush in finally so partial progress survives a crash
            flush(train_buf, f_train)
            flush(eval_buf, f_eval)
            f_train.close()
            f_eval.close()

        elapsed = time.time() - start
        print()
        print(f"Done in {_fmt(elapsed)}: {train_tokens:,} train tokens ({train_seqs:,} seqs), "
              f"{eval_tokens:,} eval tokens ({eval_seqs:,} seqs)")
        print(f"  docs: seen={docs_seen:,}  train_used={docs_used_train:,}  eval_used={docs_used_eval:,}")
        for k, v in domain_targets.items():
            pct = 100 * domain_so_far[k] / max(1, v)
            print(f"  {k}: {domain_so_far[k]:,}/{v:,}  ({pct:.1f}%)")

    # Upload step
    if args.no_upload:
        print()
        print("--no-upload set; skipping HF upload.")
        return

    if not train_path.exists() or train_path.stat().st_size == 0:
        sys.exit(f"Cannot upload — {train_path} missing or empty")
    if not eval_path.exists() or eval_path.stat().st_size == 0:
        sys.exit(f"Cannot upload — {eval_path} missing or empty")

    print()
    print(f"Uploading to {args.target_repo} ...")
    api = HfApi()
    for src, dst in [
        (train_path, f"data/Pretrain/{args.train_name}"),
        (eval_path, f"data/Pretrain/{args.eval_name}"),
        (Path(args.tokenizer), f"tokenizer/{Path(args.tokenizer).name}"),
    ]:
        size_gb = src.stat().st_size / 1e9
        print(f"  uploading {src} ({size_gb:.2f} GB) -> {dst}")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=args.target_repo,
            repo_type="dataset",
            commit_message=f"Repack from gmongaras/SlimPajama-627B_Reupload "
                           f"(target {args.tokens:,} real BPE tokens)",
        )
        print(f"    done in {_fmt(time.time() - t0)}")

    print()
    print("Upload complete. Verify with:")
    print(f"  hf download {args.target_repo} --repo-type dataset --local-dir _verify")


if __name__ == "__main__":
    main()
    # Hard exit — datasets streaming connections sometimes hang in cleanup,
    # leaving the process zombied for 20+ minutes. Same workaround as
    # data/retrieval_scripts/slimpajama.py.
    os._exit(0)
