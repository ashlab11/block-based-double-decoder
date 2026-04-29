"""Pack tokenized documents into fixed-length sequences for training.

Usage:
    # Default (original 50M run):
    python data/retrieval_scripts/pack_dataset.py

    # 1B run with 32K tokenizer:
    python data/retrieval_scripts/pack_dataset.py \
        --tokenizer tokenizer/tokenizer_32k.json \
        --input data/Pretrain/slimpajama_20b.jsonl \
        --output data/Pretrain/slimpajama_20b_packed.jsonl \
        --eval-input data/Pretrain/slimpajama_20b_eval.jsonl \
        --eval-output data/Pretrain/slimpajama_20b_eval_packed.jsonl
"""

from datasets import load_dataset
import argparse
import os
import time
from transformers import PreTrainedTokenizerFast


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def _serialize_ids(ids):
    """Fast JSON serialization for a list of ints (avoids json.dumps overhead)."""
    return '{"input_ids":[' + ','.join(map(str, ids)) + ']}\n'


def pack(ds, tokenizer, ctx_len, writer, buffer_size=10000):
    """Pack tokenized documents into fixed-length sequences."""
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")

    current = [bos_id]
    buf = []
    num_sequences = 0
    start_time = time.time()
    last_report_time = start_time

    for doc in ds:
        ids = doc["input_ids"]
        while ids:
            if len(ids) + len(current) > ctx_len:
                take = ctx_len - len(current)
                current += ids[:take]
                buf.append(_serialize_ids(current))
                num_sequences += 1
                if len(buf) >= buffer_size:
                    writer.writelines(buf)
                    buf.clear()
                ids = ids[take:]
                current = [bos_id]
            else:
                current += ids
                space = ctx_len - len(current)
                if space > 2:
                    current.extend([eos_id, bos_id])
                if space == 1:
                    current.append(eos_id)
                    buf.append(_serialize_ids(current))
                    num_sequences += 1
                    if len(buf) >= buffer_size:
                        writer.writelines(buf)
                        buf.clear()
                    current = [bos_id]
                if space == 0:
                    buf.append(_serialize_ids(current))
                    num_sequences += 1
                    if len(buf) >= buffer_size:
                        writer.writelines(buf)
                        buf.clear()
                    current = [bos_id]
                ids = []

            # Progress report every 15 seconds
            now = time.time()
            if now - last_report_time >= 15:
                elapsed = now - start_time
                seqs_per_sec = num_sequences / max(elapsed, 1)
                tokens_packed = num_sequences * ctx_len
                print(f"  {num_sequences:>10,} seqs | {tokens_packed:,} tokens | "
                      f"{seqs_per_sec:,.0f} seq/s | elapsed {_fmt_time(elapsed)}")
                last_report_time = now

    writer.writelines(buf)
    elapsed = time.time() - start_time
    seqs_per_sec = num_sequences / max(elapsed, 1)
    print(f"  Done: {num_sequences:,} sequences in {_fmt_time(elapsed)} ({seqs_per_sec:,.0f} seq/s)")
    return num_sequences


def build_packed_dataset(in_path, out_path, tokenizer, ctx_len):
    print(f"Packing {in_path} -> {out_path} (seq_len={ctx_len})")

    num_cpus = os.cpu_count() or 1
    num_proc = max(1, min(num_cpus, 16))  # cap at 16 to avoid memory issues

    # Load into memory for parallel tokenization (much faster than streaming)
    print(f"  Loading and tokenizing with {num_proc} workers...")
    tok_start = time.time()
    ds = load_dataset("json", data_files=in_path, split="train")
    ds = ds.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        batch_size=5000,
        num_proc=num_proc,
        remove_columns=['text'],
    )
    print(f"  Tokenized {len(ds):,} documents in {_fmt_time(time.time() - tok_start)}")

    with open(out_path, "w") as f:
        return pack(ds, tokenizer, ctx_len, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--input", type=str, default="data/Pretrain/slimpajama.jsonl")
    parser.add_argument("--output", type=str, default="data/Pretrain/slimpajama_packed.jsonl")
    parser.add_argument("--eval-input", type=str, default="data/Pretrain/slimpajama_eval.jsonl")
    parser.add_argument("--eval-output", type=str, default="data/Pretrain/slimpajama_eval_packed.jsonl")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)

    total_start = time.time()
    # Pack eval first (smaller)
    build_packed_dataset(args.eval_input, args.eval_output, tokenizer, args.seq_len)
    # Pack train
    build_packed_dataset(args.input, args.output, tokenizer, args.seq_len)
    print(f"\nAll packing complete in {_fmt_time(time.time() - total_start)}")
