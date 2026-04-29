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

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _tokenize_chunk(tokenizer_file, texts):
    """Tokenize a chunk of texts in a worker process."""
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    return tok(texts)["input_ids"]


def pack(token_iter, bos_id, eos_id, ctx_len, writer, buffer_size=10000):
    """Pack tokenized documents into fixed-length sequences."""
    current = [bos_id]
    buf = []
    num_sequences = 0
    start_time = time.time()
    last_report_time = start_time

    for ids in token_iter:
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


def _parallel_tokenize(in_path, tokenizer_file, num_workers, chunk_size=10000):
    """Read raw JSONL and tokenize with parallel worker processes.

    Submits chunks to a process pool so multiple CPU cores tokenize
    simultaneously. Results are yielded in submission order.
    """
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    batch_texts = []

    with open(in_path) as f:
        for line in f:
            batch_texts.append(json.loads(line)["text"])
            if len(batch_texts) >= chunk_size:
                futures.append(executor.submit(_tokenize_chunk, tokenizer_file, batch_texts))
                batch_texts = []

                # Yield completed results to keep memory bounded
                while len(futures) > num_workers * 2:
                    fut = futures.pop(0)
                    for ids in fut.result():
                        yield ids

    if batch_texts:
        futures.append(executor.submit(_tokenize_chunk, tokenizer_file, batch_texts))

    # Drain remaining futures in order
    for fut in futures:
        for ids in fut.result():
            yield ids

    executor.shutdown(wait=False)


def build_packed_dataset(in_path, out_path, tokenizer, tokenizer_file, ctx_len):
    print(f"Packing {in_path} -> {out_path} (seq_len={ctx_len})")

    num_workers = max(1, min(os.cpu_count() or 1, 16))
    print(f"  Parallel tokenization with {num_workers} workers...")

    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")

    token_iter = _parallel_tokenize(in_path, tokenizer_file, num_workers)

    with open(out_path, "w") as f:
        return pack(token_iter, bos_id, eos_id, ctx_len, f)


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
    build_packed_dataset(args.eval_input, args.eval_output, tokenizer, args.tokenizer, args.seq_len)
    build_packed_dataset(args.input, args.output, tokenizer, args.tokenizer, args.seq_len)
    print(f"\nAll packing complete in {_fmt_time(time.time() - total_start)}")
