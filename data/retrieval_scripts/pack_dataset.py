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
import time
from tokenizers import Tokenizer


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


def _extract_text(line):
    """Extract text value from JSONL without full JSON parsing.
    Assumes format: {"text": "..."}  (text is the only or first field)."""
    # Find the value after "text":
    idx = line.index('"text"')
    # Skip past "text": and optional whitespace to the opening quote
    idx = line.index('"', idx + 6)  # find the colon-side quote end
    # Wait — safer to just find ": " then the next quote
    colon = line.index(':', idx)
    start = line.index('"', colon) + 1
    # Find the closing quote (handle escaped quotes)
    end = start
    while True:
        end = line.index('"', end)
        # Count preceding backslashes
        num_bs = 0
        check = end - 1
        while check >= start and line[check] == '\\':
            num_bs += 1
            check -= 1
        if num_bs % 2 == 0:
            break
        end += 1
    raw = line[start:end]
    # Unescape common JSON escapes
    if '\\' in raw:
        raw = raw.replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n').replace('\\t', '\t')
    return raw


def _read_texts_batched(in_path, batch_size=10000):
    """Read JSONL file and yield batches of text strings."""
    import json
    batch = []
    with open(in_path) as f:
        for line in f:
            batch.append(json.loads(line)["text"])
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_packed_dataset(in_path, out_path, tokenizer, ctx_len):
    """Tokenize and pack in one pass using the Rust tokenizer directly."""
    print(f"Packing {in_path} -> {out_path} (seq_len={ctx_len})")

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    current = [bos_id]
    buf = []
    num_sequences = 0
    start_time = time.time()
    last_report_time = start_time
    buffer_size = 10000

    with open(out_path, "w") as writer:
        for text_batch in _read_texts_batched(in_path, batch_size=10000):
            # encode_batch uses Rust-level parallelism across all CPU cores
            encoded = tokenizer.encode_batch(text_batch)

            for enc in encoded:
                ids = enc.ids
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer_32k.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--input", type=str, default="data/Pretrain/slimpajama.jsonl")
    parser.add_argument("--output", type=str, default="data/Pretrain/slimpajama_packed.jsonl")
    parser.add_argument("--eval-input", type=str, default="data/Pretrain/slimpajama_eval.jsonl")
    parser.add_argument("--eval-output", type=str, default="data/Pretrain/slimpajama_eval_packed.jsonl")
    args = parser.parse_args()

    # Use the Rust tokenizer directly (not the HF wrapper) for encode_batch
    tokenizer = Tokenizer.from_file(args.tokenizer)

    total_start = time.time()
    build_packed_dataset(args.eval_input, args.eval_output, tokenizer, args.seq_len)
    build_packed_dataset(args.input, args.output, tokenizer, args.seq_len)
    print(f"\nAll packing complete in {_fmt_time(time.time() - total_start)}")
