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
import json
import argparse
from transformers import PreTrainedTokenizerFast


def pack(ds, tokenizer, ctx_len, writer, buffer_size=1000):
    """Pack tokenized documents into fixed-length sequences."""
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")

    current = [bos_id]
    buf = []
    num_sequences = 0

    for doc in ds:
        ids = doc["input_ids"]
        while ids:
            if len(ids) + len(current) > ctx_len:
                take = ctx_len - len(current)
                current += ids[:take]
                buf.append(json.dumps({"input_ids": current}) + "\n")
                num_sequences += 1
                if len(buf) >= buffer_size:
                    writer.writelines(buf)
                    buf.clear()
                if num_sequences % 100_000 == 0:
                    print(f"  Packed {num_sequences:,} sequences...")
                ids = ids[take:]
                current = [bos_id]
            else:
                current += ids
                space = ctx_len - len(current)
                if space > 2:
                    current.extend([eos_id, bos_id])
                if space == 1:
                    current.append(eos_id)
                    buf.append(json.dumps({"input_ids": current}) + "\n")
                    num_sequences += 1
                    if len(buf) >= buffer_size:
                        writer.writelines(buf)
                        buf.clear()
                    if num_sequences % 100_000 == 0:
                        print(f"  Packed {num_sequences:,} sequences...")
                    current = [bos_id]
                if space == 0:
                    buf.append(json.dumps({"input_ids": current}) + "\n")
                    num_sequences += 1
                    if len(buf) >= buffer_size:
                        writer.writelines(buf)
                        buf.clear()
                    if num_sequences % 100_000 == 0:
                        print(f"  Packed {num_sequences:,} sequences...")
                    current = [bos_id]
                ids = []

    writer.writelines(buf)
    print(f"  Done: {num_sequences:,} sequences packed")
    return num_sequences


def build_packed_dataset(in_path, out_path, tokenizer, ctx_len):
    print(f"Packing {in_path} -> {out_path} (seq_len={ctx_len})")
    ds = load_dataset("json", data_files=in_path, split="train", streaming=True)
    ds = ds.map(lambda x: tokenizer(x["text"]), batched=True, batch_size=1000, remove_columns=['text'])

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

    # Pack eval first (smaller)
    build_packed_dataset(args.eval_input, args.eval_output, tokenizer, args.seq_len)
    # Pack train
    build_packed_dataset(args.input, args.output, tokenizer, args.seq_len)
