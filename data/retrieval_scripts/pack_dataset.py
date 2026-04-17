from datasets import load_dataset
import torch
import json
import os 
from transformers import PreTrainedTokenizerFast

SEQ_LEN = 2048

def pack(ds, tokenizer, ctx_len, writer, buffer_size=1000): 
    """Yield packed lists of token-ids <= ctx_len."""
    
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    
    current = [bos_id]
    buf = []
    for doc in ds:
        # First normalize using the same normalization as training
        ids = doc["input_ids"]
        while ids: 
            # Need to leave room for eos and bos
            if len(ids) + len(current) > ctx_len:
                take = ctx_len - len(current)
                current += ids[:take]
                buf.append(json.dumps({"input_ids": current}) + "\n")
                if len(buf) >= buffer_size:
                    writer.writelines(buf)
                    buf.clear()
                ids = ids[take:]
                current = [bos_id]
            else:
                current += ids
                space = ctx_len - len(current)
                if space > 2:
                    #Enough room for both
                    current.extend([eos_id, bos_id]) 
                if space == 1:
                    # Only room for eos
                    current.append(eos_id)
                    buf.append(json.dumps({"input_ids": current}) + "\n")
                    if len(buf) >= buffer_size:
                        writer.writelines(buf)
                        buf.clear()
                    current = [bos_id]
                if space == 0:
                    # No room for anything
                    buf.append(json.dumps({"input_ids": current}) + "\n")
                    if len(buf) >= buffer_size:
                        writer.writelines(buf)
                        buf.clear()
                    current = [bos_id]
                ids = []
    
    writer.writelines(buf)
    return writer

def build_packed_dataset(in_path, out_path, tokenizer, ctx_len):
    # ds is any hf streaming dataset with a "text" column
    ds = load_dataset("json", data_files=in_path, split="train", streaming=True)
    ds = ds.map(lambda x: tokenizer(x["text"]), batched=True, batch_size=1000, remove_columns=['text'])
    
    with open(out_path, "w") as f:
        pack(ds, tokenizer, ctx_len, f)

if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer/tokenizer.json"
    )
    build_packed_dataset("data/SFT/slimpajama_eval.jsonl", "data/Pretrain/slimpajama_eval_packed.jsonl", tokenizer, SEQ_LEN)
    build_packed_dataset("data/Pretrain/slimpajama.jsonl", "data/Pretrain/slimpajama_packed.jsonl", tokenizer, SEQ_LEN)