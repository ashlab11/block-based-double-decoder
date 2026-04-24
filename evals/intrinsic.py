"""
Intrinsic evaluation metrics: Held-out Perplexity, Bits-per-Byte,
Next-Token Accuracy, and Positional Accuracy.

All evals use batched GPU scoring for maximum throughput.
"""

import os
import math
import torch
import numpy as np
from datasets import load_dataset
from evals.utils import get_sequence_log_probs_batch


def _load_eval_dataset(eval_file, tokenizer, max_examples, fallback_name="Wikitext-103"):
    """Load packed eval data from local file, or fall back to Wikitext-103.
    Returns a list (not generator) so it can be batched.
    """
    id_lists = []
    if os.path.exists(eval_file):
        ds = load_dataset("json", data_files=eval_file, split="train", streaming=True)
        for i, example in enumerate(ds):
            if i >= max_examples:
                break
            id_lists.append(example["input_ids"])
    else:
        print(f"  [INFO] {eval_file} not found — falling back to {fallback_name}")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        for example in ds:
            if len(id_lists) >= max_examples:
                break
            text = example["text"]
            if len(text.strip()) < 50:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 10:
                continue
            id_lists.append(ids)

    return id_lists


# ── Held-Out Perplexity ───────────────────────────────────────────────────

def eval_held_out_perplexity(model, tokenizer, device, is_enc_dec,
                             eval_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                             max_examples=500, batch_size=64):
    """Compute perplexity on packed held-out sequences (batched)."""
    all_ids = _load_eval_dataset(eval_file, tokenizer, max_examples)

    total_nll = 0.0
    total_tokens = 0

    results = get_sequence_log_probs_batch(
        model, tokenizer, all_ids, device, is_enc_dec,
        batch_size=batch_size, desc="Held-out PPL"
    )
    for log_probs, _ in results:
        total_nll += -log_probs.sum().item()
        total_tokens += log_probs.shape[0]

    avg_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return {"name": "Held-Out Perplexity", "perplexity": ppl, "avg_loss": avg_loss,
            "total_tokens": total_tokens}


# ── Bits per Byte ─────────────────────────────────────────────────────────

def eval_bpb(model, tokenizer, device, is_enc_dec, max_examples=200, batch_size=64):
    """Compute bits-per-byte on Wikitext-103 raw (batched)."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")

    texts = []
    id_lists = []
    for example in ds:
        if len(id_lists) >= max_examples:
            break
        text = example["text"]
        if len(text.strip()) < 20:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            continue
        texts.append(text)
        id_lists.append(ids)

    results = get_sequence_log_probs_batch(
        model, tokenizer, id_lists, device, is_enc_dec,
        batch_size=batch_size, desc="BPB (Wikitext-103)"
    )

    total_nll = 0.0
    total_bytes = 0
    for i, (log_probs, _) in enumerate(results):
        total_nll += -log_probs.sum().item()
        total_bytes += len(texts[i].encode("utf-8"))

    bpb = total_nll / (math.log(2) * max(total_bytes, 1))
    return {"name": "Bits-per-Byte", "bpb": bpb, "total_bytes": total_bytes,
            "num_passages": len(id_lists)}


# ── Next-Token Accuracy ──────────────────────────────────────────────────

def eval_token_accuracy(model, tokenizer, device, is_enc_dec,
                        eval_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                        max_examples=500, batch_size=64):
    """Top-1 next-token prediction accuracy on held-out data (batched)."""
    all_ids = _load_eval_dataset(eval_file, tokenizer, max_examples)

    correct = 0
    total = 0

    results = get_sequence_log_probs_batch(
        model, tokenizer, all_ids, device, is_enc_dec,
        batch_size=batch_size, desc="Token Accuracy"
    )
    for i, (_, predicted) in enumerate(results):
        targets = torch.tensor(all_ids[i][1:], device=device)
        correct += (predicted == targets).sum().item()
        total += targets.shape[0]

    acc = correct / max(total, 1)
    return {"name": "Next-Token Accuracy", "accuracy": acc, "correct": correct, "total": total}


# ── Positional Accuracy ──────────────────────────────────────────────────

def eval_positional_accuracy(model, tokenizer, device, is_enc_dec,
                             eval_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                             max_examples=500, seq_len=2048, batch_size=64):
    """Next-token accuracy binned by position in the sequence (batched)."""
    all_ids = _load_eval_dataset(eval_file, tokenizer, max_examples)

    num_bins = 64
    bin_size = max(seq_len // num_bins, 1)
    correct_bins = np.zeros(num_bins, dtype=np.int64)
    total_bins = np.zeros(num_bins, dtype=np.int64)

    results = get_sequence_log_probs_batch(
        model, tokenizer, all_ids, device, is_enc_dec,
        batch_size=batch_size, desc="Positional Accuracy"
    )
    for i, (_, predicted) in enumerate(results):
        targets = torch.tensor(all_ids[i][1:], device=device)
        matches = (predicted == targets).cpu().numpy()

        for pos in range(len(matches)):
            b = min(pos // bin_size, num_bins - 1)
            correct_bins[b] += int(matches[pos])
            total_bins[b] += 1

    bin_acc = np.where(total_bins > 0, correct_bins / total_bins, 0.0)

    bins = []
    for b in range(num_bins):
        start = b * bin_size
        end = min((b + 1) * bin_size, seq_len) - 1
        bins.append({"range": f"{start}-{end}", "accuracy": float(bin_acc[b]),
                     "total": int(total_bins[b])})

    overall_acc = int(correct_bins.sum()) / max(int(total_bins.sum()), 1)
    return {
        "name": "Positional Accuracy",
        "overall_accuracy": overall_acc,
        "bins": bins,
    }
