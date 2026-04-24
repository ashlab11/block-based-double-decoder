"""
Intrinsic evaluation metrics: Held-out Perplexity, Bits-per-Byte,
Next-Token Accuracy, and Positional Accuracy.
"""

import math
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from evals.utils import get_sequence_log_probs


# ── Held-Out Perplexity ───────────────────────────────────────────────────

def eval_held_out_perplexity(model, tokenizer, device, is_enc_dec,
                             eval_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                             max_examples=500):
    """Compute perplexity on packed held-out sequences.

    Returns: {"perplexity": float, "avg_loss": float, "total_tokens": int}
    """
    ds = load_dataset("json", data_files=eval_file, split="train", streaming=True)

    total_nll = 0.0
    total_tokens = 0

    for i, example in enumerate(tqdm(ds, desc="Held-out PPL", total=max_examples)):
        if i >= max_examples:
            break
        ids = example["input_ids"]
        log_probs, _ = get_sequence_log_probs(model, tokenizer, ids, device, is_enc_dec)
        total_nll += -log_probs.sum().item()
        total_tokens += log_probs.shape[0]

    avg_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return {"name": "Held-Out Perplexity", "perplexity": ppl, "avg_loss": avg_loss,
            "total_tokens": total_tokens}


# ── Bits per Byte ─────────────────────────────────────────────────────────

def eval_bpb(model, tokenizer, device, is_enc_dec, max_examples=200):
    """Compute bits-per-byte on Wikitext-103 raw.

    BPB = total_NLL_nats / (ln(2) * total_utf8_bytes)
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")

    total_nll = 0.0
    total_bytes = 0

    count = 0
    for example in tqdm(ds, desc="BPB (Wikitext-103)"):
        text = example["text"]
        if len(text.strip()) < 20:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            continue

        log_probs, _ = get_sequence_log_probs(model, tokenizer, ids, device, is_enc_dec)
        total_nll += -log_probs.sum().item()
        total_bytes += len(text.encode("utf-8"))

        count += 1
        if count >= max_examples:
            break

    bpb = total_nll / (math.log(2) * max(total_bytes, 1))
    return {"name": "Bits-per-Byte", "bpb": bpb, "total_bytes": total_bytes,
            "num_passages": count}


# ── Next-Token Accuracy ──────────────────────────────────────────────────

def eval_token_accuracy(model, tokenizer, device, is_enc_dec,
                        eval_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                        max_examples=500):
    """Top-1 next-token prediction accuracy on held-out data."""
    ds = load_dataset("json", data_files=eval_file, split="train", streaming=True)

    correct = 0
    total = 0

    for i, example in enumerate(tqdm(ds, desc="Token Accuracy", total=max_examples)):
        if i >= max_examples:
            break
        ids = example["input_ids"]
        _, predicted = get_sequence_log_probs(model, tokenizer, ids, device, is_enc_dec)

        targets = torch.tensor(ids[1:], device=device)
        correct += (predicted == targets).sum().item()
        total += targets.shape[0]

    acc = correct / max(total, 1)
    return {"name": "Next-Token Accuracy", "accuracy": acc, "correct": correct, "total": total}


# ── Positional Accuracy ──────────────────────────────────────────────────

def eval_positional_accuracy(model, tokenizer, device, is_enc_dec,
                             eval_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                             max_examples=500, seq_len=2048):
    """Next-token accuracy binned by position in the sequence.

    Returns accuracy for each position bucket, showing how prediction
    improves with more context.
    """
    ds = load_dataset("json", data_files=eval_file, split="train", streaming=True)

    # Bin into 64 buckets across the sequence length
    num_bins = 64
    bin_size = max(seq_len // num_bins, 1)
    correct_bins = np.zeros(num_bins, dtype=np.int64)
    total_bins = np.zeros(num_bins, dtype=np.int64)

    for i, example in enumerate(tqdm(ds, desc="Positional Accuracy", total=max_examples)):
        if i >= max_examples:
            break
        ids = example["input_ids"]
        _, predicted = get_sequence_log_probs(model, tokenizer, ids, device, is_enc_dec)

        targets = torch.tensor(ids[1:], device=device)
        matches = (predicted == targets).cpu().numpy()

        for pos in range(len(matches)):
            b = min(pos // bin_size, num_bins - 1)
            correct_bins[b] += int(matches[pos])
            total_bins[b] += 1

    bin_acc = np.where(total_bins > 0, correct_bins / total_bins, 0.0)

    # Create position labels for each bin
    bins = []
    for b in range(num_bins):
        start = b * bin_size
        end = min((b + 1) * bin_size, seq_len) - 1
        acc = float(bin_acc[b])
        bins.append({"range": f"{start}-{end}", "accuracy": acc, "total": int(total_bins[b])})

    overall_acc = int(correct_bins.sum()) / max(int(total_bins.sum()), 1)
    return {
        "name": "Positional Accuracy",
        "overall_accuracy": overall_acc,
        "bins": bins,
    }
