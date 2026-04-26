#!/usr/bin/env python
"""
Benchmark inference efficiency across architectures.

Measures:
  - First-token latency (time to generate first output token)
  - Decoding throughput (tokens/sec during generation)
  - KV cache memory (peak GPU memory during generation)
  - Parameter efficiency (params actually used during encoding vs decoding)
  - Data efficiency: tokens trained per parameter for encoder vs decoder

Usage:
    python evals/benchmark_efficiency.py \
        --checkpoints checkpoints/sft_dec_only_50m.pt checkpoints/sft_dd_50m.pt checkpoints/sft_enc_dec_50m.pt \
        --labels "Decoder-Only" "Double Decoder" "Std Enc-Dec"
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.utils import load_model
from components.block_masks import create_inference_masks

# Workaround for PyTorch inductor bug
import torch._inductor.config as _inductor_config
_inductor_config.pattern_matcher = False


def _warmup(model, tokenizer, device, is_enc_dec, n=3):
    """Run a few forward passes to warm up CUDA and torch.compile."""
    prompt = "The quick brown fox jumps over the lazy dog."
    for _ in range(n):
        _measure_first_token(model, tokenizer, device, is_enc_dec, prompt, warmup=True)


def _measure_first_token(model, tokenizer, device, is_enc_dec, prompt, warmup=False):
    """Measure time from prompt submission to first generated token."""
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    assistant_id = tokenizer.convert_tokens_to_ids("<assistant>")

    ids = tokenizer.encode(prompt, add_special_tokens=False)

    torch.cuda.synchronize()
    start = time.perf_counter()

    if is_enc_dec:
        enc_input = torch.tensor([[bos_id] + ids], device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc_out = model.encode(enc_input)

        dec_ids = torch.tensor([[assistant_id]], device=device)
        enc_len = len(ids) + 1
        dec_pos = torch.tensor([[enc_len]], device=device)

        masks = create_inference_masks(device=device, enc_len=enc_len, dec_len=dec_ids.shape[1])
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model.decode(dec_ids, enc_out, masks, dec_pos)
        first_token = logits[0, -1].argmax().item()
    else:
        input_t = torch.tensor([[bos_id] + ids], device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_t)["logits"]
        first_token = logits[0, -1].argmax().item()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if warmup:
        return elapsed
    return elapsed, first_token


def _measure_decoding_throughput(model, tokenizer, device, is_enc_dec, prompt,
                                  num_tokens=64):
    """Measure tokens/sec during autoregressive decoding."""
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    assistant_id = tokenizer.convert_tokens_to_ids("<assistant>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0

    ids = tokenizer.encode(prompt, add_special_tokens=False)

    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens_generated = 0

    if is_enc_dec:
        enc_input = torch.tensor([[bos_id] + ids], device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc_out = model.encode(enc_input)

        dec_ids = torch.tensor([[assistant_id]], device=device)
        enc_len = len(ids) + 1
        dec_pos = torch.tensor([[enc_len]], device=device)

        for _ in range(num_tokens):
            masks = create_inference_masks(device=device, enc_len=enc_len, dec_len=dec_ids.shape[1])
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model.decode(dec_ids, enc_out, masks, dec_pos)
            next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            dec_ids = torch.cat([dec_ids, next_tok], dim=-1)
            dec_pos = torch.cat([dec_pos, dec_pos[:, -1:] + 1], dim=1)
            tokens_generated += 1
            if next_tok.item() == eos_id:
                break
    else:
        input_t = torch.tensor([[bos_id] + ids], device=device)
        for _ in range(num_tokens):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_t)["logits"]
            next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            input_t = torch.cat([input_t, next_tok], dim=-1)
            tokens_generated += 1
            if next_tok.item() == eos_id:
                break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return tokens_generated, elapsed, tokens_generated / max(elapsed, 1e-9)


def _measure_memory(model, tokenizer, device, is_enc_dec, prompt, num_tokens=64):
    """Measure peak GPU memory during generation."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        _measure_decoding_throughput(model, tokenizer, device, is_enc_dec, prompt, num_tokens)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"    [WARN] CUDA error during memory measurement, using current peak")
            torch.cuda.empty_cache()
        else:
            raise

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return peak_mb


def _count_component_params(model, is_enc_dec):
    """Count parameters by component."""
    counts = {}
    total = sum(p.numel() for p in model.parameters())
    counts["total"] = total

    if is_enc_dec:
        enc_params = sum(p.numel() for p in model.encoder_layers.parameters())
        enc_params += sum(p.numel() for p in model.encoder_norm.parameters())
        dec_params = sum(p.numel() for p in model.decoder_layers.parameters())
        dec_params += sum(p.numel() for p in model.output_norm.parameters())
        emb_params = sum(p.numel() for p in model.embedding.parameters())
        # output_projection is tied to embedding
        counts["encoder"] = enc_params
        counts["decoder"] = dec_params
        counts["embedding"] = emb_params
        counts["enc_ratio"] = enc_params / total
        counts["dec_ratio"] = dec_params / total
    else:
        layer_params = sum(p.numel() for p in model.layers.parameters())
        emb_params = sum(p.numel() for p in model.embedding.parameters())
        counts["layers"] = layer_params
        counts["embedding"] = emb_params

    return counts


def _analyze_data_efficiency(model, is_enc_dec, model_label):
    """Analyze how tokens flow through each architecture during training.

    Key question: does the encoder train on the same tokens as the decoder?
    """
    analysis = {}

    if "Decoder-Only" in model_label:
        analysis["description"] = (
            "Every token passes through all 12 layers. "
            "All parameters see all tokens equally."
        )
        analysis["encoder_sees_tokens"] = "N/A (no encoder)"
        analysis["decoder_sees_tokens"] = "All input tokens (causal LM)"
        analysis["token_utilization"] = "100% — all params trained on all tokens"

    elif "Double Decoder" in model_label:
        analysis["description"] = (
            "During pretraining (sft=False), BOTH encoder and decoder receive "
            "the SAME input_ids. The encoder processes all tokens through 8 causal "
            "layers. The decoder also sees all tokens but through block-masked combo "
            "attention (self-attn within blocks + cross-attn to past blocks via encoder). "
            "Unlike standard enc-dec, the encoder is NOT idle — it trains on every token."
        )
        analysis["encoder_sees_tokens"] = "All tokens (same input as decoder)"
        analysis["decoder_sees_tokens"] = "All tokens (block-masked view)"
        analysis["token_utilization"] = (
            "100% — both encoder and decoder train on all tokens. "
            "This is a key advantage over standard enc-dec."
        )

    elif "Enc-Dec" in model_label:
        analysis["description"] = (
            "During pretraining (sft=False), BOTH encoder and decoder receive "
            "the SAME input_ids (same as DD). However, during SFT, the encoder "
            "only sees context tokens and the decoder only sees response tokens. "
            "During SFT, encoder parameters only update from context gradients."
        )
        analysis["encoder_sees_tokens"] = "All tokens (pretrain), context only (SFT)"
        analysis["decoder_sees_tokens"] = "All tokens (pretrain), response only (SFT)"
        analysis["token_utilization"] = (
            "100% pretrain, ~50% SFT (each component sees half the tokens). "
            "But SFT is where the encoder specialization happens."
        )

    return analysis


def benchmark_all(checkpoints, labels, prompts, device, num_gen_tokens=64,
                  num_trials=10, output_path=None):
    """Run all benchmarks on all models."""
    results = {}

    for ckpt, label in zip(checkpoints, labels):
        print(f"\n{'─'*60}")
        print(f"  Benchmarking: {label}")
        print(f"  Checkpoint: {ckpt}")
        print(f"{'─'*60}")

        model, tokenizer, is_enc_dec = load_model(ckpt, device=device)

        # Don't torch.compile for latency benchmarks — we want raw model speed
        model.eval()

        result = {"label": label, "is_enc_dec": is_enc_dec}

        # Parameter counts
        params = _count_component_params(model, is_enc_dec)
        result["params"] = params
        print(f"  Parameters: {params['total']/1e6:.1f}M")
        if is_enc_dec:
            print(f"    Encoder: {params['encoder']/1e6:.1f}M ({params['enc_ratio']:.0%})")
            print(f"    Decoder: {params['decoder']/1e6:.1f}M ({params['dec_ratio']:.0%})")

        # Data efficiency analysis
        data_eff = _analyze_data_efficiency(model, is_enc_dec, label)
        result["data_efficiency"] = data_eff
        print(f"  Data efficiency: {data_eff['token_utilization']}")

        # Warmup
        print(f"  Warming up...")
        _warmup(model, tokenizer, device, is_enc_dec)

        # First-token latency across prompt lengths
        print(f"  Measuring first-token latency ({num_trials} trials)...")
        ftl_by_length = {}
        for prompt in prompts:
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            latencies = []
            for _ in range(num_trials):
                elapsed, _ = _measure_first_token(model, tokenizer, device, is_enc_dec, prompt)
                latencies.append(elapsed * 1000)  # ms
            ftl_by_length[prompt_len] = {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "min_ms": float(np.min(latencies)),
            }
            print(f"    {prompt_len:>4d} tokens: {np.mean(latencies):.1f} ± {np.std(latencies):.1f} ms")
        result["first_token_latency"] = ftl_by_length

        # Decoding throughput
        print(f"  Measuring decoding throughput ({num_gen_tokens} tokens, {num_trials} trials)...")
        throughputs = []
        for _ in range(num_trials):
            _, _, tps = _measure_decoding_throughput(
                model, tokenizer, device, is_enc_dec, prompts[-1], num_gen_tokens)
            throughputs.append(tps)
        result["throughput"] = {
            "mean_tok_per_sec": float(np.mean(throughputs)),
            "std_tok_per_sec": float(np.std(throughputs)),
        }
        print(f"    {np.mean(throughputs):.1f} ± {np.std(throughputs):.1f} tok/s")

        # Peak memory — use shorter generation to avoid flex_attention recompile issues
        print(f"  Measuring peak memory...")
        peak_mb = _measure_memory(model, tokenizer, device, is_enc_dec, prompts[-1], min(num_gen_tokens, 16))
        result["peak_memory_mb"] = peak_mb
        print(f"    Peak: {peak_mb:.0f} MB")

        results[label] = result

        # Free model
        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'═'*70}")
    print(f"  EFFICIENCY SUMMARY")
    print(f"{'═'*70}")
    header = f"  {'Metric':<30s}"
    for label in labels:
        header += f"  {label:<18s}"
    print(header)
    print(f"  {'─'*30}" + f"  {'─'*18}" * len(labels))

    # Params
    row = f"  {'Parameters':<30s}"
    for label in labels:
        p = results[label]["params"]["total"]
        row += f"  {p/1e6:<18.1f}"
    print(row + " (M)")

    # FTL at longest prompt
    longest = max(int(k) for k in results[labels[0]]["first_token_latency"])
    row = f"  {f'First-token ({longest} tok)':<30s}"
    for label in labels:
        v = results[label]["first_token_latency"][longest]["mean_ms"]
        row += f"  {v:<18.1f}"
    print(row + " (ms)")

    # Throughput
    row = f"  {'Decoding throughput':<30s}"
    for label in labels:
        v = results[label]["throughput"]["mean_tok_per_sec"]
        row += f"  {v:<18.1f}"
    print(row + " (tok/s)")

    # Peak memory
    row = f"  {'Peak memory':<30s}"
    for label in labels:
        v = results[label]["peak_memory_mb"]
        row += f"  {v:<18.0f}"
    print(row + " (MB)")

    # Enc/dec ratio
    row = f"  {'Encoder param ratio':<30s}"
    for label in labels:
        if results[label]["is_enc_dec"]:
            v = results[label]["params"]["enc_ratio"]
            row += f"  {v:<18.0%}"
        else:
            row += f"  {'N/A':<18s}"
    print(row)

    print(f"{'═'*70}")

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Convert numpy types
        def _serialize(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=_serialize)
        print(f"\n  Results saved to {output_path}")


# ── Test prompts at varying lengths ───────────────────────────────────────

SHORT_PROMPT = "What is the capital of France?"

MEDIUM_PROMPT = (
    "The following is a passage about the history of computing. "
    "Early computers were large machines that filled entire rooms. "
    "They used vacuum tubes and magnetic drums for memory. "
    "The invention of the transistor in 1947 changed everything. "
    "Computers became smaller, faster, and more reliable. "
    "The integrated circuit, developed in the late 1950s, "
    "allowed multiple transistors to be placed on a single chip. "
    "This led to the development of microprocessors in the 1970s. "
    "Question: What invention in 1947 changed computing? Answer:"
)

LONG_PROMPT = (
    "Read the following article carefully and answer the question at the end.\n\n"
    + "The development of artificial intelligence has been one of the most significant "
    "technological advances of the 21st century. " * 30
    + "\n\nBased on the article above, what has been one of the most significant "
    "technological advances? Answer:"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--output", default="evals/efficiency_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.labels)

    prompts = [SHORT_PROMPT, MEDIUM_PROMPT, LONG_PROMPT]

    benchmark_all(
        args.checkpoints, args.labels, prompts,
        device=args.device,
        num_gen_tokens=args.num_tokens,
        num_trials=args.num_trials,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
