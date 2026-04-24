"""
Combo Attention Weight Analysis — unique to the block-based double-decoder.

Hooks into ComboAttention layers to extract the sigmoid(dec_lse - enc_lse)
blending weights, revealing how the model dynamically balances self-attention
(within-block decoder) vs. cross-attention (to encoder context).
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def _has_combo_attention(model):
    """Check if the model uses ComboAttention (encoder-decoder architecture)."""
    return hasattr(model, "decoder_layers")


def eval_attention_weights(model, tokenizer, device, is_enc_dec,
                           eval_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                           max_examples=50, output_dir="evals/attention_outputs"):
    """Extract and analyze combo attention blending weights.

    For each ComboAttention layer, captures sigmoid(dec_lse - enc_lse)
    across sequence positions. This weight controls how much the decoder
    relies on its own self-attention vs. cross-attention to the encoder.

    dec_w ≈ 1.0 → self-attention dominates
    dec_w ≈ 0.0 → cross-attention dominates
    """
    if not is_enc_dec or not _has_combo_attention(model):
        return {"name": "Attention Analysis", "error": "Not an encoder-decoder model"}

    # Register hooks on ComboAttention layers
    blending_weights = {}
    hooks = []

    for layer_idx, dec_layer in enumerate(model.decoder_layers):
        combo_attn = dec_layer.combo_attn
        storage_key = f"layer_{layer_idx}"
        blending_weights[storage_key] = []

        def make_hook(key):
            def hook_fn(module, input, output):
                # ComboAttention.forward computes dec_w = sigmoid(dec_lse - enc_lse)
                # We need to intercept the computation. Since we can't easily hook
                # into the middle of forward, we'll recompute from stored LSE values.
                pass
            return hook_fn

    # Alternative approach: monkey-patch ComboAttention.forward to store blending weights
    original_forwards = {}

    for layer_idx, dec_layer in enumerate(model.decoder_layers):
        combo_attn = dec_layer.combo_attn
        key = f"layer_{layer_idx}"
        blending_weights[key] = []
        original_forwards[key] = combo_attn.forward

        def make_patched_forward(original_fn, store_key):
            def patched_forward(x, encoder_inputs, block_masks=None, decoder_input_positions=None):
                # Call original to get output, but also capture dec_lse and enc_lse
                B, L, D = x.size()
                _, L_enc, _ = encoder_inputs.size()

                query = combo_attn.q_proj(x).reshape(B, L, combo_attn.num_heads, combo_attn.head_dim)
                dec_kv = combo_attn.kv_proj(x) if combo_attn.shared else combo_attn.dec_kv_proj(x)
                dec_key, dec_value = torch.split(dec_kv, [combo_attn.dim, combo_attn.dim], dim=-1)
                dec_key = dec_key.reshape(B, L, combo_attn.num_heads, combo_attn.head_dim)
                dec_value = dec_value.reshape(B, L, combo_attn.num_heads, combo_attn.head_dim)

                query = combo_attn.query_rotary(query, input_pos=decoder_input_positions)
                dec_key = combo_attn.key_rotary(dec_key, input_pos=decoder_input_positions)

                query = query.transpose(1, 2)
                dec_key = dec_key.transpose(1, 2)
                dec_value = dec_value.transpose(1, 2)

                enc_kv = combo_attn.kv_proj(encoder_inputs) if combo_attn.shared else combo_attn.enc_kv_proj(encoder_inputs)
                enc_key, enc_value = torch.split(enc_kv, [combo_attn.dim, combo_attn.dim], dim=-1)
                enc_key = enc_key.reshape(B, L_enc, combo_attn.num_heads, combo_attn.head_dim)
                enc_value = enc_value.reshape(B, L_enc, combo_attn.num_heads, combo_attn.head_dim)
                enc_key = combo_attn.key_rotary(enc_key)
                enc_key = enc_key.transpose(1, 2)
                enc_value = enc_value.transpose(1, 2)

                from torch.nn.attention.flex_attention import flex_attention
                try:
                    from flash_attn import flash_attn_func
                    has_flash = True
                except ImportError:
                    has_flash = False

                use_flash = block_masks is None and has_flash
                if use_flash:
                    query_f = query.transpose(1, 2)
                    dec_key_f, dec_value_f = dec_key.transpose(1, 2), dec_value.transpose(1, 2)
                    enc_key_f, enc_value_f = enc_key.transpose(1, 2), enc_value.transpose(1, 2)
                    dec_output, dec_lse, _ = flash_attn_func(query_f, dec_key_f, dec_value_f, causal=True, return_attn_probs=True)
                    enc_output, enc_lse, _ = flash_attn_func(query_f, enc_key_f, enc_value_f, causal=False, return_attn_probs=True)
                elif block_masks is not None:
                    self_mask = block_masks.get("self_mask")
                    cross_mask = block_masks.get("cross_mask")
                    dec_output, dec_lse = flex_attention(query, dec_key, dec_value, block_mask=self_mask, return_lse=True)
                    enc_output, enc_lse = flex_attention(query, enc_key, enc_value, block_mask=cross_mask, return_lse=True)
                else:
                    return original_fn(x, encoder_inputs, block_masks, decoder_input_positions)

                if combo_attn.logit_biases:
                    dec_lse = (dec_lse + combo_attn.logit_bias_proj[:, 0].reshape(1, combo_attn.num_heads, 1)) * combo_attn.mix_temp.reshape(1, combo_attn.num_heads, 1)
                    enc_lse = (enc_lse + combo_attn.logit_bias_proj[:, 1].reshape(1, combo_attn.num_heads, 1)) * combo_attn.mix_temp.reshape(1, combo_attn.num_heads, 1)

                dec_w = torch.sigmoid(dec_lse - enc_lse)

                # Store blending weights: [B, num_heads, seq_len]
                blending_weights[store_key].append(
                    dec_w.detach().cpu().float().numpy()
                )

                dec_w_expanded = dec_w.unsqueeze(-1)
                dec_w_expanded = dec_w_expanded.transpose(1, 2) if use_flash else dec_w_expanded
                output = dec_w_expanded * dec_output + (1 - dec_w_expanded) * enc_output
                output = output.reshape(B, L, D) if use_flash else output.transpose(1, 2).reshape(B, L, D)
                return combo_attn.out_proj(output)

            return patched_forward
        combo_attn.forward = make_patched_forward(original_forwards[key], key)

    # Run forward passes to collect weights
    ds = load_dataset("json", data_files=eval_file, split="train", streaming=True)

    model.eval()
    with torch.no_grad():
        for i, example in enumerate(tqdm(ds, desc="Attention Analysis", total=max_examples)):
            if i >= max_examples:
                break
            ids = example["input_ids"]
            input_t = torch.tensor([ids], device=device)
            seq_len = len(ids)
            blocks = torch.tensor([seq_len // 2], device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(input_ids=input_t, blocks=blocks, sft=False)

    # Restore original forwards
    for layer_idx, dec_layer in enumerate(model.decoder_layers):
        key = f"layer_{layer_idx}"
        dec_layer.combo_attn.forward = original_forwards[key]

    # Aggregate and analyze
    analysis = {}
    for key, weight_list in blending_weights.items():
        if not weight_list:
            continue
        # Stack: [num_examples, B, num_heads, seq_len]
        # Since B=1 during eval, squeeze
        weights = np.concatenate(weight_list, axis=0)  # [N, num_heads, seq_len]

        # Per-head mean blending weight (how much does each head prefer self vs cross?)
        per_head_mean = weights.mean(axis=(0, 2)).tolist()  # [num_heads]

        # Per-position mean (averaged over examples and heads)
        per_position_mean = weights.mean(axis=(0, 1)).tolist()  # [seq_len]

        # Overall statistics
        analysis[key] = {
            "per_head_mean_dec_weight": per_head_mean,
            "per_position_mean_dec_weight": per_position_mean,
            "overall_mean": float(weights.mean()),
            "overall_std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
        }

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "attention_weights.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("\n── Combo Attention Blending Weight Analysis ──")
    print("  dec_w ≈ 1.0 → self-attention dominates")
    print("  dec_w ≈ 0.0 → cross-attention dominates\n")
    for key, stats in analysis.items():
        print(f"  {key}:")
        print(f"    Overall mean dec_w: {stats['overall_mean']:.4f} ± {stats['overall_std']:.4f}")
        head_means = stats["per_head_mean_dec_weight"]
        for h, m in enumerate(head_means):
            label = "self" if m > 0.5 else "cross"
            print(f"    Head {h}: {m:.4f} ({label}-dominant)")

    return {
        "name": "Attention Analysis",
        "analysis": analysis,
        "output_path": output_path,
    }
