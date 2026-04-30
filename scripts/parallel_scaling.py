#!/usr/bin/env python3
"""
Train all 5 scaling-law Double_Decoder models on one GPU in a single process.

Instead of launching 25 sequential subprocesses (scaling_laws.py run), this
trains all 5 model sizes (0.5M–30M) concurrently in a shared training loop,
eliminating per-run startup overhead and maximizing GPU utilization by keeping
all models resident in memory simultaneously.

Uses fast tensor masks instead of flex_attention's create_block_mask to avoid
torch.compile overhead that dominates runtime for small models.

Usage:
    python scripts/parallel_scaling.py
    python scripts/parallel_scaling.py --tokens 50_000_000 --batch-size 48
    python scripts/parallel_scaling.py --compile            # enable torch.compile
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import PreTrainedTokenizerFast

from models.double_decoder import Double_Decoder
from collators.double_decoder.pretrain import BasicPretrainCollator
from components.initialization import initialize_model

# ── Constants ───────────────────────────────────────────────────────────────

ARCHITECTURES = [
    ("0.5M",  dict(dim=64,  num_encoder_layers=7,  num_decoder_layers=3)),
    ("2.5M",  dict(dim=128, num_encoder_layers=9,  num_decoder_layers=4)),
    ("5M",    dict(dim=192, num_encoder_layers=8,  num_decoder_layers=4)),
    ("15M",   dict(dim=256, num_encoder_layers=13, num_decoder_layers=6)),
    ("30M",   dict(dim=384, num_encoder_layers=12, num_decoder_layers=5)),
]

SEQ_LEN = 2048
MUP_BASE_DIM = 64
BASE_LR = 0.002  # lr_for_dim(64) = 2e-3 * sqrt(64/64)

# BF16 peak TFLOPS for MFU calculation
GPU_PEAK_TFLOPS = {
    "H100": 990, "H200": 990, "A100": 312, "A100-SXM": 624,
    "L40": 362, "B200": 4500,
}


# ── Monkey-patches for fast tensor masks ────────────────────────────────────
#
# The block structure is simple:
#   self_mask[q, kv]  = (q >= kv) AND (same_block)
#   cross_mask[q, kv] = (kv_block < q_block) OR (kv == 0)
#
# flex_attention's create_block_mask(_compile=True) runs torch.compile to
# build fused kernels for these masks — great for long sequences, but the
# compilation overhead dominates for small models at seq_len=2048.
#
# Instead, we create [seq_len, seq_len] boolean tensors (16MB total) and
# use standard SDPA / manual attention. Same block structure, no compile.

def _create_pretrain_masks_tensor(blocks, seq_len, device):
    """Same semantics as create_pretrain_masks, returns plain bool tensors."""
    from components.block_masks import _precompute_block_ids
    block_ids = _precompute_block_ids(blocks, seq_len, device)
    pos = torch.arange(seq_len, device=device)

    # Self: causal AND same block
    causal = pos[:, None] >= pos[None, :]                    # [Q, KV]
    same_block = block_ids[:, None] == block_ids[None, :]    # [Q, KV]
    self_mask = causal & same_block

    # Cross: previous block OR first token
    earlier = block_ids[None, :] < block_ids[:, None]        # [Q, KV]
    first_token = pos[None, :] == 0                          # [1, KV]
    cross_mask = earlier | first_token

    return {"self_mask": self_mask, "cross_mask": cross_mask}


def _patched_create_masks(batch_size, blocks, device, input_ids,
                          encoder_input_ids, decoder_input_ids, sft):
    """Drop-in replacement for block_masks.create_masks using tensor masks."""
    if sft:
        # SFT not used in scaling runs — fall back to original
        from components.block_masks import create_sft_masks
        return create_sft_masks(batch_size, blocks, device,
                                encoder_input_ids.shape[1],
                                decoder_input_ids.shape[1])
    return _create_pretrain_masks_tensor(blocks, input_ids.shape[1], device)


def _patched_self_attn_forward(self, x, block_masks=None, input_pos=None, **kwargs):
    """SelfAttention.forward with tensor mask support via SDPA."""
    B, L, D = x.size()

    qkv = self.qkv(x)
    query, key, value = torch.split(qkv, [self.dim, self.dim, self.dim], dim=-1)
    query = query.reshape(B, L, self.num_heads, self.head_dim)
    key = key.reshape(B, L, self.num_heads, self.head_dim)
    value = value.reshape(B, L, self.num_heads, self.head_dim)

    query = self.rotary_emb(query, input_pos=input_pos)
    key = self.rotary_emb(key, input_pos=input_pos)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    self_mask = None if block_masks is None else block_masks['self_mask']

    if self_mask is None:
        output = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=self.attn_scale)
    elif isinstance(self_mask, torch.Tensor):
        # Tensor mask: use SDPA with bool attn_mask [Q, KV]
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=self_mask, scale=self.attn_scale)
    else:
        from torch.nn.attention.flex_attention import flex_attention
        output = flex_attention(query, key, value,
                                block_mask=self_mask, scale=self.attn_scale)

    if self.gating:
        gating_modulator = self.gater(x).reshape(B, self.num_heads, L, self.head_dim)
        output = output * gating_modulator

    output = output.transpose(1, 2).reshape(B, L, D)
    return self.out_proj(output)


def _patched_combo_attn_forward(self, x, encoder_inputs,
                                block_masks=None, decoder_input_positions=None):
    """ComboAttention.forward with tensor mask support via manual attention."""
    B, L, D = x.size()
    _, L_enc, _ = encoder_inputs.size()

    # --- Projections (unchanged) ---
    query = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)

    dec_kv = self.kv_proj(x) if self.shared else self.dec_kv_proj(x)
    dec_key, dec_value = torch.split(dec_kv, [self.dim, self.dim], dim=-1)
    dec_key = dec_key.reshape(B, L, self.num_heads, self.head_dim)
    dec_value = dec_value.reshape(B, L, self.num_heads, self.head_dim)

    query = self.query_rotary(query, input_pos=decoder_input_positions)
    dec_key = self.key_rotary(dec_key, input_pos=decoder_input_positions)

    query = query.transpose(1, 2)
    dec_key = dec_key.transpose(1, 2)
    dec_value = dec_value.transpose(1, 2)

    enc_kv = self.kv_proj(encoder_inputs) if self.shared else self.enc_kv_proj(encoder_inputs)
    enc_key, enc_value = torch.split(enc_kv, [self.dim, self.dim], dim=-1)
    enc_key = enc_key.reshape(B, L_enc, self.num_heads, self.head_dim)
    enc_value = enc_value.reshape(B, L_enc, self.num_heads, self.head_dim)

    enc_key = self.key_rotary(enc_key)
    enc_key = enc_key.transpose(1, 2)
    enc_value = enc_value.transpose(1, 2)

    # --- Attention ---
    try:
        from flash_attn import flash_attn_func
        HAS_FLASH = True
    except ImportError:
        HAS_FLASH = False

    use_flash = block_masks is None and HAS_FLASH

    if use_flash:
        query, enc_key, dec_key = query.transpose(1, 2), enc_key.transpose(1, 2), dec_key.transpose(1, 2)
        enc_value, dec_value = enc_value.transpose(1, 2), dec_value.transpose(1, 2)
        dec_output, dec_lse, _ = flash_attn_func(
            query, dec_key, dec_value, causal=True,
            return_attn_probs=True, softmax_scale=self.attn_scale)
        enc_output, enc_lse, _ = flash_attn_func(
            query, enc_key, enc_value, causal=False,
            return_attn_probs=True, softmax_scale=self.attn_scale)

    elif block_masks is None or isinstance(block_masks.get('self_mask'), torch.Tensor):
        # Manual attention: either no masks or tensor block masks.
        # We need LSE for the sigmoid gate, so compute scores explicitly.
        scale = self.attn_scale or 1.0 / (self.head_dim ** 0.5)

        # Decoder self-attention
        dec_scores = torch.matmul(query, dec_key.transpose(-2, -1)) * scale
        if block_masks is not None:
            # Tensor block mask [Q, KV] broadcasts to [B, H, Q, KV]
            dec_scores.masked_fill_(~block_masks['self_mask'], float('-inf'))
        else:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            dec_scores.masked_fill_(causal_mask, float('-inf'))
        dec_lse = torch.logsumexp(dec_scores, dim=-1)
        dec_attn = torch.softmax(dec_scores, dim=-1)
        dec_output = torch.matmul(dec_attn, dec_value)

        # Encoder cross-attention
        enc_scores = torch.matmul(query, enc_key.transpose(-2, -1)) * scale
        if block_masks is not None:
            enc_scores.masked_fill_(~block_masks['cross_mask'], float('-inf'))
        enc_lse = torch.logsumexp(enc_scores, dim=-1)
        enc_attn = torch.softmax(enc_scores, dim=-1)
        enc_output = torch.matmul(enc_attn, enc_value)

    else:
        # flex_attention path (BlockMask objects)
        from torch.nn.attention.flex_attention import flex_attention
        self_mask = block_masks['self_mask']
        cross_mask = block_masks['cross_mask']
        dec_output, dec_lse = flex_attention(
            query, dec_key, dec_value,
            block_mask=self_mask, return_lse=True, scale=self.attn_scale)
        enc_output, enc_lse = flex_attention(
            query, enc_key, enc_value,
            block_mask=cross_mask, return_lse=True, scale=self.attn_scale)

    # --- Mixing (unchanged) ---
    if self.logit_biases:
        dec_lse = ((dec_lse + self.logit_bias_proj[:, 0].reshape(1, self.num_heads, 1))
                   * self.mix_temp.reshape(1, self.num_heads, 1))
        enc_lse = ((enc_lse + self.logit_bias_proj[:, 1].reshape(1, self.num_heads, 1))
                   * self.mix_temp.reshape(1, self.num_heads, 1))

    dec_w = torch.sigmoid(dec_lse - enc_lse).unsqueeze(-1)
    dec_w = dec_w.transpose(1, 2) if use_flash else dec_w
    output = dec_w * dec_output + (1 - dec_w) * enc_output

    output = output.reshape(B, L, D) if use_flash else output.transpose(1, 2).reshape(B, L, D)
    return self.out_proj(output)


def install_fast_masks():
    """Monkey-patch block_masks and attention modules for fast tensor masks."""
    import components.block_masks as bm
    from components.attention import SelfAttention, ComboAttention

    bm.create_masks = _patched_create_masks
    SelfAttention.forward = _patched_self_attn_forward
    ComboAttention.forward = _patched_combo_attn_forward


# ── Helpers ─────────────────────────────────────────────────────────────────

def detect_gpu_tflops():
    """Best-effort GPU peak TFLOPS detection for MFU reporting."""
    if not torch.cuda.is_available():
        return 200.0
    name = torch.cuda.get_device_name(0).upper()
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.upper() in name:
            return float(val)
    return 200.0


def non_emb_param_count(model):
    """Non-embedding parameters (the '6ND' N)."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.embedding.weight.numel()
    return total - emb


def build_model(arch, vocab_size, device, use_compile=False):
    dim = arch["dim"]
    model = Double_Decoder(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=dim // 64,
        num_encoder_layers=arch["num_encoder_layers"],
        num_decoder_layers=arch["num_decoder_layers"],
        seq_len=SEQ_LEN,
        shared=True,
        logit_biases=False,
        init_strategy="xavier_uniform",
        gradient_checkpointing=False,
        mup_base_dim=MUP_BASE_DIM,
    )
    model = model.to(device)
    if use_compile:
        model = torch.compile(model)
    return model


def build_optimizer(model, dim):
    """AdamW with μP per-group learning rates."""
    mup_mult = MUP_BASE_DIM / dim
    embed_params, hidden_decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embedding" in name or "output_projection" in name:
            embed_params.append(p)
        elif isinstance(dict(model.named_modules()).get(name.rsplit(".", 1)[0]), (nn.LayerNorm, nn.RMSNorm)):
            no_decay.append(p)
        elif p.dim() <= 1:
            no_decay.append(p)
        else:
            hidden_decay.append(p)
    return AdamW([
        {"params": embed_params, "lr": BASE_LR, "weight_decay": 0.1},
        {"params": hidden_decay, "lr": BASE_LR * mup_mult, "weight_decay": 0.1},
        {"params": no_decay, "lr": BASE_LR, "weight_decay": 0.0},
    ], betas=(0.9, 0.95), eps=1e-8, fused=True)


def build_scheduler(optimizer, total_steps):
    """Linear warmup (5%) → linear decay to 10% of peak."""
    warmup = max(1, int(total_steps * 0.05))
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 1.0 - progress * 0.9)
    return LambdaLR(optimizer, lr_lambda)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel scaling-law training")
    parser.add_argument("--tokens", type=int, default=10_000_000,
                        help="Total training tokens per model (default: 10M)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Micro-batch size in sequences (default: 32)")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps (default: 16, eff_batch=512)")
    parser.add_argument("--eval-batches", type=int, default=10,
                        help="Number of eval batches per model (default: 10)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/parallel_scaling")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (on by default)")
    parser.add_argument("--train-file", type=str,
                        default="data/Pretrain/slimpajama_6b_packed.jsonl")
    parser.add_argument("--eval-file", type=str,
                        default="data/Pretrain/slimpajama_6b_eval_packed.jsonl")
    parser.add_argument("--tokenizer-file", type=str,
                        default="tokenizer/tokenizer_32k.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    use_compile = not args.no_compile

    # Patch in fast tensor masks (no flex_attention / torch.compile overhead)
    install_fast_masks()

    gpu_tflops = detect_gpu_tflops()
    gpu_name = torch.cuda.get_device_name(0)
    tokens_per_step = args.batch_size * args.grad_accum * SEQ_LEN
    total_steps = max(1, args.tokens // tokens_per_step)
    total_micro = total_steps * args.grad_accum

    print(f"GPU: {gpu_name}  (peak BF16: {gpu_tflops:.0f} TFLOPS)")
    print(f"Tokens per model: {args.tokens:,}  |  Optimizer steps: {total_steps}")
    print(f"Effective batch: {args.batch_size} × {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} seqs = {tokens_per_step:,} tok/step")
    print(f"torch.compile: {'ON' if use_compile else 'OFF'}  |  matmul precision: TF32")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(PROJECT_ROOT / args.tokenizer_file))
    vocab_size = tokenizer.vocab_size
    bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    print(f"Tokenizer: vocab_size={vocab_size}  bos={bos_token_id}  eos={eos_token_id}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading data...")
    collator = BasicPretrainCollator(
        bos_token_id=bos_token_id, eos_token_id=eos_token_id, max_seq_len=SEQ_LEN)

    train_ds = load_dataset(
        "json", data_files=str(PROJECT_ROOT / args.train_file),
        split="train", streaming=True,
    )
    eval_ds = load_dataset(
        "json", data_files=str(PROJECT_ROOT / args.eval_file),
        split="train",
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        collate_fn=collator, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size,
        collate_fn=collator, drop_last=True,
    )

    # ── Create all models ───────────────────────────────────────────────────
    print(f"\nBuilding {len(ARCHITECTURES)} models:\n")
    trainers = []
    for name, arch in ARCHITECTURES:
        model = build_model(arch, vocab_size, device, use_compile=use_compile)
        opt = build_optimizer(model, arch["dim"])
        sched = build_scheduler(opt, total_steps)
        ne = non_emb_param_count(model)
        mup_mult = MUP_BASE_DIM / arch["dim"]
        print(f"  {name:>5}: dim={arch['dim']:>3}  "
              f"layers={arch['num_encoder_layers']:>2}+{arch['num_decoder_layers']:<2}  "
              f"non_emb={ne:>10,}  hidden_lr={BASE_LR * mup_mult:.1e}")
        trainers.append({
            "name": name, "model": model, "opt": opt, "sched": sched,
            "ne": ne, "step": 0, "micro": 0,
            "loss_sum": 0.0, "loss_n": 0,
            "curve": [], "tokens_seen": 0,
        })

    total_ne = sum(t["ne"] for t in trainers)
    total_all = sum(sum(p.numel() for p in t["model"].parameters()) for t in trainers)
    print(f"\n  Total params (all models): {total_all:,}")
    print(f"  Total non-emb params:      {total_ne:,}")
    print(f"  Weight memory (bf16):      ~{total_all * 2 / 1e6:.0f} MB")
    print(f"  Optimizer memory (fp32):   ~{total_all * 8 / 1e6:.0f} MB")

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training {len(trainers)} models × {total_steps} steps "
          f"({total_micro} micro-batches)")
    print(f"{'='*70}\n")

    for t in trainers:
        t["model"].train()
        t["opt"].zero_grad(set_to_none=True)

    # Warmup: trigger torch.compile tracing on a dummy batch
    if use_compile:
        print("\nCompiling models (one-time cost)...")
        compile_t0 = time.time()
        dummy_ids = torch.randint(0, vocab_size, (args.batch_size, SEQ_LEN), device=device)
        dummy_labels = dummy_ids.clone()
        dummy_blocks = torch.sort(torch.randperm(SEQ_LEN - 2, device=device)[:4] + 1)[0]
        dummy_batch = {"input_ids": dummy_ids, "labels": dummy_labels,
                       "blocks": dummy_blocks, "sft": False}
        for t in trainers:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = t["model"](**dummy_batch)
            out["loss"].backward()
            t["opt"].zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f"  Compiled in {time.time() - compile_t0:.1f}s\n")

    t0 = time.time()

    for batch_idx, raw_batch in enumerate(train_loader):
        if batch_idx >= total_micro:
            break

        # Move batch to GPU (shared across all models)
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in raw_batch.items()
        }

        # Sanity check on first batch
        if batch_idx == 0:
            max_id = batch["input_ids"].max().item()
            assert max_id < vocab_size, (
                f"Token ID {max_id} >= vocab_size {vocab_size}. "
                f"Data was likely tokenized with a different tokenizer. "
                f"Try --tokenizer-file tokenizer/tokenizer_32k.json")

        # Forward + backward for each model sequentially
        for t in trainers:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = t["model"](**batch)
                loss = out["loss"]

            (loss / args.grad_accum).backward()
            t["loss_sum"] += loss.detach().item()
            t["loss_n"] += 1
            t["micro"] += 1

            # Optimizer step after accumulation
            if t["micro"] % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(t["model"].parameters(), 1.0)
                t["opt"].step()
                t["sched"].step()
                t["opt"].zero_grad(set_to_none=True)
                t["step"] += 1
                t["tokens_seen"] += tokens_per_step

                avg_loss = t["loss_sum"] / t["loss_n"]
                t["curve"].append((t["step"], round(avg_loss, 4)))
                t["loss_sum"] = 0.0
                t["loss_n"] = 0

        # Log once per optimizer step
        current_step = trainers[0]["step"]
        if current_step > 0 and trainers[0]["micro"] % args.grad_accum == 0:
            elapsed = time.time() - t0
            total_flops = sum(6 * t["ne"] * t["tokens_seen"] for t in trainers)
            agg_mfu = total_flops / (elapsed * gpu_tflops * 1e12) * 100
            losses = "  ".join(
                f'{t["name"]}={t["curve"][-1][1]:.3f}' for t in trainers
            )
            print(f"  step {current_step:>4}/{total_steps}  "
                  f"[{elapsed:5.1f}s]  agg_MFU={agg_mfu:.1f}%  {losses}")

    torch.cuda.synchronize()
    train_time = time.time() - t0

    # Final MFU
    total_flops = sum(6 * t["ne"] * t["tokens_seen"] for t in trainers)
    agg_mfu = total_flops / (train_time * gpu_tflops * 1e12) * 100

    print(f"\n{'='*70}")
    print(f"  Training done in {train_time:.1f}s")
    print(f"  Aggregate MFU: {agg_mfu:.1f}%  "
          f"(total FLOPs: {total_flops:.2e})")
    print(f"{'='*70}")

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("\nEvaluating...")
    results = {}
    for t in trainers:
        model = t["model"]
        model.eval()
        total_loss, total_tok = 0.0, 0
        with torch.no_grad():
            for i, raw_batch in enumerate(eval_loader):
                if i >= args.eval_batches:
                    break
                batch = {
                    k: v.to(device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in raw_batch.items()
                }
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                ntok = (batch["labels"] != -100).sum().item()
                total_loss += out["loss"].item() * ntok
                total_tok += ntok

        avg_loss = total_loss / max(1, total_tok)
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        print(f"  {t['name']:>5}: eval_loss={avg_loss:.4f}  ppl={ppl:.1f}")

        results[t["name"]] = {
            "name": t["name"],
            "non_emb_params": t["ne"],
            "total_params": sum(p.numel() for p in t["model"].parameters()),
            "tokens_seen": t["tokens_seen"],
            "steps": t["step"],
            "final_eval_loss": round(avg_loss, 4),
            "final_eval_ppl": round(ppl, 2),
            "train_curve": t["curve"],
            "training_time_sec": round(train_time, 2),
            "aggregate_mfu_pct": round(agg_mfu, 2),
        }

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_label = f"{args.tokens // 1_000_000}M"
    out_path = out_dir / f"parallel_{tok_label}tok_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
