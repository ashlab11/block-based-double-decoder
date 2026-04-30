#!/usr/bin/env python3
"""
Full scaling-law grid: 3 model types × 5 param sizes × 5 token budgets.

Trains Double_Decoder (dd), StandardEncDec (sed), and DecoderOnlyModel (dec)
at all 25 (param, token) grid points. Models are compiled once and
re-initialized for each token budget.

Usage:
    python scripts/parallel_scaling.py                              # full grid
    python scripts/parallel_scaling.py --token-budgets 10M,50M      # subset
    python scripts/parallel_scaling.py --model-types dd,dec          # subset
    python scripts/parallel_scaling.py --no-compile                  # skip compile
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch._dynamo
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import PreTrainedTokenizerFast

from models.double_decoder import Double_Decoder
from models.standard_enc_dec import StandardEncDec
from models.decoder import DecoderOnlyModel
from collators.double_decoder.pretrain import BasicPretrainCollator
from components.initialization import initialize_model

# ── Constants ───────────────────────────────────────────────────────────────

MODEL_TYPES = ["dd", "sed", "dec"]
MODEL_TYPE_NAMES = {"dd": "Double_Decoder", "sed": "StandardEncDec", "dec": "DecoderOnly"}

# Fixed depth, width-only scaling for clean μP transfer.
# enc=8, dec=4 across all sizes; only dim varies.
# DD/Dec non-emb ≈ 144·dim²;  StandardEncDec ≈ 160·dim² (extra cross-attn params).
FIXED_ENC_LAYERS = 8
FIXED_DEC_LAYERS = 4

ARCHITECTURES = [
    ("0.6M",  dict(dim=64,  num_encoder_layers=8, num_decoder_layers=4)),
    ("2.4M",  dict(dim=128, num_encoder_layers=8, num_decoder_layers=4)),
    ("5.3M",  dict(dim=192, num_encoder_layers=8, num_decoder_layers=4)),
    ("14.7M", dict(dim=320, num_encoder_layers=8, num_decoder_layers=4)),
    ("28.9M", dict(dim=448, num_encoder_layers=8, num_decoder_layers=4)),
]

TOKEN_BUDGETS = [
    ("10M",  10_000_000),
    ("50M",  50_000_000),
    ("100M", 100_000_000),
    ("300M", 300_000_000),
    ("600M", 600_000_000),
]

SEQ_LEN = 2048
MUP_BASE_DIM = 64
BASE_LR = 0.002

GPU_PEAK_TFLOPS = {
    "H100": 990, "H200": 990, "A100": 312, "A100-SXM": 624,
    "L40": 362, "B200": 4500,
}


# ── Monkey-patch: cache block masks across models ───────────────────────────

_mask_cache_blocks_id = None
_mask_cache_result = None

def _cached_create_masks(batch_size, blocks, device, input_ids,
                         encoder_input_ids, decoder_input_ids, sft):
    global _mask_cache_blocks_id, _mask_cache_result
    blocks_id = id(blocks)
    if blocks_id != _mask_cache_blocks_id:
        _mask_cache_blocks_id = blocks_id
        from components.block_masks import create_pretrain_masks, create_sft_masks
        if sft:
            _mask_cache_result = create_sft_masks(
                batch_size, blocks, device,
                encoder_input_ids.shape[1], decoder_input_ids.shape[1])
        else:
            _mask_cache_result = create_pretrain_masks(
                blocks, input_ids.shape[1], device)
    return _mask_cache_result


def install_fast_masks():
    """Monkey-patch create_masks with a cached version for DD and StandardEncDec."""
    import models.double_decoder as dd
    import models.standard_enc_dec as sed
    dd.create_masks = _cached_create_masks
    sed.create_masks = _cached_create_masks


# ── Helpers ─────────────────────────────────────────────────────────────────

def detect_gpu_tflops():
    if not torch.cuda.is_available():
        return 200.0
    name = torch.cuda.get_device_name(0).upper()
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.upper() in name:
            return float(val)
    return 200.0


def non_emb_param_count(model):
    total = sum(p.numel() for p in model.parameters())
    emb = model.embedding.weight.numel()
    return total - emb


def build_model(model_type, arch, vocab_size, device, use_compile=False):
    dim = arch["dim"]
    num_heads = dim // 64
    enc = arch["num_encoder_layers"]
    dec = arch["num_decoder_layers"]

    if model_type == "dd":
        model = Double_Decoder(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads,
            num_encoder_layers=enc, num_decoder_layers=dec,
            seq_len=SEQ_LEN, shared=True, logit_biases=False,
            init_strategy="xavier_uniform", gradient_checkpointing=False,
            mup_base_dim=MUP_BASE_DIM)
    elif model_type == "sed":
        # StandardDecoder layers are 16d² (self+cross+MLP) vs DD's 12d².
        # Use fewer decoder layers to match DD/Dec param count at same width:
        # DD: (enc+dec)*12d² = 144d²;  SED: enc*12d² + sed_dec*16d² = 144d²
        # → sed_dec = dec * 12/16 = dec * 3/4
        sed_dec = max(1, (dec * 3) // 4)
        model = StandardEncDec(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads,
            num_encoder_layers=enc, num_decoder_layers=sed_dec,
            seq_len=SEQ_LEN, init_strategy="xavier_uniform",
            gradient_checkpointing=False, mup_base_dim=MUP_BASE_DIM)
    elif model_type == "dec":
        model = DecoderOnlyModel(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads,
            num_layers=enc + dec, seq_len=SEQ_LEN,
            init_strategy="xavier_uniform", mup_base_dim=MUP_BASE_DIM)

    model = model.to(device)
    if use_compile:
        model = torch.compile(model)
    return model


def build_optimizer(model, dim):
    mup_mult = MUP_BASE_DIM / dim
    embed_params, hidden_decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embedding" in name or "output_projection" in name:
            embed_params.append(p)
        elif isinstance(dict(model.named_modules()).get(
                name.rsplit(".", 1)[0]), (nn.LayerNorm, nn.RMSNorm)):
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
    warmup = max(1, int(total_steps * 0.05))
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 1.0 - progress * 0.9)
    return LambdaLR(optimizer, lr_lambda)


def model_forward(model, batch, needs_blocks):
    """Dispatch forward based on model type."""
    if needs_blocks:
        return model(**batch)
    else:
        return model(input_ids=batch["input_ids"], labels=batch["labels"])


def eval_model(model, eval_loader, device, max_batches, needs_blocks):
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for i, raw_batch in enumerate(eval_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in raw_batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                try:
                    out = model_forward(model, batch, needs_blocks)
                except Exception:
                    # Fallback to eager if compiled eval hits inductor bugs
                    eager = getattr(model, "_orig_mod", model)
                    out = model_forward(eager, batch, needs_blocks)
            ntok = (batch["labels"] != -100).sum().item()
            total_loss += out["loss"].item() * ntok
            total_tok += ntok
    model.train()
    avg_loss = total_loss / max(1, total_tok)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    return avg_loss, ppl


# ── Training one token budget ───────────────────────────────────────────────

def train_one_budget(models_info, tok_label, total_tokens, batch_size, grad_accum,
                     train_loader, eval_loader, device, gpu_tflops, eval_batches):
    tokens_per_step = batch_size * grad_accum * SEQ_LEN
    total_steps = max(1, total_tokens // tokens_per_step)
    total_micro = total_steps * grad_accum
    log_interval = max(1, total_steps // 20)

    # Re-init weights, build fresh optimizer/scheduler
    trainers = []
    for m in models_info:
        eager = getattr(m["model"], "_orig_mod", m["model"])
        initialize_model(eager, "xavier_uniform")
        opt = build_optimizer(m["model"], m["arch"]["dim"])
        sched = build_scheduler(opt, total_steps)
        trainers.append({
            "name": m["name"], "model_type": m["model_type"],
            "display": f"{m['model_type']}_{m['name']}",
            "model": m["model"], "opt": opt, "sched": sched, "ne": m["ne"],
            "needs_blocks": m["needs_blocks"], "arch": m["arch"],
            "step": 0, "micro": 0, "loss_sum": 0.0, "loss_n": 0,
            "train_curve": [], "eval_curve": [], "tokens_seen": 0,
        })

    for t in trainers:
        t["model"].train()
        t["opt"].zero_grad(set_to_none=True)

    t0 = time.time()

    for batch_idx, raw_batch in enumerate(train_loader):
        if batch_idx >= total_micro:
            break

        batch = {k: v.to(device, non_blocking=True)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in raw_batch.items()}

        for t in trainers:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model_forward(t["model"], batch, t["needs_blocks"])
                loss = out["loss"]
            (loss / grad_accum).backward()
            t["loss_sum"] += loss.detach().item()
            t["loss_n"] += 1
            t["micro"] += 1

            if t["micro"] % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(t["model"].parameters(), 1.0)
                t["opt"].step()
                t["sched"].step()
                t["opt"].zero_grad(set_to_none=True)
                t["step"] += 1
                t["tokens_seen"] += tokens_per_step
                avg_loss = t["loss_sum"] / t["loss_n"]
                t["train_curve"].append((t["step"], t["tokens_seen"], round(avg_loss, 4)))
                t["loss_sum"] = 0.0
                t["loss_n"] = 0

        current_step = trainers[0]["step"]
        is_step_boundary = trainers[0]["micro"] % grad_accum == 0

        # Periodic eval (~5 eval points during training for loss curves)
        eval_interval = max(1, total_steps // 5)
        if current_step > 0 and is_step_boundary and current_step % eval_interval == 0:
            for t in trainers:
                eval_loss, _ = eval_model(t["model"], eval_loader, device,
                                          eval_batches, t["needs_blocks"])
                t["eval_curve"].append((t["step"], t["tokens_seen"], round(eval_loss, 4)))

        # Log progress
        if current_step > 0 and current_step % log_interval == 0 and is_step_boundary:
            elapsed = time.time() - t0
            total_flops = sum(6 * t["ne"] * t["tokens_seen"] for t in trainers)
            mfu = total_flops / (elapsed * gpu_tflops * 1e12) * 100
            print(f"    step {current_step:>5}/{total_steps}  "
                  f"[{elapsed:6.1f}s]  MFU={mfu:.1f}%")
            for mt in MODEL_TYPES:
                mt_trainers = [t for t in trainers if t["model_type"] == mt]
                if mt_trainers:
                    losses = "  ".join(f'{t["name"]}={t["train_curve"][-1][2]:.3f}'
                                       for t in mt_trainers)
                    print(f"      {mt:>3}: {losses}")

    torch.cuda.synchronize()
    train_time = time.time() - t0
    total_flops = sum(6 * t["ne"] * t["tokens_seen"] for t in trainers)
    agg_mfu = total_flops / (train_time * gpu_tflops * 1e12) * 100
    print(f"    Done in {train_time:.1f}s  MFU={agg_mfu:.1f}%")

    # Eval + write per-run results
    scaling_dir = PROJECT_ROOT / "checkpoints" / "scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for t in trainers:
        avg_loss, ppl = eval_model(t["model"], eval_loader, device,
                                   eval_batches, t["needs_blocks"])
        print(f"      {t['display']:>12}: eval_loss={avg_loss:.4f}  ppl={ppl:.1f}")

        arch = t["arch"]
        total_params = sum(p.numel() for p in t["model"].parameters())

        # Add final eval to eval_curve
        final_eval = (t["step"], t["tokens_seen"], round(avg_loss, 4))
        eval_curve = t["eval_curve"]
        if not eval_curve or eval_curve[-1][0] != t["step"]:
            eval_curve.append(final_eval)

        run_result = {
            "final_eval_loss": round(avg_loss, 4),
            "final_eval_ppl": round(ppl, 2),
            "total_steps": t["step"],
            "tokens_seen": t["tokens_seen"],
            "total_params": total_params,
            "non_emb_params": t["ne"],
            "training_time_sec": round(train_time, 2),
            "model_type": t["model_type"],
            "hparams": {
                "model_cls": MODEL_TYPE_NAMES[t["model_type"]],
                "dim": arch["dim"],
                "num_encoder_layers": arch["num_encoder_layers"],
                "num_decoder_layers": arch["num_decoder_layers"],
                "num_heads": arch["dim"] // 64,
                "seq_len": SEQ_LEN,
                "mup_base_dim": MUP_BASE_DIM,
                "lr": BASE_LR,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum,
                "total_tokens": total_tokens,
            },
            # Curves: each entry is [step, tokens_seen, loss]
            "train_curve": t["train_curve"],
            "eval_curve": eval_curve,
        }

        # Write: checkpoints/scaling/{prefix}_{size}_{tokens}tok_results.json
        run_name = f"{t['model_type']}_{t['name']}_{tok_label}tok"
        run_path = scaling_dir / f"{run_name}_results.json"
        with open(run_path, "w") as f:
            json.dump(run_result, f, indent=2)

        results[t["display"]] = run_result
        results[t["display"]]["aggregate_mfu_pct"] = round(agg_mfu, 2)

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full scaling-law grid training")
    parser.add_argument("--token-budgets", type=str, default=None,
                        help="Comma-separated token budgets (e.g. '10M,50M')")
    parser.add_argument("--model-types", type=str, default=None,
                        help="Comma-separated model types (e.g. 'dd,dec'). "
                             "Default: all 3 (dd,sed,dec)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="checkpoints/parallel_scaling")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--train-file", type=str,
                        default="data/Pretrain/slimpajama_6b_packed.jsonl")
    parser.add_argument("--eval-file", type=str,
                        default="data/Pretrain/slimpajama_6b_eval_packed.jsonl")
    parser.add_argument("--tokenizer-file", type=str,
                        default="tokenizer/tokenizer_32k.json")
    args = parser.parse_args()

    # Parse subsets
    if args.token_budgets:
        requested = set(args.token_budgets.split(","))
        budgets = [(l, v) for l, v in TOKEN_BUDGETS if l in requested]
    else:
        budgets = TOKEN_BUDGETS

    if args.model_types:
        model_types = [mt.strip() for mt in args.model_types.split(",")]
    else:
        model_types = MODEL_TYPES

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    use_compile = not args.no_compile
    install_fast_masks()

    # 15 models × multiple inner functions need >8 cached compiled graphs
    torch._dynamo.config.cache_size_limit = 64

    gpu_tflops = detect_gpu_tflops()
    gpu_name = torch.cuda.get_device_name(0)
    tokens_per_step = args.batch_size * args.grad_accum * SEQ_LEN

    print(f"GPU: {gpu_name}  (peak BF16: {gpu_tflops:.0f} TFLOPS)")
    print(f"Effective batch: {args.batch_size} × {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} seqs = {tokens_per_step:,} tok/step")
    print(f"torch.compile: {'ON' if use_compile else 'OFF'}  |  matmul precision: TF32")
    print(f"Model types: {model_types}")
    print(f"Token budgets: {[l for l, _ in budgets]}")

    # ── Tokenizer + Data ────────────────────────────────────────────────────
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(PROJECT_ROOT / args.tokenizer_file))
    vocab_size = tokenizer.vocab_size
    bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    print(f"Tokenizer: vocab_size={vocab_size}")

    print("Loading data...")
    # BasicPretrainCollator produces blocks — works for all model types.
    # DecoderOnly models just ignore blocks in their forward().
    collator = BasicPretrainCollator(
        bos_token_id=bos_token_id, eos_token_id=eos_token_id, max_seq_len=SEQ_LEN)

    train_ds = load_dataset(
        "json", data_files=str(PROJECT_ROOT / args.train_file),
        split="train", streaming=True)
    eval_ds = load_dataset(
        "json", data_files=str(PROJECT_ROOT / args.eval_file),
        split="train")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              collate_fn=collator, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             collate_fn=collator, drop_last=True)

    # ── Run each model type sequentially ───────────────────────────────────
    # Build/compile 5 models per type, train all budgets, then tear down.
    # This keeps only 5 models in memory and in the compile cache at a time,
    # avoiding the cache thrashing that kills MFU with 15 interleaved models.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    sweep_t0 = time.time()

    for mt in model_types:
        needs_blocks = mt in ("dd", "sed")
        print(f"\n{'#'*70}")
        print(f"  Model type: {MODEL_TYPE_NAMES[mt]} ({mt})")
        print(f"{'#'*70}")

        # Build and compile 5 models for this type
        print(f"\n  Building 5 models:")
        models_info = []
        for name, arch in ARCHITECTURES:
            model = build_model(mt, arch, vocab_size, device, use_compile=use_compile)
            ne = non_emb_param_count(model)
            print(f"    {name:>6}: dim={arch['dim']:>3}  non_emb={ne:>10,}")
            models_info.append({
                "name": name, "model_type": mt, "arch": arch,
                "model": model, "ne": ne, "needs_blocks": needs_blocks,
            })

        # Compile warmup for this type's 5 models
        if use_compile:
            print(f"  Compiling...")
            compile_t0 = time.time()
            dummy_ids = torch.randint(0, vocab_size, (args.batch_size, SEQ_LEN), device=device)
            dummy_blocks = torch.sort(torch.randperm(SEQ_LEN - 2, device=device)[:4] + 1)[0]
            dummy_batch = {"input_ids": dummy_ids, "labels": dummy_ids.clone(),
                           "blocks": dummy_blocks, "sft": False}
            for m in models_info:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model_forward(m["model"], dummy_batch, m["needs_blocks"])
                out["loss"].backward()
                for p in m["model"].parameters():
                    if p.grad is not None:
                        p.grad = None
            torch.cuda.synchronize()
            print(f"  Compiled in {time.time() - compile_t0:.1f}s")

        # Train all token budgets for this model type
        for tok_label, total_tokens in budgets:
            total_steps = max(1, total_tokens // tokens_per_step)
            total_micro = total_steps * args.grad_accum

            print(f"\n  {'='*66}")
            print(f"    {mt} × {tok_label} tokens  |  {total_steps} steps  |  "
                  f"{total_micro} micro-batches")
            print(f"  {'='*66}\n")

            results = train_one_budget(
                models_info, tok_label, total_tokens, args.batch_size, args.grad_accum,
                train_loader, eval_loader, device, gpu_tflops, args.eval_batches)

            # Merge into all_results[tok_label]
            if tok_label not in all_results:
                all_results[tok_label] = {}
            all_results[tok_label].update(results)

            # Save per-budget results (incrementally)
            out_path = out_dir / f"parallel_{tok_label}tok_results.json"
            with open(out_path, "w") as f:
                json.dump(all_results[tok_label], f, indent=2)

        # Free GPU memory before next model type
        del models_info
        torch.cuda.empty_cache()

    # ── Summary ─────────────────────────────────────────────────────────────
    sweep_time = time.time() - sweep_t0
    combined_path = out_dir / "scaling_grid_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Full grid complete in {sweep_time/60:.1f} min")
    print(f"  Results → {combined_path}")
    print(f"{'='*70}")

    # Print loss grids per model type
    for mt in model_types:
        print(f"\n  Eval loss grid — {MODEL_TYPE_NAMES[mt]} ({mt}):\n")
        header = f"  {'params':<8}" + "".join(f"  {l:>8}" for l, _ in budgets)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, _ in ARCHITECTURES:
            row = f"  {name:<8}"
            display = f"{mt}_{name}"
            for tok_label, _ in budgets:
                if tok_label in all_results and display in all_results[tok_label]:
                    loss = all_results[tok_label][display]["final_eval_loss"]
                    row += f"  {loss:>8.4f}"
                else:
                    row += f"  {'—':>8}"
            print(row)

    print(f"\nTo collect in scaling_laws.py format:")
    print(f"  python scripts/scaling_laws.py collect")


if __name__ == "__main__":
    main()
