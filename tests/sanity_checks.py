#!/usr/bin/env python
"""Preflight sanity checks for 1B Double Decoder training.

Usage:
    python tests/sanity_checks.py --check param_count
    python tests/sanity_checks.py --check all
    torchrun --nproc_per_node=2 tests/sanity_checks.py --check ddp_smoke
"""

import sys
import os
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


# ── Utilities ────────────────────────────────────────────────────────────────

def _print_result(name, passed, detail=""):
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def _build_1b_model(device="cpu"):
    from models.double_decoder import Double_Decoder
    model = Double_Decoder(
        vocab_size=32768,
        dim=1536,
        num_heads=24,
        num_encoder_layers=24,
        num_decoder_layers=12,
        seq_len=2048,
        shared=True,
        logit_biases=False,
        gradient_checkpointing=True,
    )
    return model.to(device)


def _random_batch(batch_size=2, seq_len=2048, vocab_size=32768, device="cpu"):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    blocks = torch.sort(torch.randperm(seq_len - 2)[:5] + 1)[0].to(device)
    return {"input_ids": input_ids, "labels": labels, "blocks": blocks, "sft": False}


# ── Check 1: Parameter Count ────────────────────────────────────────────────

def check_param_count():
    print("\n── Check 1: Parameter Count ──")
    model = _build_1b_model()

    total = sum(p.numel() for p in model.parameters())
    unique = sum(p.numel() for p in {id(p): p for id_, p in ((id(p), p) for p in model.parameters())}.values())

    # Per-component breakdown
    embed_params = model.embedding.weight.numel()
    enc_params = sum(p.numel() for layer in model.encoder_layers for p in layer.parameters())
    dec_params = sum(p.numel() for layer in model.decoder_layers for p in layer.parameters())
    norm_params = sum(p.numel() for p in model.encoder_norm.parameters()) + \
                  sum(p.numel() for p in model.output_norm.parameters())

    print(f"  Embedding:      {embed_params:>12,}")
    print(f"  Encoder (24L):  {enc_params:>12,}")
    print(f"  Decoder (12L):  {dec_params:>12,}")
    print(f"  Norms:          {norm_params:>12,}")
    print(f"  Total (raw):    {total:>12,}")
    print(f"  Total (unique): {unique:>12,}")

    ok = True
    ok &= _print_result("Total params ~1B", 0.9e9 < unique < 1.2e9, f"{unique / 1e9:.3f}B")
    ok &= _print_result("Tied weights", model.output_projection.weight is model.embedding.weight)
    ok &= _print_result("dim=1536", model.dim == 1536)
    ok &= _print_result("24 encoder layers", len(model.encoder_layers) == 24)
    ok &= _print_result("12 decoder layers", len(model.decoder_layers) == 12)
    return ok


# ── Check 2: Forward Pass ───────────────────────────────────────────────────

def check_forward_pass():
    print("\n── Check 2: Forward Pass ──")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_1b_model(device)
    batch = _random_batch(batch_size=2, device=device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(**batch)

    loss = out["loss"]
    logits = out["logits"]
    expected_loss = math.log(32768)  # ~10.4 for random weights

    ok = True
    ok &= _print_result("Output has loss", loss is not None)
    ok &= _print_result("Output has logits", logits is not None)
    ok &= _print_result("Logits shape", logits.shape == (2, 2048, 32768), str(logits.shape))
    ok &= _print_result("Loss is scalar", loss.dim() == 0)
    ok &= _print_result("Loss is finite", loss.isfinite().item())
    ok &= _print_result(f"Loss ≈ ln(32768)={expected_loss:.1f}",
                        abs(loss.item() - expected_loss) < expected_loss * 0.15,
                        f"got {loss.item():.2f}")
    return ok


# ── Check 3: Gradient Flow ──────────────────────────────────────────────────

def check_gradient_flow():
    print("\n── Check 3: Gradient Flow ──")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_1b_model(device)
    batch = _random_batch(batch_size=2, device=device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(**batch)
    out["loss"].backward()

    ok = True
    no_grad = []
    nan_grad = []
    zero_grad = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(name)
        elif not p.grad.isfinite().all():
            nan_grad.append(name)
        elif p.grad.norm().item() == 0:
            zero_grad.append(name)

    ok &= _print_result("All params have gradients", len(no_grad) == 0,
                        f"{len(no_grad)} missing" if no_grad else "")
    ok &= _print_result("No NaN/inf gradients", len(nan_grad) == 0,
                        f"{len(nan_grad)} bad" if nan_grad else "")
    ok &= _print_result("No zero gradients", len(zero_grad) == 0,
                        f"{len(zero_grad)} zero" if zero_grad else "")

    if no_grad:
        print(f"    Missing grads: {no_grad[:5]}")
    if nan_grad:
        print(f"    NaN grads: {nan_grad[:5]}")

    return ok


# ── Check 4: Micro Training ─────────────────────────────────────────────────

def check_micro_train(steps=100):
    print(f"\n── Check 4: Micro Training ({steps} steps) ──")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_1b_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(device == "cuda"))
    model.train()

    losses = []
    for i in range(steps):
        batch = _random_batch(batch_size=2, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(**batch)
        loss = out["loss"]
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (i + 1) % 25 == 0:
            print(f"    Step {i+1}: loss={losses[-1]:.4f}")

    first_loss = np.mean(losses[:5])
    last_loss = np.mean(losses[-5:])
    reduction = (first_loss - last_loss) / first_loss

    ok = True
    ok &= _print_result("No NaN losses", all(math.isfinite(l) for l in losses))
    ok &= _print_result(f"Loss decreased ≥20%", reduction >= 0.20,
                        f"{reduction*100:.1f}% reduction ({first_loss:.2f} → {last_loss:.2f})")
    return ok


# ── Check 5: Memory Profile ─────────────────────────────────────────────────

def check_memory_profile():
    print("\n── Check 5: Memory Profile ──")
    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available")
        return True

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    model = _build_1b_model(device)
    batch = _random_batch(batch_size=16, device=device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(**batch)
    out["loss"].backward()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"  Peak memory: {peak_gb:.1f} GB / {total_gb:.1f} GB available")
    print(f"  Headroom: {total_gb - peak_gb:.1f} GB")

    ok = _print_result("Fits in GPU memory", peak_gb < total_gb * 0.95,
                       f"{peak_gb:.1f}GB used of {total_gb:.1f}GB")
    del model, batch, out
    torch.cuda.empty_cache()
    return ok


# ── Check 6: Data Pipeline ──────────────────────────────────────────────────

def check_data_pipeline():
    print("\n── Check 6: Data Pipeline ──")
    from datasets import load_dataset

    # Try 1B data files first, fall back to original
    for prefix in ["slimpajama_20b", "slimpajama"]:
        train_file = f"data/Pretrain/{prefix}_packed.jsonl"
        if os.path.exists(train_file):
            break
    else:
        print("  [SKIP] No packed data files found")
        return True

    print(f"  Loading from {train_file}")
    ds = load_dataset("json", data_files=train_file, split="train", streaming=True)

    ok = True
    n_checked = 0
    bad_len = 0
    bad_range = 0
    vocab_size = 32768 if "20b" in train_file else 8192

    for example in ds:
        ids = example["input_ids"]
        if len(ids) != 2048:
            bad_len += 1
        if any(t < 0 or t >= vocab_size for t in ids):
            bad_range += 1
        n_checked += 1
        if n_checked >= 1000:
            break

    ok &= _print_result(f"Checked {n_checked} sequences", n_checked >= 100)
    ok &= _print_result("All sequences length 2048", bad_len == 0, f"{bad_len} bad")
    ok &= _print_result(f"All token IDs in [0, {vocab_size})", bad_range == 0, f"{bad_range} bad")
    return ok


# ── Check 7: Tokenizer ──────────────────────────────────────────────────────

def check_tokenizer():
    print("\n── Check 7: Tokenizer ──")
    from transformers import PreTrainedTokenizerFast

    # Try 32K first, fall back to 8K
    for path in ["tokenizer/tokenizer_32k.json", "tokenizer/tokenizer.json"]:
        if os.path.exists(path):
            break
    else:
        print("  [SKIP] No tokenizer found")
        return True

    print(f"  Loading {path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    expected_vocab = 32768 if "32k" in path else 8192

    ok = True
    ok &= _print_result(f"Vocab size = {expected_vocab}", tokenizer.vocab_size == expected_vocab,
                        f"got {tokenizer.vocab_size}")

    # Round-trip test
    test_strings = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "import torch\nprint(torch.cuda.is_available())",
    ]
    roundtrip_ok = True
    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if decoded.strip() != s.strip():
            roundtrip_ok = False
            print(f"    Roundtrip fail: '{s}' → '{decoded}'")

    ok &= _print_result("Round-trip encode/decode", roundtrip_ok)

    # Chars per token
    long_text = " ".join(test_strings) * 10
    ids = tokenizer.encode(long_text)
    cpt = len(long_text) / max(1, len(ids))
    ok &= _print_result("Chars/token in [2.5, 6.0]", 2.5 <= cpt <= 6.0, f"{cpt:.2f}")

    # Special tokens
    bos = tokenizer.convert_tokens_to_ids("<s>")
    eos = tokenizer.convert_tokens_to_ids("</s>")
    ok &= _print_result("BOS token exists", bos is not None and bos >= 0, f"id={bos}")
    ok &= _print_result("EOS token exists", eos is not None and eos >= 0, f"id={eos}")

    return ok


# ── Check 8: Block Masks ────────────────────────────────────────────────────

def check_block_masks():
    print("\n── Check 8: Block Masks ──")
    from components.block_masks import create_masks

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    seq_len = 64  # small for speed
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    blocks = torch.tensor([10, 25, 40], device=device)

    masks = create_masks(
        batch_size=batch_size, blocks=blocks, device=device,
        input_ids=input_ids, encoder_input_ids=input_ids,
        decoder_input_ids=input_ids, sft=False
    )

    ok = True
    ok &= _print_result("Masks returned", masks is not None)
    ok &= _print_result("Has self_mask", "self_mask" in masks)
    ok &= _print_result("Has cross_mask", "cross_mask" in masks)
    return ok


# ── Check 9: ComboAttention Stability ───────────────────────────────────────

def check_combo_attn_stability():
    print("\n── Check 9: ComboAttention Stability (dim=1536, bf16) ──")
    from components.attention import ComboAttention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = ComboAttention(dim=1536, num_heads=24, seq_len=2048, shared=True).to(device)

    x = torch.randn(2, 128, 1536, device=device)  # short seq for speed
    enc = torch.randn(2, 128, 1536, device=device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device == "cuda"):
        # No block masks → uses flash_attn path
        out = attn(x, enc, block_masks=None)

    ok = True
    ok &= _print_result("Output shape correct", out.shape == x.shape, str(out.shape))
    ok &= _print_result("No NaN in output", not torch.isnan(out).any().item())
    ok &= _print_result("No Inf in output", not torch.isinf(out).any().item())
    return ok


# ── Check 10: Config Validation ─────────────────────────────────────────────

def check_config_validation():
    print("\n── Check 10: Config Validation ──")
    from omegaconf import OmegaConf
    from configs import build_config_from_dict

    config_path = "configs/runs/pretrain_1b.yaml"
    if not os.path.exists(config_path):
        print("  [SKIP] pretrain_1b.yaml not found")
        return True

    cfg_dict = OmegaConf.load(config_path)
    cfg = build_config_from_dict(cfg_dict)

    ok = True
    ok &= _print_result("dim % 64 == 0", cfg.dim % 64 == 0, f"dim={cfg.dim}")
    ok &= _print_result("enc = 2 * dec layers", cfg.num_encoder_layers == 2 * cfg.num_decoder_layers)
    ok &= _print_result("total_tokens > 0", cfg.total_tokens > 0, f"{cfg.total_tokens:,}")
    ok &= _print_result("gradient_checkpointing enabled", cfg.gradient_checkpointing)
    ok &= _print_result("wandb_entity set", bool(cfg.wandb_entity), cfg.wandb_entity)
    ok &= _print_result("wandb_project set", bool(cfg.wandb_project), cfg.wandb_project)

    steps = cfg.total_tokens // (cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len)
    ok &= _print_result("Enough training steps (>1000)", steps > 1000, f"{steps:,} steps")

    return ok


# ── Check 11: Compile Compatibility ─────────────────────────────────────────

def check_compile_compat():
    print("\n── Check 11: Compile Compatibility ──")
    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available (torch.compile needs CUDA)")
        return True

    device = "cuda"
    model = _build_1b_model(device)
    batch = _random_batch(batch_size=2, device=device)

    # Eager forward
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        eager_out = model(**batch)
    eager_loss = eager_out["loss"].item()

    # Compiled forward
    compiled = torch.compile(model, fullgraph=False, dynamic=False)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        compiled_out = compiled(**batch)
    compiled_loss = compiled_out["loss"].item()

    ok = True
    ok &= _print_result("Compiled model runs", True)
    ok &= _print_result("Loss matches eager (within tolerance)",
                        abs(eager_loss - compiled_loss) < 0.1,
                        f"eager={eager_loss:.4f}, compiled={compiled_loss:.4f}")

    del model, compiled
    torch.cuda.empty_cache()
    return ok


# ── Check 12: DDP Smoke Test ────────────────────────────────────────────────

def check_ddp_smoke():
    print("\n── Check 12: DDP Smoke Test ──")
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_available() or not dist.is_initialized():
        # Try to init
        if "RANK" in os.environ:
            dist.init_process_group(backend='nccl')
        else:
            print("  [SKIP] Not launched with torchrun")
            return True

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"  Running DDP with {world_size} GPUs")

    model = _build_1b_model(device)
    model = DDP(model, device_ids=[device.index])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()

    losses = []
    for i in range(10):
        batch = _random_batch(batch_size=2, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(**batch)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(out["loss"].item())

    # Verify losses are similar across ranks
    loss_tensor = torch.tensor(losses[-1], device=device)
    all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_tensor)

    ok = True
    if rank == 0:
        loss_vals = [l.item() for l in all_losses]
        max_diff = max(loss_vals) - min(loss_vals)
        ok &= _print_result("DDP ran 10 steps", True)
        ok &= _print_result("Loss similar across ranks", max_diff < 1.0,
                           f"losses={[f'{l:.2f}' for l in loss_vals]}")
        ok &= _print_result("No NaN losses", all(math.isfinite(l) for l in losses))

    dist.destroy_process_group()
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_CHECKS = {
    "param_count": check_param_count,
    "forward_pass": check_forward_pass,
    "gradient_flow": check_gradient_flow,
    "micro_train": check_micro_train,
    "memory_profile": check_memory_profile,
    "data_pipeline": check_data_pipeline,
    "tokenizer": check_tokenizer,
    "block_masks": check_block_masks,
    "combo_attn_stability": check_combo_attn_stability,
    "config_validation": check_config_validation,
    "compile_compat": check_compile_compat,
    "ddp_smoke": check_ddp_smoke,
}

# Checks that don't require GPU
CPU_SAFE_CHECKS = {"param_count", "data_pipeline", "tokenizer", "config_validation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=str, default="all",
                        help=f"Check to run: {', '.join(ALL_CHECKS.keys())}, or 'all'")
    parser.add_argument("--steps", type=int, default=100,
                        help="Steps for micro_train check (default: 100)")
    args = parser.parse_args()

    if args.check == "all":
        # Run all except ddp_smoke (needs torchrun)
        checks = [k for k in ALL_CHECKS if k != "ddp_smoke"]
    else:
        checks = [args.check]

    print("=" * 60)
    print("  PREFLIGHT SANITY CHECKS — 1B Double Decoder")
    print("=" * 60)

    results = {}
    for name in checks:
        if name not in ALL_CHECKS:
            print(f"\nUnknown check: {name}")
            print(f"Available: {', '.join(ALL_CHECKS.keys())}")
            sys.exit(1)
        try:
            if name == "micro_train":
                passed = ALL_CHECKS[name](steps=args.steps)
            else:
                passed = ALL_CHECKS[name]()
        except Exception as e:
            print(f"\n  [FAIL ✗] {name} — Exception: {e}")
            import traceback
            traceback.print_exc()
            passed = False
        results[name] = passed

    # Summary
    print("\n" + "=" * 60)
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"  Results: {n_pass}/{n_total} checks passed")
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"    [{status}] {name}")
    print("=" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
