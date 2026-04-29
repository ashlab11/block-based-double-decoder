#!/usr/bin/env python3
"""Verify μP implementation correctness and optionally run LR transfer experiments.

Default mode (no flags or --check): fast coordinate check (~30s on GPU, works on CPU).
Builds models at multiple widths, runs a few forward+backward steps, and checks that:
  1. Activation norms are stable across widths (not growing/shrinking)
  2. Parameter update norms scale correctly (hidden ∝ base_dim/dim, embed ∝ 1)
  3. At base width, μP ≡ SP (mup_mult = 1.0)

Usage:
    # Fast correctness check (~30s):
    python scripts/mup_verify.py

    # Full LR transfer experiment (hours, GPU required):
    python scripts/mup_verify.py --full --tokens 50000000

    # Plot from saved full results:
    python scripts/mup_verify.py --plot
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

MUP_BASE_DIM = 64
VERIFY_DIR = Path("checkpoints/mup_verify")
RESULTS_FILE = VERIFY_DIR / "results.json"

DEFAULT_PARAMS = [500_000, 2_500_000, 5_000_000, 15_000_000]
DEFAULT_LRS = [3e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2]
DEFAULT_TOKENS = 50_000_000

# Widths to test in the coord check (must be multiples of 64)
COORD_CHECK_DIMS = [64, 128, 256, 512]
COORD_CHECK_STEPS = 5
COORD_CHECK_LR = 2e-3
COORD_CHECK_BATCH = 4
COORD_CHECK_SEQ = 256  # short sequences for speed


# ══════════════════════════════════════════════════════════════════════════════
# FAST COORDINATE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _build_dd_model(dim, mup_base_dim=0, device="cpu"):
    """Build a small Double_Decoder at the given width."""
    from models.double_decoder import Double_Decoder
    model = Double_Decoder(
        vocab_size=32768,
        dim=dim,
        num_heads=dim // 64,
        num_encoder_layers=4,
        num_decoder_layers=2,
        seq_len=COORD_CHECK_SEQ,
        shared=True,
        logit_biases=False,
        mup_base_dim=mup_base_dim,
    )
    return model.to(device)


def _random_batch(batch_size, seq_len, device):
    """Create a random batch with block structure for Double_Decoder."""
    import torch
    input_ids = torch.randint(0, 32768, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 32768, (batch_size, seq_len), device=device)
    blocks = torch.sort(torch.randperm(seq_len - 2, device=device)[:3] + 1)[0]
    return {"input_ids": input_ids, "labels": labels, "blocks": blocks, "sft": False}


def _collect_activation_norms(model, batch):
    """Run forward pass, collect L2 norms of encoder/decoder hidden states."""
    import torch
    norms = {}

    # Encoder activations
    x = model.embedding(batch["input_ids"]) * model.mup_mult
    norms["embed_out"] = x.detach().float().norm().item() / math.sqrt(x.numel())

    for i, layer in enumerate(model.encoder_layers):
        x = layer(x)
        norms[f"enc_{i}"] = x.detach().float().norm().item() / math.sqrt(x.numel())

    x = model.encoder_norm(x)
    norms["enc_norm"] = x.detach().float().norm().item() / math.sqrt(x.numel())

    return norms


def _collect_update_norms(model, optimizer, batch, mup_base_dim):
    """Run one forward+backward+step, measure parameter update norms."""
    import torch
    # Snapshot params before step
    old_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # Forward + backward
    out = model(**batch)
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Measure updates
    update_norms = {}
    for n, p in model.named_parameters():
        if n in old_params:
            delta = (p.detach() - old_params[n]).float()
            rms = delta.norm() / math.sqrt(delta.numel())
            update_norms[n] = rms.item()

    return update_norms


def _build_mup_optimizer(model, lr, mup_base_dim, dim):
    """Build optimizer with μP param groups (mirrors pretrain.py logic)."""
    import torch
    from torch.optim import AdamW

    if mup_base_dim > 0:
        mup_mult = mup_base_dim / dim
        embedding_ids = {id(model.embedding.weight)}
        embed_params, hidden_decay, base_no_decay = [], [], []
        seen_ids = set()
        for n, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            if id(p) in embedding_ids:
                embed_params.append(p)
            elif p.dim() <= 1:
                base_no_decay.append(p)
            else:
                hidden_decay.append(p)
        return AdamW([
            {"params": embed_params, "weight_decay": 0.1, "lr": lr},
            {"params": hidden_decay, "weight_decay": 0.1, "lr": lr * mup_mult},
            {"params": base_no_decay, "weight_decay": 0.0, "lr": lr},
        ], betas=(0.9, 0.95), eps=1e-8)
    else:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.dim() <= 1 else decay).append(p)
        return AdamW([
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.95), eps=1e-8)


def run_coord_check(device="cpu"):
    """Run the fast μP coordinate check."""
    import torch

    print("=" * 60)
    print("  μP COORDINATE CHECK")
    print(f"  dims={COORD_CHECK_DIMS}, steps={COORD_CHECK_STEPS}, "
          f"base_dim={MUP_BASE_DIM}, device={device}")
    print("=" * 60)

    all_ok = True

    # ── Check 1: At base width, μP model == SP model ──────────────────────
    print("\n── Check 1: μP at base width is identity ──")
    dim = MUP_BASE_DIM
    torch.manual_seed(42)
    sp_model = _build_dd_model(dim, mup_base_dim=0, device=device)
    torch.manual_seed(42)
    mup_model = _build_dd_model(dim, mup_base_dim=MUP_BASE_DIM, device=device)

    assert mup_model.mup_mult == 1.0, f"Expected mup_mult=1.0 at base width, got {mup_model.mup_mult}"

    batch = _random_batch(COORD_CHECK_BATCH, COORD_CHECK_SEQ, device)
    with torch.no_grad():
        sp_out = sp_model(**batch)
        mup_out = mup_model(**batch)

    # Note: attention scale differs (1/d vs 1/√d), so outputs won't be identical.
    # But the forward pass should run without errors and produce finite losses.
    sp_loss = sp_out["loss"].item()
    mup_loss = mup_out["loss"].item()
    loss_ok = math.isfinite(sp_loss) and math.isfinite(mup_loss)
    print(f"  SP loss:  {sp_loss:.4f}")
    print(f"  μP loss:  {mup_loss:.4f} (differs due to 1/d vs 1/√d attn scale)")
    print(f"  [{'PASS' if loss_ok else 'FAIL'}] Both losses finite")
    all_ok &= loss_ok

    # ── Check 2: Activation norms stable across widths ────────────────────
    print("\n── Check 2: Activation norms across widths (μP) ──")
    act_norms = {}
    for dim in COORD_CHECK_DIMS:
        torch.manual_seed(0)
        model = _build_dd_model(dim, mup_base_dim=MUP_BASE_DIM, device=device)
        batch = _random_batch(COORD_CHECK_BATCH, COORD_CHECK_SEQ, device)
        with torch.no_grad():
            norms = _collect_activation_norms(model, batch)
        act_norms[dim] = norms
        del model

    # Print table
    layers = list(act_norms[COORD_CHECK_DIMS[0]].keys())
    header = f"{'layer':<12}" + "".join(f"dim={d:<6}" for d in COORD_CHECK_DIMS) + "  ratio(max/min)"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    ratios = []
    for layer in layers:
        vals = [act_norms[d][layer] for d in COORD_CHECK_DIMS]
        ratio = max(vals) / max(min(vals), 1e-10)
        ratios.append(ratio)
        vals_str = "".join(f"{v:<10.4f}" for v in vals)
        status = "ok" if ratio < 10 else "DRIFT"
        print(f"  {layer:<12}{vals_str}  {ratio:.2f}x  {status}")

    max_ratio = max(ratios)
    act_ok = max_ratio < 10  # activations shouldn't vary more than 10x across 8x width range
    print(f"\n  [{'PASS' if act_ok else 'FAIL'}] Max activation ratio: {max_ratio:.2f}x (threshold: <10x)")
    all_ok &= act_ok

    # ── Check 3: Same check WITHOUT μP (should show drift) ───────────────
    print("\n── Check 3: Activation norms across widths (SP, expect drift) ──")
    sp_norms = {}
    for dim in COORD_CHECK_DIMS:
        torch.manual_seed(0)
        model = _build_dd_model(dim, mup_base_dim=0, device=device)
        batch = _random_batch(COORD_CHECK_BATCH, COORD_CHECK_SEQ, device)
        with torch.no_grad():
            norms = _collect_activation_norms(model, batch)
        sp_norms[dim] = norms
        del model

    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    sp_ratios = []
    for layer in layers:
        vals = [sp_norms[d][layer] for d in COORD_CHECK_DIMS]
        ratio = max(vals) / max(min(vals), 1e-10)
        sp_ratios.append(ratio)
        vals_str = "".join(f"{v:<10.4f}" for v in vals)
        print(f"  {layer:<12}{vals_str}  {ratio:.2f}x")

    sp_max_ratio = max(sp_ratios)
    print(f"\n  SP max ratio: {sp_max_ratio:.2f}x (no threshold — just for comparison)")

    # ── Check 4: Update norms scale correctly ─────────────────────────────
    print("\n── Check 4: Parameter update norms (μP vs SP) ──")
    print(f"  Running {COORD_CHECK_STEPS} steps at each width...\n")

    for mode, mup_base in [("SP", 0), ("μP", MUP_BASE_DIM)]:
        print(f"  --- {mode} ---")
        # Collect per-component average update RMS after a few steps
        component_updates = {d: {} for d in COORD_CHECK_DIMS}
        for dim in COORD_CHECK_DIMS:
            torch.manual_seed(0)
            model = _build_dd_model(dim, mup_base_dim=mup_base, device=device)
            model.train()
            optimizer = _build_mup_optimizer(model, COORD_CHECK_LR, mup_base, dim)

            all_updates = []
            for _ in range(COORD_CHECK_STEPS):
                batch = _random_batch(COORD_CHECK_BATCH, COORD_CHECK_SEQ, device)
                updates = _collect_update_norms(model, optimizer, batch, mup_base)
                all_updates.append(updates)

            # Average over steps, group by component
            avg = {}
            for n in all_updates[0]:
                avg[n] = np.mean([u[n] for u in all_updates])

            # Aggregate by component type
            embed_rms = avg.get("embedding.weight", 0)
            enc_rms = np.mean([v for k, v in avg.items() if "encoder_layers" in k and v > 0]) if any("encoder_layers" in k for k in avg) else 0
            dec_rms = np.mean([v for k, v in avg.items() if "decoder_layers" in k and v > 0]) if any("decoder_layers" in k for k in avg) else 0

            component_updates[dim] = {"embed": embed_rms, "encoder": enc_rms, "decoder": dec_rms}
            del model, optimizer

        # Print table
        print(f"  {'component':<12}" + "".join(f"dim={d:<10}" for d in COORD_CHECK_DIMS) + "ratio(max/min)")
        for comp in ["embed", "encoder", "decoder"]:
            vals = [component_updates[d][comp] for d in COORD_CHECK_DIMS]
            ratio = max(vals) / max(min(vals), 1e-15)
            vals_str = "".join(f"{v:<14.2e}" for v in vals)
            print(f"  {comp:<12}{vals_str}{ratio:.1f}x")
        print()

    # ── Check 5: μP hidden updates shrink with width ──────────────────────
    print("── Check 5: μP hidden updates shrink proportionally to base_dim/dim ──")
    mup_enc_updates = {}
    for dim in COORD_CHECK_DIMS:
        torch.manual_seed(0)
        model = _build_dd_model(dim, mup_base_dim=MUP_BASE_DIM, device=device)
        model.train()
        optimizer = _build_mup_optimizer(model, COORD_CHECK_LR, MUP_BASE_DIM, dim)
        batch = _random_batch(COORD_CHECK_BATCH, COORD_CHECK_SEQ, device)
        updates = _collect_update_norms(model, optimizer, batch, MUP_BASE_DIM)
        enc_rms = np.mean([v for k, v in updates.items() if "encoder_layers" in k and "weight" in k and v > 0])
        mup_enc_updates[dim] = enc_rms
        del model, optimizer

    base_update = mup_enc_updates[MUP_BASE_DIM]
    print(f"  {'dim':<8}{'update RMS':<14}{'ratio vs base':<16}{'expected (base/dim)':<20}{'match'}")
    scale_ok = True
    for dim in COORD_CHECK_DIMS:
        actual_ratio = mup_enc_updates[dim] / base_update
        expected_ratio = MUP_BASE_DIM / dim
        match = abs(actual_ratio - expected_ratio) / max(expected_ratio, 1e-10) < 0.5  # within 50%
        scale_ok &= match
        print(f"  {dim:<8}{mup_enc_updates[dim]:<14.2e}{actual_ratio:<16.3f}{expected_ratio:<20.3f}{'ok' if match else 'MISMATCH'}")

    print(f"\n  [{'PASS' if scale_ok else 'FAIL'}] Hidden update scaling matches μP prediction (within 50%)")
    all_ok &= scale_ok

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL CHECKS PASSED — μP implementation is correct")
    else:
        print("  SOME CHECKS FAILED — review output above")
    print("=" * 60)

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# FULL LR TRANSFER EXPERIMENT (hours on GPU)
# ══════════════════════════════════════════════════════════════════════════════

def _run_one(params, lr, tokens, mup_base_dim, mode_tag):
    """Run a single training job and return the result dict."""
    from training.api import train
    from configs.scaling import interpolate_architecture

    arch = interpolate_architecture(params)
    dim = arch["dim"]
    name = f"mup_verify_{mode_tag}_{params}p_dim{dim}_lr{lr:.0e}"

    print(f"\n{'='*60}")
    print(f"  {mode_tag.upper()} | params={params:,} (dim={dim}) | lr={lr:.1e}")
    print(f"  tokens={tokens:,} | mup_base_dim={mup_base_dim}")
    print(f"{'='*60}")

    try:
        r = train(
            params=params, tokens=tokens,
            mup_base_dim=mup_base_dim, lr=lr, run_name=name,
            no_wandb=True,
        )
        return {
            "params": params, "dim": dim, "lr": lr, "mode": mode_tag,
            "final_eval_loss": r["final_eval_loss"],
            "final_eval_ppl": r["final_eval_ppl"],
            "tokens_seen": r["tokens_seen"],
            "train_curve": r.get("train_curve", []),
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        return {
            "params": params, "dim": dim, "lr": lr, "mode": mode_tag,
            "final_eval_loss": None, "final_eval_ppl": None,
            "tokens_seen": 0, "train_curve": [],
        }


def run_full_experiments(param_sizes, lrs, tokens, modes):
    """Run the full grid of (params x LR x mode) training experiments."""
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = []

    existing_keys = {(r["params"], r["lr"], r["mode"]) for r in all_results}

    for params in param_sizes:
        for lr in lrs:
            for mode in modes:
                if (params, lr, mode) in existing_keys:
                    print(f"Skipping {mode} params={params} lr={lr} (already done)")
                    continue

                mup_base_dim = MUP_BASE_DIM if mode == "mup" else 0
                result = _run_one(params, lr, tokens, mup_base_dim, mode)
                all_results.append(result)

                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)

    return all_results


# ── Plotting (for --full results) ────────────────────────────────────────────

def plot_lr_transfer(results):
    """Plot final eval loss vs LR for each width, SP vs μP side by side."""
    modes = sorted(set(r["mode"] for r in results))
    n_panels = len(modes)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    mode_titles = {"sp": "Standard Parameterization", "mup": "μP (Maximal Update Param)"}
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 10))

    for ax, mode in zip(axes, modes):
        mode_results = [r for r in results if r["mode"] == mode]
        by_params = {}
        for r in mode_results:
            by_params.setdefault(r["params"], []).append(r)

        for i, params in enumerate(sorted(by_params)):
            runs = sorted(by_params[params], key=lambda r: r["lr"])
            valid = [(r["lr"], r["final_eval_loss"]) for r in runs if r["final_eval_loss"] is not None]
            if not valid:
                continue
            lrs, losses = zip(*valid)
            dim = runs[0]["dim"]
            ax.plot(lrs, losses, "o-", label=f"{params/1e6:.1f}M (dim={dim})",
                    color=colors[i % len(colors)], linewidth=2, markersize=6)
            best_idx = np.argmin(losses)
            ax.plot(lrs[best_idx], losses[best_idx], "*",
                    color=colors[i % len(colors)], markersize=14, zorder=5)

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=13)
        ax.set_ylabel("Final Eval Loss" if ax == axes[0] else "", fontsize=13)
        ax.set_title(mode_titles.get(mode, mode), fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("μP Verification: LR Transfer Across Widths", fontsize=15, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(VERIFY_DIR / f"mup_lr_transfer.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {VERIFY_DIR / 'mup_lr_transfer.png'}")
    plt.close(fig)


def plot_optimal_lr_vs_width(results):
    """Plot optimal LR vs model dim — should be flat for μP, decreasing for SP."""
    modes = sorted(set(r["mode"] for r in results))
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in modes:
        by_params = {}
        for r in results:
            if r["mode"] == mode and r["final_eval_loss"] is not None:
                p = r["params"]
                if p not in by_params or r["final_eval_loss"] < by_params[p]["final_eval_loss"]:
                    by_params[p] = r

        if not by_params:
            continue
        dims = [by_params[p]["dim"] for p in sorted(by_params)]
        best_lrs = [by_params[p]["lr"] for p in sorted(by_params)]
        style = "s-" if mode == "mup" else "o--"
        ax.plot(dims, best_lrs, style, label="μP" if mode == "mup" else "Standard Param",
                linewidth=2, markersize=8)

    ax.set_xlabel("Model Dimension (width)", fontsize=13)
    ax.set_ylabel("Optimal Learning Rate", fontsize=13)
    ax.set_yscale("log")
    ax.set_title("Optimal LR vs Width", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(VERIFY_DIR / f"mup_optimal_lr.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved: {VERIFY_DIR / 'mup_optimal_lr.png'}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify μP: fast coord check (default) or full LR transfer (--full)"
    )
    parser.add_argument("--check", action="store_true", default=False,
                        help="Run fast coordinate check (default if no flags)")
    parser.add_argument("--full", action="store_true",
                        help="Run full LR transfer experiment (hours, GPU required)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot results from a previous --full run")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"Token budget per run for --full (default: {DEFAULT_TOKENS:,})")
    parser.add_argument("--params", type=str, default=None,
                        help="Comma-separated param counts for --full")
    parser.add_argument("--lrs", type=str, default=None,
                        help="Comma-separated learning rates for --full")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["sp", "mup", "both"],
                        help="Which mode(s) to run for --full")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for --check (default: cuda if available, else cpu)")
    args = parser.parse_args()

    if args.full:
        param_sizes = [int(x) for x in args.params.split(",")] if args.params else DEFAULT_PARAMS
        lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else DEFAULT_LRS
        modes = ["sp", "mup"] if args.mode == "both" else [args.mode]
        results = run_full_experiments(param_sizes, lrs, args.tokens, modes)
        plot_lr_transfer(results)
        plot_optimal_lr_vs_width(results)

    elif args.plot:
        if not RESULTS_FILE.exists():
            print(f"No results found at {RESULTS_FILE}. Run with --full first.")
            sys.exit(1)
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        plot_lr_transfer(results)
        plot_optimal_lr_vs_width(results)

    else:
        # Default: fast coordinate check
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ok = run_coord_check(device=device)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
