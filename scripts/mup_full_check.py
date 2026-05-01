#!/usr/bin/env python3
"""Comprehensive μP verification.

Runs 7 ordered checks:
  1. Raw-signal coord check at init
  2. Coord check after K Adam steps (feature-learning regime)
  3. Update-RMS scaling check (Adam: embed flat, hidden ∝ base_dim/dim)
  4. base-dim sanity: μP at base width should ≡ SP
  5. Optimizer param-group sanity
  6. Logit magnitude across widths
  7. LR-transfer experiment (only with --full; hours on GPU)

Usage:
    python scripts/mup_full_check.py             # checks 1-6 (~1 min)
    python scripts/mup_full_check.py --full      # all 7 (hours, GPU)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.double_decoder import Double_Decoder

# ── Config ──────────────────────────────────────────────────────────────────

BASE_DIM = 64
WIDTHS = [64, 128, 256, 512, 1024]
SEQ_LEN = 256
BATCH = 4
VOCAB = 32768
N_STEPS = 5
LR = 2e-3

FLAT_RATIO_OK = 2.0
FLAT_RATIO_WARN = 5.0
QK_STD_OK = (0.5, 5.0)
ENTROPY_NEAR_UNIFORM_FRAC = 0.85


# ── AttnProbe: patch SDPA + flex_attention to log pre-softmax stats ─────────

class AttnProbe:
    def __init__(self):
        self.qk_stds, self.qk_means, self.entropies, self.scales = [], [], [], []
        self._orig_sdpa = None
        self._orig_flex = None

    def reset(self):
        self.qk_stds.clear(); self.qk_means.clear()
        self.entropies.clear(); self.scales.clear()

    def _record(self, q, k, scale, is_causal=False, attn_mask=None):
        with torch.no_grad():
            head_dim = q.shape[-1]
            eff = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
            qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * eff
            if is_causal:
                Lq, Lk = qk.shape[-2], qk.shape[-1]
                m = torch.triu(torch.ones(Lq, Lk, device=qk.device, dtype=torch.bool), diagonal=1)
                qk = qk.masked_fill(m, float('-inf'))
            elif attn_mask is not None and attn_mask.dtype != torch.bool:
                qk = qk + attn_mask
            finite = qk[qk.isfinite()]
            if finite.numel():
                self.qk_stds.append(finite.std().item())
                self.qk_means.append(finite.mean().item())
                probs = qk.softmax(dim=-1)
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
                row_ok = qk.isfinite().any(-1)
                if row_ok.any():
                    self.entropies.append(ent[row_ok].mean().item())
                self.scales.append(eff)

    def _sdpa_wrap(self, q, k, v, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, **kw):
        self._record(q, k, scale, is_causal=is_causal, attn_mask=attn_mask)
        return self._orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                               is_causal=is_causal, scale=scale, **kw)

    def _flex_wrap(self, q, k, v, score_mod=None, block_mask=None,
                   scale=None, return_lse=False, **kw):
        self._record(q, k, scale, is_causal=False)
        return self._orig_flex(q, k, v, score_mod=score_mod, block_mask=block_mask,
                               scale=scale, return_lse=return_lse, **kw)

    def __enter__(self):
        self.reset()
        self._orig_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._sdpa_wrap
        import components.attention as ca
        self._orig_flex = ca.flex_attention
        ca.flex_attention = self._flex_wrap
        return self

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self._orig_sdpa
        import components.attention as ca
        ca.flex_attention = self._orig_flex


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_model(dim, mup_base_dim=0, device="cpu", seed=0):
    torch.manual_seed(seed)
    return Double_Decoder(
        vocab_size=VOCAB, dim=dim, num_heads=dim // 64,
        num_encoder_layers=4, num_decoder_layers=2,
        seq_len=SEQ_LEN, shared=True, logit_biases=False,
        mup_base_dim=mup_base_dim,
    ).to(device)


def _batch(device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), generator=g).to(device)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), generator=g).to(device)
    blocks = torch.tensor([SEQ_LEN // 4, SEQ_LEN // 2, 3 * SEQ_LEN // 4], device=device)
    return {"input_ids": input_ids, "labels": labels, "blocks": blocks, "sft": False}


def _build_opt(model, lr, mup_base_dim, dim):
    if mup_base_dim > 0:
        mult = mup_base_dim / dim
        embed_ids = {id(model.embedding.weight)}
        embed, hidden, base = [], [], []
        seen = set()
        for _, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (embed if id(p) in embed_ids else
             (base if p.dim() <= 1 else hidden)).append(p)
        return AdamW([
            {"params": embed, "weight_decay": 0.1, "lr": lr},
            {"params": hidden, "weight_decay": 0.1, "lr": lr * mult},
            {"params": base, "weight_decay": 0.0, "lr": lr},
        ], betas=(0.9, 0.95), eps=1e-8)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() <= 1 else decay).append(p)
    return AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)


def _rms(t):
    return t.detach().float().pow(2).mean().sqrt().item()


def _ratio(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return float('nan')
    lo = max(min(vals), 1e-15)
    return max(vals) / lo


def _verdict(ratio, ok=FLAT_RATIO_OK, warn=FLAT_RATIO_WARN):
    if math.isnan(ratio): return "n/a"
    if ratio <= ok: return "PASS"
    if ratio <= warn: return "WARN"
    return "FAIL"


def _print_widthwise(title, rows):
    print(f"\n  {title}")
    head = f"    {'metric':<22}" + "".join(f"dim={d:<11}" for d in WIDTHS) + f"{'ratio':<8}{'verdict'}"
    print(head)
    print("    " + "-" * (len(head) - 4))
    for label, vals, verdict_override in rows:
        cells = "".join(f"{v:<15.4g}" if v is not None else f"{'-':<15}" for v in vals)
        r = _ratio(vals)
        v = verdict_override if verdict_override is not None else _verdict(r)
        rstr = f"{r:.2f}x" if not math.isnan(r) else "n/a"
        print(f"    {label:<22}{cells}{rstr:<8}{v}")


# ── Probes ──────────────────────────────────────────────────────────────────

def _probe_one_width(dim, mup_base_dim, device, n_train_steps=0, seed=0):
    """Build a model at `dim`, optionally take N Adam steps, then probe.

    Returns dict of:
      embed_out_enc, embed_out_dec, enc_*, dec_*, logits,
      qk_std, qk_mean, softmax_entropy, attn_scale_used
    """
    model = _build_model(dim, mup_base_dim=mup_base_dim, device=device, seed=seed)
    if n_train_steps > 0:
        model.train()
        opt = _build_opt(model, LR, mup_base_dim, dim)
        for s in range(n_train_steps):
            batch = _batch(device, seed=100 + s)
            out = model(**batch)
            out["loss"].backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        del opt

    model.eval()
    batch = _batch(device, seed=999)
    rec = {}
    layer_rms = {}
    hooks = []

    def hook(name):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            layer_rms[name] = _rms(t)
        return fn

    for i, lyr in enumerate(model.encoder_layers):
        hooks.append(lyr.register_forward_hook(hook(f"enc_{i}")))
    for i, lyr in enumerate(model.decoder_layers):
        hooks.append(lyr.register_forward_hook(hook(f"dec_{i}")))

    probe = AttnProbe()
    try:
        with torch.no_grad():
            raw_emb = model.embedding(batch["input_ids"])
            rec["embed_raw"] = _rms(raw_emb)
            rec["embed_x_mupmult"] = _rms(raw_emb * model.mup_mult)
            with probe:
                out = model(**batch)
            rec["logits"] = _rms(out["logits"])
            rec["loss"] = float(out["loss"].item())
            if probe.qk_stds:
                rec["qk_std"] = float(np.mean(probe.qk_stds))
                rec["qk_mean"] = float(np.mean(probe.qk_means))
                rec["attn_scale"] = float(np.mean(probe.scales))
            if probe.entropies:
                rec["softmax_entropy"] = float(np.mean(probe.entropies))
                rec["entropy_uniform"] = math.log(SEQ_LEN)
        for k, v in layer_rms.items():
            rec[k] = v
    finally:
        for h in hooks:
            h.remove()
    del model
    return rec


def _coord_check_table(records, title):
    """Render a width-vs-metric table from per-width records, return ok flag."""
    metrics_residual = ["embed_raw", "embed_x_mupmult"]
    metrics_residual += [f"enc_{i}" for i in range(4)]
    metrics_residual += [f"dec_{i}" for i in range(2)]
    metrics_residual += ["logits"]
    metrics_attn = ["qk_std", "qk_mean", "softmax_entropy", "attn_scale"]

    rows_resid, rows_attn = [], []
    for m in metrics_residual:
        vals = [records[d].get(m) for d in WIDTHS]
        rows_resid.append((m, vals, None))
    for m in metrics_attn:
        vals = [records[d].get(m) for d in WIDTHS]
        verdict = None
        if m == "qk_std":
            in_range = [v is not None and QK_STD_OK[0] <= v <= QK_STD_OK[1] for v in vals]
            verdict = "PASS" if all(in_range) else (
                "WARN" if any(in_range) else "FAIL"
            )
        elif m == "softmax_entropy":
            uni = math.log(SEQ_LEN)
            near = [v is not None and v > ENTROPY_NEAR_UNIFORM_FRAC * uni for v in vals]
            if all(near):
                verdict = "FAIL"
            elif any(near):
                verdict = "WARN"
        elif m == "qk_mean":
            verdict = "info"
        rows_attn.append((m, vals, verdict))

    _print_widthwise(f"{title} — residual stream", rows_resid)
    _print_widthwise(f"{title} — attention stats (uniform entropy = ln({SEQ_LEN}) = {math.log(SEQ_LEN):.3f})",
                     rows_attn)

    # Use only metrics that should genuinely be flat in the network's actual
    # forward pass. embed_raw is what both encode/decode see; embed_x_mupmult
    # is informational (only meaningful if you're evaluating an alternative
    # parametrization that puts mup_mult on the embedding side).
    flat_metrics = ["embed_raw"] + [f"enc_{i}" for i in range(4)] \
                   + [f"dec_{i}" for i in range(2)] + ["logits"]
    ratios_resid = [_ratio([records[d].get(m) for d in WIDTHS]) for m in flat_metrics]
    ratios_resid = [r for r in ratios_resid if not math.isnan(r)]
    qk_ratio = _ratio([records[d].get("qk_std") for d in WIDTHS])
    qk_in_range = all(records[d].get("qk_std") is not None
                      and QK_STD_OK[0] <= records[d]["qk_std"] <= QK_STD_OK[1]
                      for d in WIDTHS)
    resid_ok = max(ratios_resid) <= FLAT_RATIO_WARN if ratios_resid else False
    qk_ok = qk_ratio <= FLAT_RATIO_WARN and qk_in_range
    return resid_ok and qk_ok


# ── CHECK 1: raw-signal coord check at init ─────────────────────────────────

def check_1(device):
    print("\n" + "=" * 78)
    print("  CHECK 1: raw-signal coord check at init (μP)")
    print("=" * 78)
    records = {d: _probe_one_width(d, BASE_DIM, device, n_train_steps=0) for d in WIDTHS}
    return _coord_check_table(records, "init, μP enabled")


# ── CHECK 2: coord check after K Adam steps ─────────────────────────────────

def check_2(device):
    print("\n" + "=" * 78)
    print(f"  CHECK 2: coord check after {N_STEPS} Adam steps (μP, feature learning)")
    print("=" * 78)
    records = {d: _probe_one_width(d, BASE_DIM, device, n_train_steps=N_STEPS)
               for d in WIDTHS}
    return _coord_check_table(records, f"after {N_STEPS} steps, μP enabled")


# ── CHECK 3: update-RMS scaling ─────────────────────────────────────────────

def check_3(device):
    print("\n" + "=" * 78)
    print("  CHECK 3: update-RMS scaling after one Adam step")
    print("=" * 78)

    def measure(dim, mup_base_dim):
        model = _build_model(dim, mup_base_dim=mup_base_dim, device=device)
        model.train()
        opt = _build_opt(model, LR, mup_base_dim, dim)
        snap = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        out = model(**_batch(device, seed=7))
        out["loss"].backward()
        opt.step()
        embed_rms, hidden_rms = [], []
        for n, p in model.named_parameters():
            if n not in snap:
                continue
            d = (p.detach() - snap[n]).float()
            r = d.norm() / math.sqrt(max(d.numel(), 1))
            if "embedding" in n:
                embed_rms.append(r.item())
            elif p.dim() >= 2 and ("encoder_layers" in n or "decoder_layers" in n):
                hidden_rms.append(r.item())
        del model, opt
        return (float(np.mean(embed_rms)) if embed_rms else 0.0,
                float(np.mean(hidden_rms)) if hidden_rms else 0.0)

    print("\n  ── μP ──")
    embed_rows = []
    hidden_rows = []
    for d in WIDTHS:
        e, h = measure(d, BASE_DIM)
        embed_rows.append(e)
        hidden_rows.append(h)
    expected = [BASE_DIM / d for d in WIDTHS]
    expected = [e / expected[0] for e in expected]
    actual = [h / max(hidden_rows[0], 1e-15) for h in hidden_rows]
    rows = [
        ("embed Δ-RMS",       embed_rows,  None),
        ("hidden Δ-RMS",      hidden_rows, "info"),       # SHOULD shrink as base/dim
        ("hidden vs base",    actual,      "info"),
        ("expected base/dim", expected,    "info"),
    ]
    _print_widthwise("μP one-step update RMS", rows)

    embed_flat = _ratio(embed_rows) <= FLAT_RATIO_WARN
    err = max(abs(a - e) / max(e, 1e-15) for a, e in zip(actual, expected))
    hidden_ok = err < 0.4
    print(f"\n    embed flat across widths: {_verdict(_ratio(embed_rows))}")
    print(f"    hidden-update follows base/dim within ±40%: {'PASS' if hidden_ok else 'FAIL'}"
          f" (max relative error {err:.2%})")

    print("\n  ── SP (sanity: hidden should stay flat) ──")
    sp_h = []
    for d in WIDTHS:
        _, h = measure(d, 0)
        sp_h.append(h)
    _print_widthwise("SP one-step update RMS", [("hidden Δ-RMS", sp_h, None)])
    sp_flat = _ratio(sp_h) <= FLAT_RATIO_WARN
    print(f"    SP hidden flat across widths: {_verdict(_ratio(sp_h))}")

    return embed_flat and hidden_ok and sp_flat


# ── CHECK 4: μP at base width ≡ SP ──────────────────────────────────────────

def check_4(device):
    print("\n" + "=" * 78)
    print(f"  CHECK 4: μP at base width (dim={BASE_DIM}) should ≡ SP")
    print("=" * 78)

    sp = _probe_one_width(BASE_DIM, 0, device, n_train_steps=0)
    mp = _probe_one_width(BASE_DIM, BASE_DIM, device, n_train_steps=0)

    print(f"\n    SP loss        : {sp['loss']:.6f}")
    print(f"    μP loss        : {mp['loss']:.6f}")
    print(f"    SP attn scale  : {sp.get('attn_scale')}")
    print(f"    μP attn scale  : {mp.get('attn_scale')}")
    print(f"    SP qk_std      : {sp.get('qk_std'):.4f}")
    print(f"    μP qk_std      : {mp.get('qk_std'):.4f}")

    loss_match = abs(sp["loss"] - mp["loss"]) < 1e-4
    scale_match = abs((sp.get('attn_scale') or 0) - (mp.get('attn_scale') or 0)) < 1e-9

    if loss_match and scale_match:
        print("    PASS — μP and SP are bitwise-equivalent at base width.")
        return True
    if not scale_match:
        print("    FAIL — attention scale differs at base width:")
        print(f"      SP uses 1/√head_dim = {1.0/math.sqrt(BASE_DIM):.6f}")
        print(f"      μP uses 1/head_dim   = {1.0/BASE_DIM:.6f}")
        print("      Fix the scale convention so SP and μP coincide at base.")
    elif not loss_match:
        print(f"    FAIL — losses diverge by {abs(sp['loss']-mp['loss']):.4g}.")
    return False


# ── CHECK 5: optimizer param-group sanity ───────────────────────────────────

def check_5(device):
    print("\n" + "=" * 78)
    print("  CHECK 5: optimizer param-group sanity")
    print("=" * 78)

    all_ok = True
    for d in WIDTHS:
        model = _build_model(d, BASE_DIM, device)
        opt = _build_opt(model, LR, BASE_DIM, d)
        groups = opt.param_groups
        expected_lrs = [LR, LR * (BASE_DIM / d), LR]
        labels = ["embed", "hidden", "norms/scalars"]
        total = sum(p.numel() for g in groups for p in g["params"])
        model_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n    dim={d}")
        for g, exp, lab in zip(groups, expected_lrs, labels):
            n_params = sum(p.numel() for p in g["params"])
            ok = abs(g["lr"] - exp) < 1e-12
            tag = "PASS" if ok else "FAIL"
            print(f"      [{tag}] {lab:<14} lr={g['lr']:.6g}"
                  f"  expected={exp:.6g}  params={n_params:,}")
            all_ok &= ok
        accounted = total == model_total
        print(f"      [{('PASS' if accounted else 'FAIL')}] total params accounted: "
              f"opt={total:,} vs model={model_total:,}")
        all_ok &= accounted
        del model, opt
    return all_ok


# ── CHECK 6: logit magnitude across widths ──────────────────────────────────

def check_6(device):
    print("\n" + "=" * 78)
    print("  CHECK 6: logit magnitude across widths (should be O(1), flat)")
    print("=" * 78)

    rows_mp, rows_sp = [], []
    for d in WIDTHS:
        m_mp = _probe_one_width(d, BASE_DIM, device, n_train_steps=0, seed=0)
        m_sp = _probe_one_width(d, 0,        device, n_train_steps=0, seed=0)
        rows_mp.append(m_mp["logits"])
        rows_sp.append(m_sp["logits"])
    _print_widthwise("logit RMS at init", [
        ("μP logits RMS", rows_mp, None),
        ("SP logits RMS", rows_sp, None),
    ])
    mp_ratio = _ratio(rows_mp)
    print(f"\n    μP logit-RMS flat across widths: {_verdict(mp_ratio)}"
          f" (ratio {mp_ratio:.2f}x)")
    return mp_ratio <= FLAT_RATIO_WARN


# ── CHECK 7: LR-transfer experiment (slow) ──────────────────────────────────

def check_7(args):
    print("\n" + "=" * 78)
    print("  CHECK 7: LR transfer experiment (this takes hours; GPU)")
    print("=" * 78)
    try:
        from mup_verify import (
            run_full_experiments, plot_lr_transfer, plot_optimal_lr_vs_width,
            DEFAULT_PARAMS, DEFAULT_LRS, DEFAULT_TOKENS,
        )
    except Exception as e:
        print(f"  Could not import mup_verify driver: {e}")
        return False

    params = ([int(x) for x in args.params.split(",")] if args.params else DEFAULT_PARAMS)
    lrs = ([float(x) for x in args.lrs.split(",")] if args.lrs else DEFAULT_LRS)
    tokens = args.tokens or DEFAULT_TOKENS
    modes = ["sp", "mup"]
    results = run_full_experiments(params, lrs, tokens, modes)
    plot_lr_transfer(results)
    plot_optimal_lr_vs_width(results)

    by = {}
    for r in results:
        if r["final_eval_loss"] is None:
            continue
        by.setdefault((r["mode"], r["params"]), []).append(r)
    best = {}
    for k, runs in by.items():
        best[k] = min(runs, key=lambda r: r["final_eval_loss"])

    mup_best_lrs = [best[("mup", p)]["lr"] for p in sorted(set(p for m, p in best if m == "mup"))]
    if not mup_best_lrs:
        print("  FAIL — no successful μP runs.")
        return False
    spread = max(mup_best_lrs) / min(mup_best_lrs)
    print(f"\n    μP optimal-LR spread across widths: {spread:.2f}x"
          f" (PASS if ≤ 2x, WARN if ≤ 4x, else FAIL)")
    if spread <= 2.0:
        return True
    if spread <= 4.0:
        print("    WARN — optimal LR drifting; check for non-μP-tracked components.")
        return True
    print("    FAIL — optimal LR is width-dependent; μP is broken.")
    return False


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="Also run check 7 (slow).")
    p.add_argument("--device", default=None)
    p.add_argument("--params", default=None, help="(check 7) comma-separated param counts")
    p.add_argument("--lrs", default=None, help="(check 7) comma-separated LRs")
    p.add_argument("--tokens", type=int, default=None, help="(check 7) tokens per run")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated subset of checks to run (e.g. '1,3,5')")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | base_dim={BASE_DIM} | widths={WIDTHS}")
    print(f"vocab={VOCAB} | seq={SEQ_LEN} | batch={BATCH} | lr={LR} | n_steps={N_STEPS}")

    selected = set(args.only.split(",")) if args.only else None
    checks = [
        ("1", "init coord",        lambda: check_1(device)),
        ("2", "post-step coord",   lambda: check_2(device)),
        ("3", "update RMS",        lambda: check_3(device)),
        ("4", "base-dim sanity",   lambda: check_4(device)),
        ("5", "optimizer groups",  lambda: check_5(device)),
        ("6", "logit magnitude",   lambda: check_6(device)),
    ]
    if args.full:
        checks.append(("7", "LR transfer", lambda: check_7(args)))

    results = []
    for tag, name, fn in checks:
        if selected and tag not in selected:
            continue
        try:
            ok = fn()
        except Exception as e:
            print(f"\n  CHECK {tag} ({name}) raised: {e!r}")
            ok = False
        results.append((tag, name, ok))

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for tag, name, ok in results:
        print(f"  [{('PASS' if ok else 'FAIL')}] check {tag}: {name}")
    overall = all(ok for _, _, ok in results)
    print("\n  " + ("ALL CHECKS PASSED" if overall else "SOME CHECKS FAILED"))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
