#!/usr/bin/env python3
"""
Full scaling-law grid: 3 model types × 5 param sizes × 5 token budgets.

Trains Double_Decoder (dd), StandardEncDec (sed), and DecoderOnlyModel (dec)
at all 25 (param, token) grid points. Models are compiled once and
re-initialized for each token budget. After every fully-trained model the
full eval suite (configurable, default "paper") runs in-process and the
results are embedded in the per-run JSON.

Usage:
    # Original small sweep (backward-compatible)
    python scripts/parallel_scaling.py --arch-set small --token-set small

    # Large sweep, tiered hardware (run each on the appropriate GPU)
    python scripts/parallel_scaling.py --arch-set large --token-set large \\
        --only-arch 5M,25M     # L40
    python scripts/parallel_scaling.py --arch-set large --token-set large \\
        --only-arch 50M,150M   # H100
    python scripts/parallel_scaling.py --arch-set large --token-set large \\
        --only-arch 300M       # B200

    # Subsets / debug
    python scripts/parallel_scaling.py --token-budgets 10M,50M
    python scripts/parallel_scaling.py --model-types dd,dec
    python scripts/parallel_scaling.py --no-compile
    python scripts/parallel_scaling.py --dry-run                    # plan only
    python scripts/parallel_scaling.py --skip-full-eval             # PPL only
    python scripts/parallel_scaling.py --mid-eval-points 5          # 5 mid evals

    # Smoke test: pretrain + paper_full evals + SFT + paper_full evals (post-SFT).
    # Requires data/SFT/ultrachat.jsonl — run data/retrieval_scripts/ultrachat.py first.
    python scripts/parallel_scaling.py \\
        --arch-set large --only-arch 5M --token-set large --token-budgets 100M \\
        --auto-batch-size --eval-suite paper_full --eval-max-examples 50 \\
        --run-sft --sft-tokens 50000000
"""

import sys
import os
import math
import time
import json
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
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
from collators.double_decoder.pretrain import DDPretrainCollator, BOUNDARY_STRATEGIES
from collators.double_decoder.sft import DDSFTCollator
from collators.encoder_decoder.pretrain import EDPretrainCollator
from collators.encoder_decoder.sft import EDSFTCollator
from collators.decoder.sft import DecoderSFTCollator
from collators.decoder.pretrain import DecoderPretrainCollator
from components.initialization import initialize_model
from configs.scaling import compute_flops_arch, arch_flop_multiplier

# ── Constants ───────────────────────────────────────────────────────────────

MODEL_TYPES = ["dd", "sed", "dec"]
MODEL_TYPE_NAMES = {"dd": "Double_Decoder", "sed": "StandardEncDec", "dec": "DecoderOnly"}

# Fixed depth, width-only scaling for clean μP transfer.
# enc=8, dec=4 across all sizes; only dim varies.
# DD/Dec non-emb ≈ 144·dim²;  StandardEncDec ≈ 160·dim² (extra cross-attn params).
FIXED_ENC_LAYERS = 8
FIXED_DEC_LAYERS = 4

# Sweep grids live in configs/scaling.py so light-weight planning scripts
# (e.g. flop_matched_sweep.py) can import them without pulling in PyTorch.
from configs.scaling import ARCH_SETS, TOKEN_SETS  # noqa: E402

# Backward-compat aliases (kept so external callers / tests still work).
ARCHITECTURES = ARCH_SETS["small"]
TOKEN_BUDGETS = TOKEN_SETS["small"]

SEQ_LEN = 2048


# ── Wandb (optional, opt-in via --wandb-project) ────────────────────────────
# Lazy-imported so the script still runs in environments without wandb installed.
# All helpers no-op when --wandb-project is not set.

_WANDB_OPTS = {"project": None, "entity": None, "prefix": None,
               "_module": None, "_imported": False}


def _wandb_module():
    if not _WANDB_OPTS["_imported"]:
        try:
            import wandb as _wb
            _WANDB_OPTS["_module"] = _wb
        except ImportError:
            print("[wandb] --wandb-project set but `wandb` not installed; "
                  "logging disabled. `pip install wandb` to enable.")
            _WANDB_OPTS["_module"] = None
        _WANDB_OPTS["_imported"] = True
    return _WANDB_OPTS["_module"]


def _wandb_enabled():
    return _WANDB_OPTS["project"] is not None and _wandb_module() is not None


def _init_wandb_runs(model_type, models_info, tok_label, base_config):
    """Spin up one wandb run per (arch, model_type) for this token budget.
    Returns dict keyed by display name `{model_type}_{arch_label}` to match
    how trainers identify themselves inside train_one_budget."""
    if not _wandb_enabled():
        return {}
    wb = _wandb_module()
    runs = {}
    prefix = _WANDB_OPTS["prefix"]
    for m in models_info:
        arch_label = m["name"]
        display = f"{model_type}_{arch_label}"
        name_parts = [p for p in (prefix, model_type, arch_label,
                                  f"{tok_label}tok") if p]
        run_name = "_".join(name_parts)
        cfg = dict(base_config)
        cfg.update({
            "model_type": model_type,
            "arch_label": arch_label,
            "tok_label": tok_label,
            "non_emb_params": m.get("ne"),
            "dim": m["arch"]["dim"],
            "num_encoder_layers": m["arch"]["num_encoder_layers"],
            "num_decoder_layers": m["arch"]["num_decoder_layers"],
        })
        try:
            runs[display] = wb.init(
                project=_WANDB_OPTS["project"],
                entity=_WANDB_OPTS["entity"],
                name=run_name,
                group=f"{arch_label}_{tok_label}tok",
                tags=[model_type, arch_label, tok_label],
                config=cfg,
                reinit=True,
            )
        except Exception as e:
            print(f"[wandb] init failed for {run_name}: {e}")
    return runs


def _finish_wandb_runs(runs):
    if not runs:
        return
    wb = _wandb_module()
    if wb is None:
        return
    for run in runs.values():
        try:
            run.finish()
        except Exception:
            pass


def _wandb_log(runs, display, metrics, step=None):
    if not runs or display not in runs:
        return
    try:
        if step is not None:
            runs[display].log(metrics, step=step)
        else:
            runs[display].log(metrics)
    except Exception as e:
        print(f"[wandb] log failed for {display}: {e}")


def _flatten_evals(evals_dict, prefix):
    """Flatten nested benchmark eval dict to wandb-loggable scalar metrics.
    e.g. {'piqa': {'acc': 0.5}} → {'pretrain_evals/piqa/acc': 0.5}"""
    flat = {}
    if not isinstance(evals_dict, dict):
        return flat
    for eval_name, metrics in evals_dict.items():
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat[f"{prefix}/{eval_name}/{k}"] = v
        elif isinstance(metrics, (int, float)):
            flat[f"{prefix}/{eval_name}"] = metrics
    return flat
MUP_BASE_DIM = 64
# head_dim at base width. Under the current `num_heads = dim // 64` convention
# (build_model below), head_dim = dim/num_heads = 64 at every width — so the μP
# attention scale (√base_head_dim/head_dim, in components/attention.py) collapses
# to standard 1/√head_dim and base-width tuning transfers untouched. If the
# scaling protocol ever changes to widen heads instead of adding them (e.g.
# fixed num_heads), set this to the head_dim that the model has at MUP_BASE_DIM
# under that protocol so attention transfers properly. Tensor Programs V §B.
MUP_BASE_HEAD_DIM = 64
BASE_LR = 0.002  # fallback default; overridden per-arch from configs/mup_tuned.json
BASE_WD = 0.1   # fallback default; overridden per-arch from configs/wd_tuned.json
TARGET_EFFECTIVE_BATCH = 512  # match configs/scaling.py — 512 seqs × 2048 tok = 1.05M tok/step

# CLI-driven toggles set by main() after argparse. Module-level so build_optimizer
# can read them without threading kwargs through train_one_budget. --no-mup zeros
# out the (mup_base_dim/dim) hidden-LR multiplier; --peak-lr overrides BASE_LR /
# configs/mup_tuned.json for the pretrain optimizer only.
MUP_ENABLED = True
LR_OVERRIDE = None
WD_OVERRIDE = None  # set by scripts/wd_sweep.py per cell; bypasses TUNED_WDS/BASE_WD

# ── μP-tuned per-arch base LRs ──────────────────────────────────────────────
# Populated from configs/mup_tuned.json (written by scripts/mup_base_sweep.py).
# Falls back to BASE_LR when the file is absent so smoke tests don't break
# before the sweep has been run. Re-read once at startup; Python module-level
# state is fine since each parallel_scaling.py invocation is single-process.

MUP_TUNED_PATH = PROJECT_ROOT / "configs" / "mup_tuned.json"
TUNED_LRS = {}  # filled by _load_tuned_lrs()


def _load_tuned_lrs():
    """Read configs/mup_tuned.json and populate TUNED_LRS. Silently no-ops
    if the file is missing — callers fall back to BASE_LR. The file format is:
        {"dd": {"base_lr": 0.002, ...}, "sed": {...}, "dec": {...}}
    Only base_lr is consumed today; other fields are reserved for later
    expansion (warmup, beta2, init scale)."""
    global TUNED_LRS
    if not MUP_TUNED_PATH.exists():
        return
    try:
        with open(MUP_TUNED_PATH) as f:
            data = json.load(f)
        TUNED_LRS = {k: v for k, v in data.items() if isinstance(v, dict)}
        print(f"[μP] loaded tuned LRs from {MUP_TUNED_PATH.name}: "
              + ", ".join(f"{k}={v.get('base_lr', '?')}" for k, v in TUNED_LRS.items()))
    except Exception as e:
        print(f"[μP] WARNING: failed to read {MUP_TUNED_PATH}: {e}; "
              f"falling back to BASE_LR={BASE_LR}")
        TUNED_LRS = {}


def base_lr_for(model_type):
    """Return the μP-tuned base LR for a model type, or BASE_LR if untuned."""
    return TUNED_LRS.get(model_type, {}).get("base_lr", BASE_LR)


# ── Per-arch tuned weight decay ─────────────────────────────────────────────
# Populated from configs/wd_tuned.json (written by scripts/wd_sweep.py). Same
# fall-back contract as TUNED_LRS: file absent → callers use BASE_WD. WD has no
# μP transfer claim — the sweep runs at base width only and the resulting WD is
# reused at every scale, mirroring the standard practice for AdamW WD.

MUP_TUNED_WD_PATH = PROJECT_ROOT / "configs" / "wd_tuned.json"
TUNED_WDS = {}  # filled by _load_tuned_wds()


def _load_tuned_wds():
    """Read configs/wd_tuned.json and populate TUNED_WDS. File format:
        {"dd": {"weight_decay": 0.05, ...}, "sed": {...}, "dec": {...}}
    Silently no-ops if the file is missing — callers fall back to BASE_WD."""
    global TUNED_WDS
    if not MUP_TUNED_WD_PATH.exists():
        return
    try:
        with open(MUP_TUNED_WD_PATH) as f:
            data = json.load(f)
        TUNED_WDS = {k: v for k, v in data.items() if isinstance(v, dict)}
        print(f"[μP] loaded tuned WDs from {MUP_TUNED_WD_PATH.name}: "
              + ", ".join(f"{k}={v.get('weight_decay', '?')}"
                          for k, v in TUNED_WDS.items()))
    except Exception as e:
        print(f"[μP] WARNING: failed to read {MUP_TUNED_WD_PATH}: {e}; "
              f"falling back to BASE_WD={BASE_WD}")
        TUNED_WDS = {}


def base_wd_for(model_type):
    """Return the per-arch tuned WD for a model type, or BASE_WD if untuned."""
    return TUNED_WDS.get(model_type, {}).get("weight_decay", BASE_WD)


GPU_PEAK_TFLOPS = {
    "H100": 990, "H200": 990, "A100": 312, "A100-SXM": 624,
    "L40": 362, "B200": 4500,
}


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


def build_model(model_type, arch, vocab_size, device, use_compile=False,
                grad_ckpt=None):
    dim = arch["dim"]
    num_heads = dim // 64
    enc = arch["num_encoder_layers"]
    dec = arch["num_decoder_layers"]

    # Activation memory scales with dim²·layers·seq_len; for the large sweep
    # (dim ≥ 320) the 3 co-resident models won't fit on 80 GB without it.
    # Decoder-only doesn't take this kwarg; it has no checkpoint hooks here.
    if grad_ckpt is None:
        grad_ckpt = dim >= 320

    # When --no-mup is set, drop the model's μP forward-pass multipliers
    # (readout mult, μP attention scale) so the model trains as a vanilla
    # transformer at this width. mup_base_dim=0 makes mup=False inside each
    # model, which sets mup_readout_mult=1.0 and attn_scale=None (SDPA falls
    # back to the standard 1/√head_dim). When MUP_ENABLED, mup_base_head_dim
    # also flows through so the attention scale uses √base_head_dim/head_dim
    # (the canonical μP form — identical to standard at base, providing the
    # extra 1/√head_dim damping when head_dim grows past base_head_dim).
    mup_base = MUP_BASE_DIM if MUP_ENABLED else 0
    mup_base_head = MUP_BASE_HEAD_DIM if MUP_ENABLED else 0

    if model_type == "dd":
        model = Double_Decoder(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads,
            num_encoder_layers=enc, num_decoder_layers=dec,
            seq_len=SEQ_LEN, shared=True, logit_biases=False,
            init_strategy="xavier_uniform", gradient_checkpointing=grad_ckpt,
            mup_base_dim=mup_base, mup_base_head_dim=mup_base_head)
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
            gradient_checkpointing=grad_ckpt,
            mup_base_dim=mup_base, mup_base_head_dim=mup_base_head)
    elif model_type == "dec":
        model = DecoderOnlyModel(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads,
            num_layers=enc + dec, seq_len=SEQ_LEN,
            init_strategy="xavier_uniform",
            mup_base_dim=mup_base, mup_base_head_dim=mup_base_head)

    model = model.to(device)
    if use_compile:
        model = torch.compile(model)
    return model


def build_optimizer(model, dim, model_type=None):
    """Build μP-aware AdamW. If model_type is given, picks the per-arch tuned
    base LR from configs/mup_tuned.json (falls back to BASE_LR otherwise).
    The model_type=None branch preserves the pre-#3 behavior so any unrelated
    callers don't break.

    Honors module-level CLI toggles: LR_OVERRIDE replaces base_lr (used by
    --peak-lr); MUP_ENABLED=False forces mup_mult=1.0 so all param groups
    train at base_lr (used by --no-mup, intended to be paired with a sane
    --peak-lr — see CLI help for guidance)."""
    if LR_OVERRIDE is not None:
        base_lr = LR_OVERRIDE
    else:
        base_lr = base_lr_for(model_type) if model_type else BASE_LR
    if WD_OVERRIDE is not None:
        wd = WD_OVERRIDE
    else:
        wd = base_wd_for(model_type) if model_type else BASE_WD
    mup_mult = (MUP_BASE_DIM / dim) if MUP_ENABLED else 1.0
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
        {"params": embed_params, "lr": base_lr, "weight_decay": wd},
        {"params": hidden_decay, "lr": base_lr * mup_mult, "weight_decay": wd},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95), eps=1e-8, fused=True)


def build_scheduler(optimizer, total_steps):
    warmup = max(1, int(total_steps * 0.05))
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 1.0 - progress * 0.9)
    return LambdaLR(optimizer, lr_lambda)


# ── Auto-tune batch size ────────────────────────────────────────────────────

def _probe_step(model, batch_size, vocab_size, seq_len, device, needs_blocks,
                model_type="dd"):
    """Run one forward+backward+sync at the given batch size. Raises OOM
    or returns successfully — caller catches OOM and shrinks.

    `blocks` shape depends on model type post-Asher-merge: DD wants a 1D
    array of split positions; SED's create_masks_ED wants per-batch encoder
    lengths of shape [batch_size]. We use a [batch_size] tensor of full-length
    values, which DD handles fine (bucketizes to one big block) and SED reads
    as "no padding".
    """
    dummy_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    if model_type == "sed":
        dummy_blocks = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
    else:
        dummy_blocks = torch.sort(torch.randperm(seq_len - 2, device=device)[:4] + 1)[0]
    batch = {"input_ids": dummy_ids, "labels": dummy_ids.clone(),
             "blocks": dummy_blocks, "sft": False}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model_forward(model, batch, needs_blocks)
    out["loss"].backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    torch.cuda.synchronize()


def auto_tune_batch_size(model, vocab_size, seq_len, device, needs_blocks,
                         max_bs=128, candidates=None, model_type="dd"):
    """Probe descending batch sizes; return largest that fits, with one
    safety step below as headroom for activation spikes mid-training."""
    if candidates is None:
        candidates = [128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1]
    candidates = [c for c in candidates if c <= max_bs]
    largest_ok = None
    for bs in candidates:
        try:
            _probe_step(model, bs, vocab_size, seq_len, device, needs_blocks,
                        model_type=model_type)
            largest_ok = bs
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    if largest_ok is None:
        raise RuntimeError(f"Could not fit batch_size=1 for model on device {device}")
    # One step below for safety. _probe_step succeeded at largest_ok, so the
    # next-smaller candidate is also safe; pick it unless we're already at 1.
    safer = next((c for c in candidates if c < largest_ok), largest_ok)
    return safer


def grad_accum_for(batch_size, target_effective):
    """Compute grad-accum to reach target effective batch."""
    return max(1, target_effective // max(1, batch_size))


def target_effective_batch_for(non_emb_params):
    """Auto-scale effective batch (in sequences) with model size.

    Critical batch grows with model capacity (McCandlish et al. 2018).
    A fixed 512 across the whole sweep is sized for the 300M end and
    starves small models of optimizer steps: at 5M params, 500M tokens
    with effective_batch=512 gives only ~477 steps (24 of warmup), and
    end loss floors well above where the model could actually reach.

    Power-law fit B(N) = 96 · (N / 5M)^0.4 anchored at user-validated
    points (5M → 96 seqs, 300M → 512 seqs). Interpolates to:
      25M → 192,  50M → 256,  150M → 384.
    Rounded up to a multiple of 32 (FP / SDPA tile-friendly), clamped
    to [32, 1024].
    """
    raw = 96.0 * (max(1, non_emb_params) / 5e6) ** 0.4
    raw = max(32.0, min(1024.0, raw))
    return int(math.ceil(raw / 32) * 32)


def step_floor_cap(total_tokens, seq_len, min_steps):
    """Largest effective batch (in seqs) that still leaves room for at least
    `min_steps` optimizer steps over `total_tokens` tokens. Used to keep the
    bottom-right of the sweep (large N × small T) from being starved of
    schedule space — at 300M × 100M with the N-only cap (512 seqs) you get
    only ~95 steps (4 warmup), and the LR barely escapes warmup before decay.

    Returns floor(total_tokens / (min_steps · seq_len)) rounded down to a
    multiple of 32, with a hard minimum of 32. If min_steps <= 0, no cap
    (returns float('inf')) so callers can min() it freely with the N-cap.

    Note: when the auto-tuned micro_batch already exceeds this cap (rare —
    typically only for tiny models with very small T), grad_accum collapses
    to 1 and effective_batch = micro_batch > target. The cap can't shrink
    micro_batch from here (that would need re-running auto_tune_batch_size
    with a max_bs constraint); it only constrains grad_accum upward.
    """
    if min_steps is None or min_steps <= 0:
        return float('inf')
    raw = total_tokens // (min_steps * seq_len)
    return max(32, (raw // 32) * 32)


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


# ── SFT helpers ─────────────────────────────────────────────────────────────

def build_sft_collator(model_type, tokenizer, seq_len):
    """Pick the right SFT collator for the model type.
    DD uses DDSFTCollator (encoder + decoder split with cross-attn block_masks).
    SED uses EDSFTCollator (T5-style enc/dec split with per-batch encoder lengths).
    DecoderOnly uses DecoderSFTCollator (single sequence with masked-context labels).
    """
    bos_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>") or 0
    cls = {"dec": DecoderSFTCollator, "sed": EDSFTCollator, "dd": DDSFTCollator}[model_type]
    return cls(bos_token_id=bos_id, eos_token_id=eos_id,
               pad_token_id=pad_id, max_seq_len=seq_len)


def build_pretrain_collator(model_type, bos_id, eos_id, pad_id, seq_len,
                            global_seed=None, boundary_strategy="random_uniform"):
    """Pick the right pretrain collator for the model type.
    DD uses DDPretrainCollator (block-decomposed prefix-LM).
    SED uses EDPretrainCollator (T5-style span corruption).
    DecoderOnly uses DecoderPretrainCollator (single causal stream).

    `boundary_strategy` only affects DD; SED and DEC ignore it. See
    collators/double_decoder/pretrain.py:BOUNDARY_STRATEGIES for the
    supported values.
    """
    if model_type == "dd":
        return DDPretrainCollator(
            bos_token_id=bos_id, eos_token_id=eos_id, max_seq_len=seq_len,
            global_seed=global_seed,
            boundary_strategy=boundary_strategy)
    if model_type == "sed":
        # sentinel_start_id=6 matches the post-merge tokenizer where the
        # first 6 special tokens occupy ids 0..5 and <sentinel_0..99> sit at 6..105.
        return EDPretrainCollator(
            max_seq_len=seq_len, pad_token_id=pad_id,
            bos_token_id=bos_id, eos_token_id=eos_id,
            sentinel_start_id=6, num_sentinel_tokens=100,
            global_seed=global_seed)
    if model_type == "dec":
        return DecoderPretrainCollator(
            bos_token_id=bos_id, eos_token_id=eos_id, max_seq_len=seq_len)
    raise ValueError(f"Unknown model_type: {model_type}")


def build_sft_optimizer(model, dim, base_lr, model_type=None):
    """SFT optimizer: same μP-aware param groups as pretrain but with the
    much smaller SFT base LR (e.g. 2e-5). Embedding + output projection get
    base_lr; hidden weights get base_lr × (mup_base_dim / dim).

    `model_type` is accepted but currently unused — the SFT LR is passed in
    explicitly from CLI (--sft-lr) rather than read from configs/mup_tuned.json
    since SFT is fine-tuning, not pretraining. Kept in the signature so future
    "tune SFT LR per arch" work can plug in without touching call sites."""
    # Honor --no-mup: when μP is globally disabled, the SFT optimizer also
    # gives every group raw base_lr (no mup_base_dim/dim multiplier). Keeps
    # SFT consistent with pretrain so the same flag means the same thing
    # across phases.
    mup_mult = (MUP_BASE_DIM / dim) if MUP_ENABLED else 1.0
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
        {"params": embed_params, "lr": base_lr, "weight_decay": 0.1},
        {"params": hidden_decay, "lr": base_lr * mup_mult, "weight_decay": 0.1},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95), eps=1e-8, fused=True)


def sft_one_model(trainer, sft_loader, device, total_micro_batches,
                  grad_accum, base_lr, log_interval=None,
                  gpu_tflops=None, tokens_per_micro=None):
    """Run SFT on a single trainer's model for `total_micro_batches // grad_accum`
    optimizer steps. Modifies the model in-place.
    Returns (avg_loss, train_time_sec, n_steps, mfu_pct).

    Uses the same compiled forward as pretrain — torch.compile will trace a new
    graph on the first SFT forward (different kwargs: encoder_input_ids/
    decoder_input_ids/sft=True for DD/SED) and cache it. ~30s warmup then fast.

    If gpu_tflops and tokens_per_micro are passed, progress lines and the
    final return include MFU using the same 6·ne·tokens approximation used
    by pretrain — directly comparable across phases.
    """
    eager = getattr(trainer["model"], "_orig_mod", trainer["model"])
    eager.train()
    opt = build_sft_optimizer(trainer["model"], trainer["arch"]["dim"], base_lr)
    total_steps = max(1, total_micro_batches // grad_accum)
    sched = build_scheduler(opt, total_steps)  # reuse 5%-warmup linear-decay
    if log_interval is None:
        log_interval = max(1, total_steps // 10)

    track_mfu = gpu_tflops is not None and tokens_per_micro is not None

    opt.zero_grad(set_to_none=True)
    losses = []  # one entry per optimizer step (token-weighted mean)
    micro = 0
    step = 0
    # Token-weighted accumulation — see training/pretrain.py for full
    # rationale. Single-process here, so no DDP world_size correction.
    accum_loss_sum = 0.0
    accum_n_valid = 0
    t0 = time.time()

    for batch_idx, raw_batch in enumerate(sft_loader):
        if batch_idx >= total_micro_batches:
            break
        batch = {k: (v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v)
                 for k, v in raw_batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = trainer["model"](**batch)
            loss = out["loss"]

        n_valid = int((batch["labels"] != -100).sum().item())
        (loss * n_valid).backward()
        accum_loss_sum += loss.detach().item() * n_valid
        accum_n_valid += n_valid
        micro += 1

        if micro % grad_accum == 0:
            if accum_n_valid > 0:
                scale = 1.0 / accum_n_valid
                for p in eager.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            avg = accum_loss_sum / max(1, accum_n_valid)
            losses.append(avg)
            accum_loss_sum = 0.0
            accum_n_valid = 0
            if step % log_interval == 0:
                elapsed = time.time() - t0
                mfu_str = ""
                if track_mfu:
                    # SFT uses arch-aware FLOPs too — the per-arch multiplier
                    # is the same as pretrain since the model topology is
                    # identical. tokens_per_micro is the per-step input-token
                    # count (bs × seq_len).
                    arch = trainer["arch"]
                    flops = compute_flops_arch(
                        trainer["model_type"], trainer["ne"],
                        micro * tokens_per_micro,
                        enc=arch["num_encoder_layers"],
                        dec=arch["num_decoder_layers"])
                    mfu = flops / (max(elapsed, 1e-6) * gpu_tflops * 1e12) * 100
                    mfu_str = f"  MFU={mfu:.1f}%"
                print(f"      [sft {trainer['display']}] step {step}/{total_steps}  "
                      f"loss={avg:.3f}  [{elapsed:.0f}s]{mfu_str}")

    train_time = time.time() - t0
    final_mfu = 0.0
    if track_mfu and micro > 0:
        arch = trainer["arch"]
        flops = compute_flops_arch(
            trainer["model_type"], trainer["ne"],
            micro * tokens_per_micro,
            enc=arch["num_encoder_layers"],
            dec=arch["num_decoder_layers"])
        final_mfu = flops / (max(train_time, 1e-6) * gpu_tflops * 1e12) * 100
    # Average of last 10% of losses gives a stable end-of-training number.
    tail = max(1, len(losses) // 10)
    avg_final_loss = sum(losses[-tail:]) / tail if losses else float("nan")
    return avg_final_loss, train_time, step, final_mfu


# ── Fractional checkpoint helper ────────────────────────────────────────────

def _save_trainer_checkpoint(t, fraction, total_tokens, batch_size, grad_accum,
                             tok_label, scaling_dir, hf_repo=None):
    """Save trainer ``t``'s weights at progress ``fraction`` (0, 1].

    Writes to ``scaling_dir`` with a ``_pct{NNN}`` suffix and, if ``hf_repo``
    is set, uploads the same file to HF Hub. Failures in the HF leg are
    logged but do NOT abort the training run — the local copy remains.
    """
    pct = int(round(fraction * 100))
    arch = t["arch"]
    actual_base_lr = (LR_OVERRIDE if LR_OVERRIDE is not None
                      else base_lr_for(t["model_type"]))
    if LR_OVERRIDE is not None:
        lr_source = "CLI --peak-lr"
    elif t["model_type"] in TUNED_LRS:
        lr_source = "configs/mup_tuned.json"
    else:
        lr_source = "BASE_LR fallback"
    eager_for_save = getattr(t["model"], "_orig_mod", t["model"])

    hparams = {
        "model_cls": MODEL_TYPE_NAMES[t["model_type"]],
        "dim": arch["dim"],
        "num_encoder_layers": arch["num_encoder_layers"],
        "num_decoder_layers": arch["num_decoder_layers"],
        "num_heads": arch["dim"] // 64,
        "seq_len": SEQ_LEN,
        "mup_base_dim": MUP_BASE_DIM if MUP_ENABLED else 0,
        "mup_enabled": MUP_ENABLED,
        "lr": actual_base_lr,
        "lr_source": lr_source,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "total_tokens": total_tokens,
    }
    ckpt_path = scaling_dir / f"{t['model_type']}_{t['name']}_{tok_label}tok_pct{pct:03d}.pt"
    torch.save({
        "state_dict": eager_for_save.state_dict(),
        "hparams": hparams,
        "vocab_size": eager_for_save.embedding.weight.shape[0],
        "model_type": t["model_type"],
        "step": t["step"],
        "tokens_seen": t["tokens_seen"],
        "fraction": fraction,
    }, ckpt_path)
    print(f"      [save@pct{pct:03d}] {t['display']} step={t['step']} "
          f"tok={t['tokens_seen']/1e6:.1f}M  -> {ckpt_path}")

    if hf_repo:
        try:
            from training.hf_hub import upload_checkpoint
            rel = f"{t['model_type']}/{t['name']}/{tok_label}tok/pct{pct:03d}.pt"
            upload_checkpoint(ckpt_path, hf_repo, rel, repo_type="model")
        except Exception as e:
            print(f"      [hf-upload] FAILED for {ckpt_path}: {e}  -- continuing")
    return ckpt_path


# ── Training one token budget ───────────────────────────────────────────────

def train_one_budget(models_info, tok_label, total_tokens, batch_size, grad_accum,
                     train_loaders, eval_loaders, device, gpu_tflops, eval_batches,
                     output_dir, mid_eval_points=0,
                     run_full_eval=True, eval_suite="paper", eval_max_examples=500,
                     eval_data_file="data/Pretrain/slimpajama_eval_packed.jsonl",
                     tokenizer=None,
                     run_sft=False, sft_train_file=None, sft_eval_file=None,
                     sft_tokens=50_000_000, sft_lr=2e-5, sft_grad_accum=4,
                     wandb_runs=None, save_checkpoints=False,
                     checkpoint_fractions=(), hf_repo=None,
                     step_callback=None):
    """Train all co-resident models for one token budget. After training,
    run held-out PPL eval, then (optionally) the full benchmark suite, and
    write per-run JSONs into output_dir.

    Args:
        mid_eval_points: how many PPL evals to run during training (0 = end-only).
            Each mid-eval costs ~10 forward batches per model — set to 5 for
            a loss curve, 0 for a fast large-token run.
        run_full_eval: if True, run evals/run_evals.run_evals_on_model() on
            each fully-trained model and embed under run_result["benchmark_evals"].
        eval_suite: group name ("paper", "quick", "all", "intrinsic", ...) or
            comma-separated eval names.
        eval_max_examples: per-eval cap (500 matches the SFT script default).
        step_callback: optional callable(trainers, step, total_steps) invoked
            at every grad-accum boundary; mup_base_sweep uses it for the
            mid-training coord probe. `None` (default) keeps the fast path.
    """
    tokens_per_step = batch_size * grad_accum * SEQ_LEN
    total_steps = max(1, total_tokens // tokens_per_step)
    total_micro = total_steps * grad_accum
    log_interval = max(1, total_steps // 20)

    # Re-init weights, build fresh optimizer/scheduler
    trainers = []
    for m in models_info:
        eager = getattr(m["model"], "_orig_mod", m["model"])
        initialize_model(eager, "xavier_uniform")
        opt = build_optimizer(m["model"], m["arch"]["dim"], model_type=m["model_type"])
        sched = build_scheduler(opt, total_steps)
        trainers.append({
            "name": m["name"], "model_type": m["model_type"],
            "display": f"{m['model_type']}_{m['name']}",
            "model": m["model"], "opt": opt, "sched": sched, "ne": m["ne"],
            "needs_blocks": m["needs_blocks"], "arch": m["arch"],
            "step": 0, "micro": 0, "accum_loss_sum": 0.0, "accum_n_valid": 0,
            "train_curve": [], "eval_curve": [], "tokens_seen": 0,
            "pct_fractions_done": set(),
        })

    for t in trainers:
        t["model"].train()
        t["opt"].zero_grad(set_to_none=True)
        # Per-model-type iterator: each trainer's collator produces a different
        # batch shape (DD blocks vs SED T5-corruption vs DEC single-stream).
        t["_train_iter"] = iter(train_loaders[t["model_type"]])

    # Resolve mid-training fractional checkpoints. 1.0 is handled by the
    # end-of-budget save block so it's not in the in-loop schedule.
    mid_fractions = sorted(f for f in (checkpoint_fractions or [])
                           if 0.0 < f < 1.0)
    end_save_uses_fractions = bool(checkpoint_fractions) and 1.0 in checkpoint_fractions

    # Output dir is needed both for mid-training saves and for the end-of-budget
    # results JSON. Materialize it up front so checkpoints have a place to land.
    scaling_dir = Path(output_dir)
    scaling_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for batch_idx in range(total_micro):
        for t in trainers:
            try:
                raw_batch = next(t["_train_iter"])
            except StopIteration:
                t["_train_iter"] = iter(train_loaders[t["model_type"]])
                raw_batch = next(t["_train_iter"])
            batch = {k: v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in raw_batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model_forward(t["model"], batch, t["needs_blocks"])
                loss = out["loss"]

            n_valid = int((batch["labels"] != -100).sum().item())
            (loss * n_valid).backward()
            t["accum_loss_sum"] += loss.detach().item() * n_valid
            t["accum_n_valid"] += n_valid
            t["micro"] += 1

            if t["micro"] % grad_accum == 0:
                if t["accum_n_valid"] > 0:
                    scale = 1.0 / t["accum_n_valid"]
                    for p in t["model"].parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    t["model"].parameters(), 1.0)
                t["opt"].step()
                t["sched"].step()
                t["opt"].zero_grad(set_to_none=True)
                t["step"] += 1
                t["tokens_seen"] += tokens_per_step
                avg_loss = t["accum_loss_sum"] / max(1, t["accum_n_valid"])
                t["train_curve"].append((t["step"], t["tokens_seen"], round(avg_loss, 4)))
                if wandb_runs:
                    try:
                        cur_lr = t["sched"].get_last_lr()[0]
                    except Exception:
                        cur_lr = None
                    # clip_grad_norm_ returns the *pre-clip* total norm. Log
                    # both that and min(., 1.0) (post-clip effective norm) so
                    # the wandb chart shows when clipping is firing.
                    gn = (grad_norm_t.item() if torch.is_tensor(grad_norm_t)
                          else float(grad_norm_t))
                    # Cap exp() input so an early-warmup loss spike doesn't
                    # produce inf and corrupt wandb's auto-scaling.
                    ppl = math.exp(min(avg_loss, 30.0))
                    _wandb_log(wandb_runs, t["display"], {
                        "train/loss": avg_loss,
                        "train/perplexity": ppl,
                        "train/grad_norm": gn,
                        "train/clip_grad_norm": min(gn, 1.0),
                        "train/tokens_seen": t["tokens_seen"],
                        **({"train/lr": cur_lr} if cur_lr is not None else {}),
                    }, step=t["step"])
                t["accum_loss_sum"] = 0.0
                t["accum_n_valid"] = 0

        current_step = trainers[0]["step"]
        is_step_boundary = trainers[0]["micro"] % grad_accum == 0

        # Mid-training fractional checkpoints. Fires once per fraction per
        # trainer when the lockstep step count crosses f * total_steps.
        if (is_step_boundary and current_step > 0 and save_checkpoints
                and mid_fractions):
            for f in mid_fractions:
                target_step = max(1, int(round(f * total_steps)))
                if current_step >= target_step:
                    for t in trainers:
                        if f in t["pct_fractions_done"]:
                            continue
                        t["pct_fractions_done"].add(f)
                        _save_trainer_checkpoint(
                            t, f, total_tokens, batch_size, grad_accum,
                            tok_label, scaling_dir, hf_repo=hf_repo)

        # Periodic eval — only if user opted in. With mid_eval_points=0 we skip
        # all mid-training PPL evals (final-only), saving ~mid_eval_points × 10
        # forward batches per model. The large sweep defaults to 0.
        if mid_eval_points > 0:
            eval_interval = max(1, total_steps // mid_eval_points)
            if current_step > 0 and is_step_boundary and current_step % eval_interval == 0:
                for t in trainers:
                    eval_loss, _ = eval_model(t["model"], eval_loaders[t["model_type"]],
                                              device, eval_batches, t["needs_blocks"])
                    t["eval_curve"].append((t["step"], t["tokens_seen"], round(eval_loss, 4)))
                    if wandb_runs:
                        _wandb_log(wandb_runs, t["display"], {
                            "eval/loss_mid": eval_loss,
                        }, step=t["step"])

        if step_callback is not None and is_step_boundary and current_step > 0:
            step_callback(trainers, current_step, total_steps)

        # Log progress
        if current_step > 0 and current_step % log_interval == 0 and is_step_boundary:
            elapsed = time.time() - t0
            # Arch-aware FLOPs: DD's combo cross-attn and SED's sparse decoder
            # output mean the universal 6NT understates DD by ~1.33× and
            # overstates SED by ~0.81×. See configs/scaling.compute_flops_arch.
            total_flops = sum(
                compute_flops_arch(t["model_type"], t["ne"], t["tokens_seen"],
                                   enc=t["arch"]["num_encoder_layers"],
                                   dec=t["arch"]["num_decoder_layers"])
                for t in trainers)
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
    total_flops = sum(
        compute_flops_arch(t["model_type"], t["ne"], t["tokens_seen"],
                           enc=t["arch"]["num_encoder_layers"],
                           dec=t["arch"]["num_decoder_layers"])
        for t in trainers)
    agg_mfu = total_flops / (train_time * gpu_tflops * 1e12) * 100
    print(f"    Done in {train_time:.1f}s  MFU={agg_mfu:.1f}%")

    # Eval + write per-run results — all artifacts land in output_dir so
    # L40/H100/B200 tier jobs writing to the same parallel_scaling/ folder
    # produce a unified result set. (scaling_dir was created above so mid-
    # training fractional checkpoints had a place to land.)

    # Lazy import — only needed when run_full_eval is True. Doing it here
    # lets `--skip-full-eval` runs work even if eval module deps are missing.
    if run_full_eval:
        from evals.run_evals import run_evals_on_model, resolve_eval_names

    results = {}
    for t in trainers:
        avg_loss, ppl = eval_model(t["model"], eval_loaders[t["model_type"]],
                                   device, eval_batches, t["needs_blocks"])
        print(f"      {t['display']:>12}: eval_loss={avg_loss:.4f}  ppl={ppl:.1f}")
        if wandb_runs:
            _wandb_log(wandb_runs, t["display"], {
                "eval/loss": avg_loss,
                "eval/ppl": ppl,
            }, step=t["step"])

        arch = t["arch"]
        total_params = sum(p.numel() for p in t["model"].parameters())

        # Add final eval to eval_curve
        final_eval = (t["step"], t["tokens_seen"], round(avg_loss, 4))
        eval_curve = t["eval_curve"]
        if not eval_curve or eval_curve[-1][0] != t["step"]:
            eval_curve.append(final_eval)

        # Arch-aware FLOPs + per-arch tuned LR snapshot. Both fields are new
        # in the post-Item-#1/#3 JSON schema. recompute_flops.py back-fills
        # `flops_arch` on older runs that only logged `non_emb_params` and
        # `tokens_seen`.
        flops_arch = compute_flops_arch(
            t["model_type"], t["ne"], t["tokens_seen"],
            enc=arch["num_encoder_layers"], dec=arch["num_decoder_layers"])
        actual_base_lr = (LR_OVERRIDE if LR_OVERRIDE is not None
                          else base_lr_for(t["model_type"]))
        if LR_OVERRIDE is not None:
            lr_source = "CLI --peak-lr"
        elif t["model_type"] in TUNED_LRS:
            lr_source = "configs/mup_tuned.json"
        else:
            lr_source = "BASE_LR fallback"

        run_result = {
            "final_eval_loss": round(avg_loss, 4),
            "final_eval_ppl": round(ppl, 2),
            "total_steps": t["step"],
            "tokens_seen": t["tokens_seen"],
            "total_params": total_params,
            "non_emb_params": t["ne"],
            "training_time_sec": round(train_time, 2),
            "model_type": t["model_type"],
            "flops_arch": flops_arch,
            "flops_naive_6NT": 6 * t["ne"] * t["tokens_seen"],
            "flop_arch_multiplier": arch_flop_multiplier(
                t["model_type"],
                enc=arch["num_encoder_layers"],
                dec=arch["num_decoder_layers"]),
            "hparams": {
                "model_cls": MODEL_TYPE_NAMES[t["model_type"]],
                "dim": arch["dim"],
                "num_encoder_layers": arch["num_encoder_layers"],
                "num_decoder_layers": arch["num_decoder_layers"],
                "num_heads": arch["dim"] // 64,
                "seq_len": SEQ_LEN,
                "mup_base_dim": MUP_BASE_DIM if MUP_ENABLED else 0,
                "mup_enabled": MUP_ENABLED,
                "lr": actual_base_lr,
                "lr_source": lr_source,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum,
                "total_tokens": total_tokens,
                "boundary_strategy": getattr(
                    train_loaders[t["model_type"]].collate_fn,
                    "boundary_strategy", None),
            },
            # Curves: each entry is [step, tokens_seen, loss]
            "train_curve": t["train_curve"],
            "eval_curve": eval_curve,
        }

        # Persist trained weights (opt-in via --save-checkpoints) so chat.py
        # / inference scripts can reload the same model. Stored alongside the
        # per-run JSON; payload is state_dict + minimal hparams + vocab_size.
        # When --checkpoint-fractions is set with 1.0, route through the
        # fractional helper so the file gets a _pct100 suffix and HF upload.
        if save_checkpoints:
            if end_save_uses_fractions:
                _save_trainer_checkpoint(
                    t, 1.0, total_tokens, batch_size, grad_accum,
                    tok_label, scaling_dir, hf_repo=hf_repo)
            elif not checkpoint_fractions:
                # Legacy single-save path (no --checkpoint-fractions specified).
                ckpt_path = scaling_dir / f"{t['model_type']}_{t['name']}_{tok_label}tok.pt"
                eager_for_save = getattr(t["model"], "_orig_mod", t["model"])
                torch.save({
                    "state_dict": eager_for_save.state_dict(),
                    "hparams": run_result["hparams"],
                    "vocab_size": eager_for_save.embedding.weight.shape[0],
                    "model_type": t["model_type"],
                }, ckpt_path)
                print(f"      [save] {t['display']} -> {ckpt_path}")

        # Pre-SFT eval suite (in-process; reuses already-compiled graph; pass
        # the eager module so eval doesn't trigger recompiles for new shapes).
        eval_names = []
        if run_full_eval:
            try:
                eval_names = resolve_eval_names(eval_suite)
            except ValueError as e:
                print(f"      [eval] bad suite '{eval_suite}': {e}")
                eval_names = []

        if eval_names:
            eager = getattr(t["model"], "_orig_mod", t["model"])
            is_enc_dec = t["model_type"] in ("dd", "sed")
            eval_t0 = time.time()
            print(f"      [pretrain-eval] running {len(eval_names)} evals on "
                  f"{t['display']} (max_examples={eval_max_examples})...")
            eager.eval()
            try:
                bench = run_evals_on_model(
                    model=eager, tokenizer=tokenizer, device=device,
                    is_enc_dec=is_enc_dec, eval_names=eval_names,
                    max_examples=eval_max_examples, batch_size=batch_size,
                    eval_file=eval_data_file,
                )
                run_result["pretrain_evals"] = bench
                run_result["pretrain_eval_time_sec"] = round(time.time() - eval_t0, 1)
                if wandb_runs:
                    _wandb_log(wandb_runs, t["display"],
                               _flatten_evals(bench, "pretrain_evals"),
                               step=t["step"])
            except Exception as e:
                print(f"      [pretrain-eval] FAILED for {t['display']}: {e}")
                import traceback
                traceback.print_exc()
                run_result["pretrain_evals_error"] = str(e)
            finally:
                eager.train()

        # ── SFT step ─────────────────────────────────────────────────────
        # Builds a per-model-type dataloader with the right SFT collator,
        # SFT-trains for sft_tokens (~50M default), then re-runs the same
        # eval suite on the SFT'd model. Both result blocks land in the
        # same per-run JSON so plot_comparison can show pre/post deltas.
        if run_sft:
            sft_t0 = time.time()

            # Drop the pretrain optimizer state before SFT builds a fresh
            # one. AdamW keeps fp32 m/v moments per parameter — cheap for
            # 5M params but climbs into GBs for 300M+. Without this, both
            # pretrain and SFT optimizers are alive simultaneously, eating
            # into the activation-memory headroom that triggered the DD
            # bs=48 OOM observed at the SFT/eval boundary.
            t["opt"] = None
            t["sched"] = None
            for p in t["model"].parameters():
                if p.grad is not None:
                    p.grad = None
            torch.cuda.empty_cache()

            # Enc-dec SFT halves the per-batch headroom: the encoder
            # forward runs over `encoder_input_ids` while the decoder
            # forward runs over `decoder_input_ids`, and both sets of
            # activations live through backward. Pretrain auto-tune
            # picked a bs that's already at the OOM ceiling for DD's
            # 12·d² layer density (cf. the 12 GiB dlogits allocation
            # that crashed the first run). Halve bs and double ga so
            # the effective batch — and hence the SGD trajectory —
            # stays identical.
            if t["model_type"] in ("dd", "sed"):
                sft_bs = max(1, batch_size // 2)
                sft_ga = max(1, sft_grad_accum * 2)
            else:
                sft_bs = batch_size
                sft_ga = sft_grad_accum

            sft_collator = build_sft_collator(t["model_type"], tokenizer, SEQ_LEN)
            try:
                sft_train_ds = load_dataset(
                    "json", data_files=sft_train_file, split="train")
            except Exception as e:
                print(f"      [sft] FAILED to load {sft_train_file}: {e}")
                run_result["sft_error"] = str(e)
                # Skip SFT for this trainer; still write JSON below.
                run_path = scaling_dir / f"{t['model_type']}_{t['name']}_{tok_label}tok_results.json"
                with open(run_path, "w") as f:
                    json.dump(run_result, f, indent=2)
                results[t["display"]] = run_result
                results[t["display"]]["aggregate_mfu_pct"] = round(agg_mfu, 2)
                continue

            sft_loader = DataLoader(
                sft_train_ds, batch_size=sft_bs, shuffle=True,
                collate_fn=sft_collator, drop_last=True,
                num_workers=2, pin_memory=True)

            # Total micro-batches to hit sft_tokens. (sft_bs × seq_len
            # tokens forward-passed per micro-batch; loss is computed over
            # output tokens only but compute scales with total seq.)
            sft_total_micro = max(sft_ga,
                                  sft_tokens // (sft_bs * SEQ_LEN))
            # Cap to dataset size so we don't iterate past the file.
            sft_total_micro = min(sft_total_micro, len(sft_loader))
            sft_steps = sft_total_micro // sft_ga

            print(f"      [sft] {t['display']}: ~{sft_tokens/1e6:.0f}M tokens "
                  f"({sft_total_micro} micro / {sft_steps} steps  "
                  f"bs={sft_bs} ga={sft_ga} lr={sft_lr})")

            try:
                avg_sft_loss, sft_train_time, sft_n_steps, sft_mfu = sft_one_model(
                    t, sft_loader, device, sft_total_micro,
                    sft_ga, sft_lr,
                    gpu_tflops=gpu_tflops,
                    tokens_per_micro=sft_bs * SEQ_LEN)
                run_result["sft_final_loss"] = round(avg_sft_loss, 4)
                run_result["sft_train_time_sec"] = round(sft_train_time, 1)
                run_result["sft_total_steps"] = sft_n_steps
                run_result["sft_tokens"] = sft_total_micro * sft_bs * SEQ_LEN
                run_result["sft_mfu_pct"] = round(sft_mfu, 2)
                run_result["sft_hparams"] = {
                    "lr": sft_lr, "grad_accum": sft_ga,
                    "batch_size": sft_bs, "tokens_target": sft_tokens,
                }
                print(f"      [sft] done: loss={avg_sft_loss:.3f}  "
                      f"time={sft_train_time:.0f}s  MFU={sft_mfu:.1f}%")
                if wandb_runs:
                    _wandb_log(wandb_runs, t["display"], {
                        "sft/loss": avg_sft_loss,
                        "sft/training_time_sec": sft_train_time,
                        "sft/mfu_pct": sft_mfu,
                        "sft/total_steps": sft_n_steps,
                        "sft/tokens": run_result["sft_tokens"],
                    })
            except Exception as e:
                print(f"      [sft] training FAILED for {t['display']}: {e}")
                import traceback
                traceback.print_exc()
                run_result["sft_error"] = str(e)

            # Post-SFT eval suite (skipped if SFT itself failed).
            if eval_names and "sft_error" not in run_result:
                eager = getattr(t["model"], "_orig_mod", t["model"])
                is_enc_dec = t["model_type"] in ("dd", "sed")
                eval_t0 = time.time()
                print(f"      [sft-eval] running {len(eval_names)} evals on "
                      f"SFT'd {t['display']} (max_examples={eval_max_examples})...")
                eager.eval()
                try:
                    bench_sft = run_evals_on_model(
                        model=eager, tokenizer=tokenizer, device=device,
                        is_enc_dec=is_enc_dec, eval_names=eval_names,
                        max_examples=eval_max_examples, batch_size=batch_size,
                        eval_file=eval_data_file,
                    )
                    run_result["sft_evals"] = bench_sft
                    run_result["sft_eval_time_sec"] = round(time.time() - eval_t0, 1)
                    if wandb_runs:
                        _wandb_log(wandb_runs, t["display"],
                                   _flatten_evals(bench_sft, "sft_evals"))
                except Exception as e:
                    print(f"      [sft-eval] FAILED for {t['display']}: {e}")
                    import traceback
                    traceback.print_exc()
                    run_result["sft_evals_error"] = str(e)
                finally:
                    eager.train()

        # Write: {output_dir}/{prefix}_{size}_{tokens}tok_results.json
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
    # Grid selection
    parser.add_argument("--arch-set", choices=sorted(ARCH_SETS.keys()),
                        default="small",
                        help="Predefined arch set: 'small' (0.6M–28.9M, original) "
                             "or 'large' (5M–300M, new sweep). Default: small")
    parser.add_argument("--token-set", choices=sorted(TOKEN_SETS.keys()),
                        default="small",
                        help="Predefined token set: 'small' (10M–600M) or "
                             "'large' (100M–6B). Default: small")
    parser.add_argument("--only-arch", type=str, default=None,
                        help="Comma-separated arch labels to run from the chosen "
                             "arch set (e.g. '5M,25M' for L40 tier of large sweep)")
    parser.add_argument("--token-budgets", type=str, default=None,
                        help="Comma-separated token budgets (e.g. '10M,50M')")
    parser.add_argument("--model-types", type=str, default=None,
                        help="Comma-separated model types (e.g. 'dd,dec'). "
                             "Default: all 3 (dd,sed,dec)")
    # Batch sizing
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-step batch size; overridden by --auto-batch-size")
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="Grad accumulation; overridden by --auto-batch-size")
    parser.add_argument("--auto-batch-size", action="store_true",
                        help="Probe largest fitting batch size per (arch, gpu) "
                             "and derive grad_accum from --target-effective-batch")
    parser.add_argument("--target-effective-batch", type=int,
                        default=None,
                        help=f"Effective batch (seqs) for grad_accum derivation. "
                             f"Default: AUTO — scales with model size via "
                             f"target_effective_batch_for(non_emb_params), "
                             f"giving 96 seqs for 5M up to 512 for 300M, then "
                             f"capped per-cell so each (arch, tokens) cell gets "
                             f"at least --min-optimizer-steps. Pass an int to "
                             f"override both (legacy fixed-batch behaviour was "
                             f"{TARGET_EFFECTIVE_BATCH}).")
    parser.add_argument("--min-optimizer-steps", type=int, default=500,
                        help="Step-budget floor: when --target-effective-batch "
                             "is AUTO, each (arch, tokens) cell's target is "
                             "capped down so total_tokens / (target · seq_len) "
                             ">= this many optimizer steps. Prevents large-N × "
                             "small-T cells (300M × 100M = 95 steps with the "
                             "N-only cap) from starving the LR schedule. Set 0 "
                             "to disable (use the N-only cap regardless of T). "
                             "Ignored when --target-effective-batch is set.")
    # Optimizer / μP toggles (pretrain only; SFT path is unaffected)
    parser.add_argument("--no-mup", action="store_true",
                        help="Disable μP entirely. With --no-mup, models are "
                             "built with mup_base_dim=0 (no readout multiplier, "
                             "no μP-flavored attention scale — falls back to "
                             "the standard 1/√head_dim) AND all optimizer param "
                             "groups use base LR directly (no mup_base_dim/dim "
                             "multiplier on hidden weights). The model trains "
                             "as a vanilla transformer at this width. Only "
                             "sensible paired with --peak-lr at a width-"
                             "appropriate value (e.g. 3e-4 for dim=576). Without "
                             "--peak-lr the base LR stays at BASE_LR=2e-3, which "
                             "will almost certainly diverge for non-tiny dims.")
    parser.add_argument("--peak-lr", type=float, default=None,
                        help="Override pretrain base LR. Bypasses configs/"
                             "mup_tuned.json and BASE_LR. Typical non-μP values: "
                             "3e-4 (safe default for ~50M dec), 6e-4 (nanoGPT-"
                             "style aggressive). Does not affect SFT (--sft-lr).")
    parser.add_argument("--max-batch-size", type=int, default=128,
                        help="Cap for the auto-tune search ceiling")
    # Eval control
    parser.add_argument("--eval-batches", type=int, default=10,
                        help="Forward batches for the cheap held-out PPL eval")
    parser.add_argument("--mid-eval-points", type=int, default=0,
                        help="Number of mid-training PPL eval points "
                             "(0 = end-only, default). Use 5 for a loss curve.")
    parser.add_argument("--eval-suite", type=str, default="paper",
                        help="Eval group ('paper'|'quick'|'all'|'intrinsic') or "
                             "comma-separated list of eval names. Default: paper")
    parser.add_argument("--eval-max-examples", type=int, default=500,
                        help="Per-eval example cap (default 500)")
    parser.add_argument("--skip-full-eval", action="store_true",
                        help="Skip the in-process benchmark suite after training")
    parser.add_argument("--eval-data-file", type=str, default=None,
                        help="Held-out file for intrinsic ppl/bpb evals "
                             "(default: derived from --eval-file)")
    # SFT step (off by default for back-compat; smoke test enables it)
    parser.add_argument("--run-sft", dest="run_sft", action="store_true",
                        default=False,
                        help="After pretrain + pretrain-eval, SFT each model on "
                             "UltraChat for --sft-tokens, then re-run the eval suite. "
                             "Per-run JSON gains pretrain_evals + sft_evals keys.")
    parser.add_argument("--no-sft", dest="run_sft", action="store_false",
                        help="Explicitly disable SFT (overrides --run-sft).")
    parser.add_argument("--sft-tokens", type=int, default=50_000_000,
                        help="SFT token budget per model (default 50M; matches "
                             "the data prep target in retrieval_scripts/ultrachat.py)")
    parser.add_argument("--sft-train-file", type=str,
                        default="data/SFT/ultrachat.jsonl",
                        help="SFT training data (run "
                             "scripts/retrieval_scripts/ultrachat.py to build it)")
    parser.add_argument("--sft-eval-file", type=str,
                        default="data/SFT/ultrachat_eval.jsonl")
    parser.add_argument("--sft-lr", type=float, default=2e-5,
                        help="SFT base LR (μP-scaled per arch internally). "
                             "Default 2e-5 matches existing configs/runs/sft_*.yaml")
    parser.add_argument("--sft-grad-accum", type=int, default=4,
                        help="SFT grad-accum steps. Effective batch is "
                             "batch_size * sft_grad_accum.")
    # Item #2: block boundary ablation. DD-only; SED/DEC ignore the flag.
    parser.add_argument("--boundary-strategy", type=str, default="random_uniform",
                        choices=list(BOUNDARY_STRATEGIES),
                        help="DD pretrain block-boundary distribution. "
                             "'random_uniform' (default) keeps existing behavior; "
                             "'prompt_style' samples from the SFT prompt-length "
                             "histogram (build with data/SFT/build_prompt_hist.py); "
                             "'single_middle' / 'logspace' for ablation controls.")
    # I/O + misc
    parser.add_argument("--output-dir", type=str, default="checkpoints/parallel_scaling")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="After each (arch, tokens, model_type) finishes pretraining, "
                             "dump model weights to <output-dir>/<model_type>_<arch>_<tokens>tok.pt "
                             "so chat.py / inference can reload them. Off by default to avoid "
                             "filling disk during full sweeps.")
    parser.add_argument("--checkpoint-fractions", type=str, default="",
                        help="Comma-separated training-progress fractions in (0,1] "
                             "at which to save checkpoints, e.g. '0.5,0.9,1.0'. Each "
                             "produces a `_pct{NNN}.pt` file in --output-dir. "
                             "Requires --save-checkpoints. Default '': legacy single "
                             "end-of-run save with no _pct suffix.")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="If set, upload each saved checkpoint to this HF Hub repo "
                             "(e.g. 'username/bbdd-scaling-checkpoints'). Requires "
                             "huggingface_hub installed and `huggingface-cli login`. "
                             "Path-in-repo: <model_type>/<arch>/<tokens>tok/pct{NNN}.pt.")
    parser.add_argument("--hf-private", action="store_true",
                        help="If --hf-repo is set and the repo doesn't exist yet, "
                             "create it as private. Default: public.")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only: print resolved batch sizes and FLOPs/cell")
    parser.add_argument("--train-file", type=str,
                        default="data/Pretrain/slimpajama_6b_packed.jsonl")
    parser.add_argument("--eval-file", type=str,
                        default="data/Pretrain/slimpajama_6b_eval_packed.jsonl")
    parser.add_argument("--tokenizer-file", type=str,
                        default="tokenizer/tokenizer_32k.json")
    # Wandb (optional, opt-in)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="If set, log each (arch, tok_budget, model_type) cell "
                             "as a wandb run. Default: no wandb.")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Wandb entity (team/user); uses your wandb default if omitted.")
    parser.add_argument("--wandb-run-name-prefix", type=str, default=None,
                        help="Optional prefix for wandb run names "
                             "(useful for namespacing related sweeps).")
    # Internal: subprocess isolation entrypoint. When set, this process trains
    # only the specified model type and exits. Spawned by the parent process so
    # CUDA allocator state and any latent compiled-kernel caches don't bleed
    # across model types in long sweeps.
    parser.add_argument("--_subprocess-mt", dest="_subprocess_mt", default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Parse + validate --checkpoint-fractions / --hf-repo. Failing here is much
    # cheaper than failing 4 hours into a 6B-token cell.
    checkpoint_fractions = []
    if args.checkpoint_fractions:
        try:
            checkpoint_fractions = sorted({
                float(x) for x in args.checkpoint_fractions.split(",") if x.strip()
            })
        except ValueError as e:
            print(f"ERROR: --checkpoint-fractions must be comma-separated floats, got "
                  f"'{args.checkpoint_fractions}': {e}")
            sys.exit(1)
        for f in checkpoint_fractions:
            if not (0.0 < f <= 1.0):
                print(f"ERROR: --checkpoint-fractions values must be in (0, 1], got {f}")
                sys.exit(1)
        if not args.save_checkpoints:
            print("ERROR: --checkpoint-fractions requires --save-checkpoints")
            sys.exit(1)

    if args.hf_repo:
        if not args.save_checkpoints:
            print("ERROR: --hf-repo requires --save-checkpoints (nothing to upload otherwise)")
            sys.exit(1)
        try:
            from training.hf_hub import verify_repo
            verify_repo(args.hf_repo, private=args.hf_private)
        except ImportError as e:
            print(f"ERROR: --hf-repo set but huggingface_hub unavailable: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: HF preflight failed for {args.hf_repo}: {e}")
            print("  Did you run `huggingface-cli login`? Token needs write scope.")
            sys.exit(1)

    if args.wandb_project:
        _WANDB_OPTS["project"] = args.wandb_project
        _WANDB_OPTS["entity"] = args.wandb_entity
        _WANDB_OPTS["prefix"] = args.wandb_run_name_prefix
        if _wandb_module() is None:
            print("[wandb] disabling wandb (import failed); continuing without it")
            _WANDB_OPTS["project"] = None

    # Pick up per-arch tuned LRs from configs/mup_tuned.json (silent no-op if
    # the file doesn't exist yet — happens until you've run mup_base_sweep).
    _load_tuned_lrs()
    # Same for per-arch tuned WDs from configs/wd_tuned.json (written by
    # scripts/wd_sweep.py, run after mup_base_sweep so each arch's LR is tuned).
    _load_tuned_wds()

    # Apply --no-mup / --peak-lr to the module globals consumed by
    # build_optimizer. We do this after _load_tuned_lrs so --peak-lr cleanly
    # overrides anything that file would have provided.
    global MUP_ENABLED, LR_OVERRIDE
    MUP_ENABLED = not args.no_mup
    LR_OVERRIDE = args.peak_lr
    if not MUP_ENABLED or LR_OVERRIDE is not None:
        print(f"[optim] mup_enabled={MUP_ENABLED}  "
              f"lr_override={LR_OVERRIDE}  "
              f"(BASE_LR={BASE_LR}, MUP_BASE_DIM={MUP_BASE_DIM})")

    # Resolve arch + token grid
    arch_list = ARCH_SETS[args.arch_set]
    if args.only_arch:
        wanted = set(s.strip() for s in args.only_arch.split(","))
        arch_list = [(name, arch) for name, arch in arch_list if name in wanted]
        if not arch_list:
            print(f"--only-arch {args.only_arch} matched nothing in arch-set "
                  f"'{args.arch_set}': {[n for n,_ in ARCH_SETS[args.arch_set]]}")
            sys.exit(1)

    token_list = TOKEN_SETS[args.token_set]
    if args.token_budgets:
        requested = set(args.token_budgets.split(","))
        budgets = [(l, v) for l, v in token_list if l in requested]
    else:
        budgets = list(token_list)

    if args.model_types:
        model_types = [mt.strip() for mt in args.model_types.split(",")]
    else:
        model_types = list(MODEL_TYPES)

    # Tag for the per-budget aggregate filename so concurrent tier jobs
    # (L40/H100/B200) don't race on the same parallel_<tok>tok_results.json.
    arch_tag = "-".join(name for name, _ in arch_list)

    if args.eval_data_file is None:
        args.eval_data_file = args.eval_file

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    use_compile = not args.no_compile

    gpu_tflops = detect_gpu_tflops()
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}  (peak BF16: {gpu_tflops:.0f} TFLOPS)")
    print(f"Arch set: {args.arch_set}  archs: {[n for n,_ in arch_list]}")
    print(f"Token set: {args.token_set}  budgets: {[l for l,_ in budgets]}")
    print(f"Model types: {model_types}")
    print(f"Auto batch size: {'ON' if args.auto_batch_size else 'OFF'}  "
          f"target effective batch: "
          f"{args.target_effective_batch if args.target_effective_batch is not None else 'AUTO (per-arch from non_emb_params)'}")
    print(f"Mid-training eval points: {args.mid_eval_points}  |  "
          f"Full eval suite: {'OFF' if args.skip_full_eval else args.eval_suite}")
    if args.run_sft:
        # Fail fast if SFT data missing — better here than 30min into a run.
        for f in (args.sft_train_file, args.sft_eval_file):
            full = PROJECT_ROOT / f if not os.path.isabs(f) else Path(f)
            if not full.exists():
                print(f"ERROR: --run-sft set but SFT data missing: {full}")
                print(f"  Run: python data/retrieval_scripts/ultrachat.py "
                      f"--tokenizer tokenizer/tokenizer_32k.json")
                sys.exit(1)
        print(f"SFT: ON  budget={args.sft_tokens/1e6:.0f}M tok  lr={args.sft_lr}  "
              f"ga={args.sft_grad_accum}  data={args.sft_train_file}")
    else:
        print(f"SFT: OFF (use --run-sft to enable)")
    print(f"torch.compile: {'ON' if use_compile else 'OFF'}  |  matmul precision: TF32")
    if _wandb_enabled():
        print(f"wandb: ON  project={_WANDB_OPTS['project']}  "
              f"entity={_WANDB_OPTS['entity'] or '<default>'}  "
              f"prefix={_WANDB_OPTS['prefix'] or '<none>'}")
    else:
        print(f"wandb: OFF (use --wandb-project to enable)")
    print(f"Output dir: {args.output_dir}  |  arch tag: {arch_tag}")
    # Surface the two new knobs prominently so users can audit them in logs
    # and so any unexpected default doesn't slip through silently.
    print(f"DD boundary strategy: {args.boundary_strategy}  "
          f"({'OK' if args.boundary_strategy == 'random_uniform' else 'NON-DEFAULT'})")
    if TUNED_LRS:
        lr_summary = "  ".join(
            f"{mt}={base_lr_for(mt):.2e}" for mt in MODEL_TYPES if mt in model_types)
        print(f"μP tuned base LRs: {lr_summary}")
    else:
        print(f"μP tuned base LRs: (none — using BASE_LR={BASE_LR} for all archs)")

    # ── Tokenizer + Data ────────────────────────────────────────────────────
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(PROJECT_ROOT / args.tokenizer_file))
    vocab_size = tokenizer.vocab_size
    bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    print(f"Tokenizer: vocab_size={vocab_size}")

    print("Loading data...")
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>") or 0
    # Per-model-type collators: DD does block-decomposed prefix-LM, SED does
    # T5-style span corruption, DEC does single-stream causal. Each gets its
    # own DataLoader so batch shapes don't have to match.
    collators = {
        mt: build_pretrain_collator(mt, bos_token_id, eos_token_id,
                                    pad_token_id, SEQ_LEN,
                                    global_seed=42 + i,
                                    boundary_strategy=args.boundary_strategy)
        for i, mt in enumerate(MODEL_TYPES)
    }

    train_ds = load_dataset(
        "json", data_files=str(PROJECT_ROOT / args.train_file),
        split="train", streaming=True)
    eval_ds = load_dataset(
        "json", data_files=str(PROJECT_ROOT / args.eval_file),
        split="train")

    train_loaders = {
        mt: DataLoader(train_ds, batch_size=args.batch_size,
                       collate_fn=collators[mt], drop_last=True)
        for mt in MODEL_TYPES
    }
    eval_loaders = {
        mt: DataLoader(eval_ds, batch_size=args.batch_size,
                       collate_fn=collators[mt], drop_last=True)
        for mt in MODEL_TYPES
    }

    # ── Dry-run: plan-only summary ─────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"  DRY RUN — no training, no eval")
        print(f"{'='*70}")
        for name, arch in arch_list:
            ne = (arch["num_encoder_layers"] + arch["num_decoder_layers"]) * 12 * arch["dim"] ** 2
            for tok_label, total_tokens in budgets:
                # Per-arch FLOPs sum: each model_type has its own multiplier
                # (DD=1.33×, SED=0.81×, DEC=1.0×). The naive 6NT line is kept
                # for comparison so users can see the magnitude of the bias
                # they would have had under the old logging.
                arch_flops = sum(
                    compute_flops_arch(mt, ne, total_tokens,
                                       enc=arch["num_encoder_layers"],
                                       dec=arch["num_decoder_layers"])
                    for mt in model_types)
                naive_flops = 6 * ne * total_tokens * len(model_types)
                est_sec = arch_flops / (gpu_tflops * 1e12 * 0.35)
                print(f"  {name:>5} × {tok_label:>5}: "
                      f"non_emb={ne:>11,}  arch_FLOPs={arch_flops:.2e}  "
                      f"(naive 6NT: {naive_flops:.2e})  "
                      f"est={est_sec/3600:.2f}h @35% MFU")
        return

    # ── Run each model type in an isolated subprocess ────────────────────
    # Each subprocess imports torch fresh, trains one model type's models
    # across all budgets, writes result JSONs, and exits — the OS reclaims
    # the CUDA allocator and any compiled-kernel caches. ~5-8s startup
    # overhead per model type is negligible compared to training time.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    sweep_t0 = time.time()

    # If we are inside a subprocess (--_subprocess-mt set), run just that one
    # model type directly (no further subprocess spawning).
    if args._subprocess_mt is not None:
        model_types = [args._subprocess_mt]

    # Spawn subprocesses when: compile is ON and we're in the parent process.
    # No-compile mode is safe in-process (no dynamo state accumulation).
    _use_subprocess = (use_compile and args._subprocess_mt is None)

    if _use_subprocess:
        for mt in model_types:
            print(f"\n{'#'*70}")
            print(f"  Spawning subprocess: {MODEL_TYPE_NAMES[mt]} ({mt})")
            print(f"{'#'*70}\n")
            # Re-invoke this script with all original args plus --_subprocess-mt.
            # Filter out --model-types and --_subprocess-mt from argv, we override.
            cmd = [sys.executable, str(Path(__file__).resolve())]
            i = 1
            skip_next = False
            while i < len(sys.argv):
                arg = sys.argv[i]
                if skip_next:
                    skip_next = False
                    i += 1
                    continue
                if arg in ("--model-types", "--_subprocess-mt"):
                    skip_next = True
                    i += 1
                    continue
                cmd.append(arg)
                i += 1
            cmd.extend(["--model-types", mt, "--_subprocess-mt", mt])

            t0 = time.time()
            result = subprocess.run(cmd)
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"\n  ERROR: subprocess for {mt} exited with code "
                      f"{result.returncode} after {elapsed:.0f}s")
                sys.exit(result.returncode)
            print(f"\n  Subprocess {mt} completed in {elapsed/60:.1f} min")

            # Collect results written by the subprocess.
            for tok_label, _ in budgets:
                result_path = out_dir / f"parallel_{tok_label}tok_{arch_tag}_results.json"
                if result_path.exists():
                    with open(result_path) as f:
                        all_results.setdefault(tok_label, {}).update(json.load(f))

    else:
        # In-process path: either inside a subprocess (single model type) or
        # compile is off (no dynamo state accumulation risk).

        # We need to know per-arch batch size before compile warmup if we want
        # the warmup to use the same shape as training. Auto-tune is per-arch
        # and per-model-type, but the dataloader is shared across model types
        # within an arch — so we tune each model and take the *min* batch size.
        for mt in model_types:
            needs_blocks = mt in ("dd", "sed")
            print(f"\n{'#'*70}")
            print(f"  Model type: {MODEL_TYPE_NAMES[mt]} ({mt})")
            print(f"{'#'*70}")

            # Build and compile models for this type
            print(f"\n  Building {len(arch_list)} models:")
            models_info = []
            for name, arch in arch_list:
                model = build_model(mt, arch, vocab_size, device, use_compile=use_compile)
                eager = getattr(model, "_orig_mod", model)
                actual_arch = dict(arch)
                if mt == "sed":
                    actual_arch["num_decoder_layers"] = len(eager.decoder_layers)
                ne = non_emb_param_count(model)
                print(f"    {name:>6}: dim={arch['dim']:>4}  non_emb={ne:>11,}  "
                      f"grad_ckpt={arch['dim'] >= 320}")
                models_info.append({
                    "name": name, "model_type": mt, "arch": actual_arch,
                    "model": model, "ne": ne, "needs_blocks": needs_blocks,
                })

            # Compile warmup (uses initial --batch-size; actual training batch is
            # set per-arch below). torch.compile re-traces if shapes change, but
            # with dynamic=False in eval and inductor's shape specialization, the
            # warmup at one shape still pays off because backward graph + key
            # kernels get cached.
            if use_compile:
                print(f"  Compiling...")
                compile_t0 = time.time()
                dummy_ids = torch.randint(0, vocab_size, (args.batch_size, SEQ_LEN), device=device)
                # SED's create_masks_ED indexes blocks per-batch, so it needs
                # blocks shape [batch_size]; DD wants split positions (1D).
                dd_blocks = torch.sort(torch.randperm(SEQ_LEN - 2, device=device)[:4] + 1)[0]
                sed_blocks = torch.full((args.batch_size,), SEQ_LEN, device=device, dtype=torch.long)
                for m in models_info:
                    blocks = sed_blocks if m["model_type"] == "sed" else dd_blocks
                    dummy_batch = {"input_ids": dummy_ids, "labels": dummy_ids.clone(),
                                   "blocks": blocks, "sft": False}
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model_forward(m["model"], dummy_batch, m["needs_blocks"])
                    out["loss"].backward()
                    for p in m["model"].parameters():
                        if p.grad is not None:
                            p.grad = None
                torch.cuda.synchronize()
                print(f"  Compiled in {time.time() - compile_t0:.1f}s")

            # Per-arch batch sizing. We auto-tune once per arch (probing all
            # model types, taking min) and store on the models_info entry so
            # train_one_budget uses the right batch_size.
            per_arch_bs = {}  # arch_name -> (batch_size, grad_accum)
            if args.auto_batch_size:
                print(f"\n  Auto-tuning batch size per arch (max={args.max_batch_size}):")
                for m in models_info:
                    bs = auto_tune_batch_size(
                        m["model"], vocab_size, SEQ_LEN, device,
                        m["needs_blocks"], max_bs=args.max_batch_size,
                        model_type=m["model_type"])
                    prior = per_arch_bs.get(m["name"])
                    per_arch_bs[m["name"]] = bs if prior is None else min(prior, bs)
                arch_ne = {m["name"]: m["ne"] for m in models_info}
                for name, bs in per_arch_bs.items():
                    tgt = (args.target_effective_batch
                           if args.target_effective_batch is not None
                           else target_effective_batch_for(arch_ne[name]))
                    ga = grad_accum_for(bs, tgt)
                    src = "CLI" if args.target_effective_batch is not None else "auto/N-cap"
                    print(f"    {name:>6}: batch_size={bs:>3}  grad_accum={ga:>3}  "
                          f"effective={bs*ga} seqs = {bs*ga*SEQ_LEN:,} tok/step  "
                          f"(target={tgt} {src})")
            else:
                for name, _ in arch_list:
                    per_arch_bs[name] = args.batch_size

            # Train all token budgets for this model type, per-arch.
            if args.auto_batch_size and len(set(per_arch_bs.values())) > 1:
                # Per-arch loop: re-create dataloader at each arch's batch size.
                for m in models_info:
                    bs = per_arch_bs[m["name"]]
                    arch_loaders = {
                        mt_: DataLoader(train_ds, batch_size=bs,
                                        collate_fn=collators[mt_], drop_last=True)
                        for mt_ in MODEL_TYPES
                    }
                    arch_eval_loaders = {
                        mt_: DataLoader(eval_ds, batch_size=bs,
                                        collate_fn=collators[mt_], drop_last=True)
                        for mt_ in MODEL_TYPES
                    }
                    for tok_label, total_tokens in budgets:
                        if args.target_effective_batch is not None:
                            tgt = args.target_effective_batch
                        else:
                            tgt = min(target_effective_batch_for(m["ne"]),
                                      step_floor_cap(total_tokens, SEQ_LEN,
                                                     args.min_optimizer_steps))
                        ga = grad_accum_for(bs, tgt)
                        tok_per_step = bs * ga * SEQ_LEN
                        total_steps = max(1, total_tokens // tok_per_step)
                        print(f"\n  {'='*66}")
                        print(f"    {mt} × {m['name']} × {tok_label} tokens  |  "
                              f"{total_steps} steps  |  bs={bs} ga={ga} (target={tgt})")
                        print(f"  {'='*66}\n")
                        base_cfg = {
                            "arch_set": args.arch_set, "token_set": args.token_set,
                            "boundary_strategy": args.boundary_strategy,
                            "batch_size": bs, "grad_accum": ga,
                            "total_tokens": total_tokens,
                            "tokens_per_step": tok_per_step,
                            "total_steps": total_steps,
                            "compile": use_compile,
                            "auto_batch_size": args.auto_batch_size,
                            "lr": (LR_OVERRIDE if LR_OVERRIDE is not None
                                   else base_lr_for(mt)),
                            "lr_source": ("CLI --peak-lr" if LR_OVERRIDE is not None
                                          else "configs/mup_tuned.json"
                                          if mt in TUNED_LRS else "BASE_LR fallback"),
                            "run_sft": args.run_sft,
                            "sft_tokens": args.sft_tokens if args.run_sft else None,
                            "eval_suite": args.eval_suite,
                            "mup_base_dim": MUP_BASE_DIM,
                            "mup_enabled": MUP_ENABLED,
                        }
                        wandb_runs = _init_wandb_runs(mt, [m], tok_label, base_cfg)
                        try:
                            results = train_one_budget(
                                [m], tok_label, total_tokens, bs, ga,
                                arch_loaders, arch_eval_loaders, device, gpu_tflops,
                                args.eval_batches,
                                output_dir=args.output_dir,
                                mid_eval_points=args.mid_eval_points,
                                run_full_eval=not args.skip_full_eval,
                                eval_suite=args.eval_suite,
                                eval_max_examples=args.eval_max_examples,
                                eval_data_file=args.eval_data_file,
                                tokenizer=tokenizer,
                                run_sft=args.run_sft,
                                sft_train_file=args.sft_train_file,
                                sft_eval_file=args.sft_eval_file,
                                sft_tokens=args.sft_tokens,
                                sft_lr=args.sft_lr,
                                sft_grad_accum=args.sft_grad_accum,
                                wandb_runs=wandb_runs,
                                save_checkpoints=args.save_checkpoints,
                                checkpoint_fractions=checkpoint_fractions,
                                hf_repo=args.hf_repo)
                        finally:
                            _finish_wandb_runs(wandb_runs)
                        all_results.setdefault(tok_label, {}).update(results)
                        out_path = out_dir / f"parallel_{tok_label}tok_{arch_tag}_results.json"
                        with open(out_path, "w") as f:
                            json.dump(all_results[tok_label], f, indent=2)
            else:
                # All archs share one batch size — co-resident path (faster).
                bs = min(per_arch_bs.values()) if per_arch_bs else args.batch_size
                if args.auto_batch_size:
                    run_loaders = {
                        mt_: DataLoader(train_ds, batch_size=bs,
                                        collate_fn=collators[mt_], drop_last=True)
                        for mt_ in MODEL_TYPES
                    }
                    run_eval_loaders = {
                        mt_: DataLoader(eval_ds, batch_size=bs,
                                        collate_fn=collators[mt_], drop_last=True)
                        for mt_ in MODEL_TYPES
                    }
                else:
                    run_loaders = train_loaders
                    run_eval_loaders = eval_loaders
                for tok_label, total_tokens in budgets:
                    if args.auto_batch_size:
                        if args.target_effective_batch is not None:
                            tgt = args.target_effective_batch
                        else:
                            n_cap = max(target_effective_batch_for(m["ne"])
                                        for m in models_info)
                            tgt = min(n_cap, step_floor_cap(total_tokens, SEQ_LEN,
                                                            args.min_optimizer_steps))
                        ga = grad_accum_for(bs, tgt)
                    else:
                        ga = args.grad_accum
                        tgt = bs * ga
                    tokens_per_step = bs * ga * SEQ_LEN
                    total_steps = max(1, total_tokens // tokens_per_step)
                    total_micro = total_steps * ga
                    print(f"\n  {'='*66}")
                    print(f"    {mt} × {tok_label} tokens  |  {total_steps} steps  |  "
                          f"{total_micro} micro-batches  |  bs={bs} ga={ga} "
                          f"(target={tgt})")
                    print(f"  {'='*66}\n")
                    base_cfg = {
                        "arch_set": args.arch_set, "token_set": args.token_set,
                        "boundary_strategy": args.boundary_strategy,
                        "batch_size": bs, "grad_accum": ga,
                        "total_tokens": total_tokens,
                        "tokens_per_step": tokens_per_step,
                        "total_steps": total_steps,
                        "compile": use_compile,
                        "auto_batch_size": args.auto_batch_size,
                        "lr": (LR_OVERRIDE if LR_OVERRIDE is not None
                               else base_lr_for(mt)),
                        "lr_source": ("CLI --peak-lr" if LR_OVERRIDE is not None
                                      else "configs/mup_tuned.json"
                                      if mt in TUNED_LRS else "BASE_LR fallback"),
                        "run_sft": args.run_sft,
                        "sft_tokens": args.sft_tokens if args.run_sft else None,
                        "eval_suite": args.eval_suite,
                        "mup_base_dim": MUP_BASE_DIM,
                        "mup_enabled": MUP_ENABLED,
                        "co_resident": True,
                    }
                    wandb_runs = _init_wandb_runs(mt, models_info, tok_label, base_cfg)
                    try:
                        results = train_one_budget(
                            models_info, tok_label, total_tokens, bs, ga,
                            run_loaders, run_eval_loaders, device, gpu_tflops,
                            args.eval_batches,
                            output_dir=args.output_dir,
                            mid_eval_points=args.mid_eval_points,
                            run_full_eval=not args.skip_full_eval,
                            eval_suite=args.eval_suite,
                            eval_max_examples=args.eval_max_examples,
                            eval_data_file=args.eval_data_file,
                            tokenizer=tokenizer,
                            run_sft=args.run_sft,
                            sft_train_file=args.sft_train_file,
                            sft_eval_file=args.sft_eval_file,
                            sft_tokens=args.sft_tokens,
                            sft_lr=args.sft_lr,
                            sft_grad_accum=args.sft_grad_accum,
                            wandb_runs=wandb_runs,
                            save_checkpoints=args.save_checkpoints,
                            checkpoint_fractions=checkpoint_fractions,
                            hf_repo=args.hf_repo)
                    finally:
                        _finish_wandb_runs(wandb_runs)
                    all_results.setdefault(tok_label, {}).update(results)
                    out_path = out_dir / f"parallel_{tok_label}tok_{arch_tag}_results.json"
                    with open(out_path, "w") as f:
                        json.dump(all_results[tok_label], f, indent=2)

            del models_info

    # ── Summary ─────────────────────────────────────────────────────────────
    sweep_time = time.time() - sweep_t0
    # Arch-tagged combined file so concurrent tier jobs don't stomp each other.
    # scripts/merge_parallel_results.py rebuilds the unified grid afterward.
    combined_path = out_dir / f"scaling_grid_results_{arch_tag}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Sweep complete in {sweep_time/60:.1f} min")
    print(f"  Results → {combined_path}")
    print(f"{'='*70}")

    # Print loss grids per model type
    for mt in model_types:
        print(f"\n  Eval loss grid — {MODEL_TYPE_NAMES[mt]} ({mt}):\n")
        header = f"  {'params':<8}" + "".join(f"  {l:>8}" for l, _ in budgets)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, _ in arch_list:
            row = f"  {name:<8}"
            display = f"{mt}_{name}"
            for tok_label, _ in budgets:
                if tok_label in all_results and display in all_results[tok_label]:
                    loss = all_results[tok_label][display]["final_eval_loss"]
                    row += f"  {loss:>8.4f}"
                else:
                    row += f"  {'—':>8}"
            print(row)

    print(f"\nTo merge across tier runs:")
    print(f"  python scripts/merge_parallel_results.py")
    print(f"To collect in scaling_laws.py format:")
    print(f"  python scripts/scaling_laws.py collect")


if __name__ == "__main__":
    main()
