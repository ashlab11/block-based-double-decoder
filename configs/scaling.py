"""Scaling law constants, architecture definitions, and interpolation utilities.

Shared by training/api.py, training/train_cli.py, and scripts/scaling_laws.py.
"""

import math

# ── Experiment grid ─────────────────────────────────────────────────────────

PARAM_LABELS = ["0.5M", "2.5M", "5M", "15M", "30M"]
TOKEN_LABELS = ["10M", "50M", "100M", "300M", "600M"]

PARAM_VALUES = {
    "0.5M":  500_000,
    "2.5M":  2_500_000,
    "5M":    5_000_000,
    "15M":   15_000_000,
    "30M":   30_000_000,
}

TOKEN_VALUES = {
    "10M":   10_000_000,
    "50M":   50_000_000,
    "100M":  100_000_000,
    "300M":  300_000_000,
    "600M":  600_000_000,
}

# ── Architecture configs ────────────────────────────────────────────────────
# Non-embedding params ≈ (enc + dec) * 12 * dim²
# dim must be a multiple of 64 (num_heads = dim // 64)
# Encoder:decoder layer ratio ≈ 2:1

ARCHITECTURES = {
    #             dim   enc  dec   actual_non_emb_params
    "0.5M":  dict(dim=64,  num_encoder_layers=7,  num_decoder_layers=3),   # 491,520
    "2.5M":  dict(dim=128, num_encoder_layers=9,  num_decoder_layers=4),   # 2,555,904
    "5M":    dict(dim=192, num_encoder_layers=8,  num_decoder_layers=4),   # 5,308,416
    "15M":   dict(dim=256, num_encoder_layers=13, num_decoder_layers=6),   # 14,942,208
    "30M":   dict(dim=384, num_encoder_layers=12, num_decoder_layers=5),   # 30,081,024
}

# ── Training constants ─────────────────────────────────────────────────────

SEQ_LEN = 2048
TARGET_EFFECTIVE_BATCH = 512
TOKENS_PER_STEP = TARGET_EFFECTIVE_BATCH * SEQ_LEN  # 1,048,576


# ── Multi-arch sweep grids (used by parallel_scaling and flop_matched_sweep) ──
# Lifted out of scripts/parallel_scaling.py so that small planning utilities
# (which don't need PyTorch/datasets) can import these without dragging in
# the full training stack.

ARCH_SETS = {
    "small": [
        ("0.6M",  dict(dim=64,  num_encoder_layers=8, num_decoder_layers=4)),
        ("2.4M",  dict(dim=128, num_encoder_layers=8, num_decoder_layers=4)),
        ("5.3M",  dict(dim=192, num_encoder_layers=8, num_decoder_layers=4)),
        ("14.7M", dict(dim=320, num_encoder_layers=8, num_decoder_layers=4)),
        ("28.9M", dict(dim=448, num_encoder_layers=8, num_decoder_layers=4)),
    ],
    "large": [
        ("5M",    dict(dim=192,  num_encoder_layers=8,  num_decoder_layers=4)),   # 5,308,416
        ("6.25M", dict(dim=192,  num_encoder_layers=10, num_decoder_layers=5)),   # 6,635,520  (2:1 ratio preserved via 10:5)
        ("12.5M", dict(dim=256,  num_encoder_layers=10, num_decoder_layers=5)),   # 11,796,480 (2:1 ratio preserved via 10:5)
        ("25M",   dict(dim=448,  num_encoder_layers=8,  num_decoder_layers=4)),   # 28,901,376
        ("50M",   dict(dim=576,  num_encoder_layers=8,  num_decoder_layers=4)),   # 47,775,744
        ("100M",  dict(dim=832,  num_encoder_layers=8,  num_decoder_layers=4)),   # 99,680,256 (num_heads=13)
        ("150M",  dict(dim=1024, num_encoder_layers=8,  num_decoder_layers=4)),
        ("300M",  dict(dim=1408, num_encoder_layers=8,  num_decoder_layers=4)),
    ],
}

TOKEN_SETS = {
    "small": [
        ("10M",  10_000_000),
        ("50M",  50_000_000),
        ("100M", 100_000_000),
        ("300M", 300_000_000),
        ("600M", 600_000_000),
    ],
    "large": [
        ("100M", 100_000_000),
        ("125M", 125_000_000),
        ("250M", 250_000_000),
        ("500M", 500_000_000),
        ("1B",   1_000_000_000),
        ("2B",   2_000_000_000),
        ("3B",   3_000_000_000),
        ("6B",   6_000_000_000),
    ],
}


# ── Helper functions ───────────────────────────────────────────────────────

def non_emb_params(dim, enc, dec):
    """Compute non-embedding parameter count (approximate, ignoring norms)."""
    return (enc + dec) * 12 * dim * dim


def compute_flops(non_emb_params, tokens):
    """Training FLOPs ≈ 6 * N * T (forward + backward).

    Generic decoder-only approximation. Use compute_flops_arch for an
    architecture-aware estimate that accounts for the encoder pass in DD/SED.
    """
    return 6 * non_emb_params * tokens


# ── Arch-aware FLOP accounting ─────────────────────────────────────────────
# The 6NT approximation is correct for decoder-only but understates DD (extra
# encoder pass + combo cross-attention) and overstates SED (decoder only emits
# ~15% target tokens). Per-arch multipliers vs DEC at the canonical
# enc=8/dec=4 ratio (see Item #1 analysis). Keep all three explicit so that
# changing layer ratios in one place doesn't silently desync the FLOP ledger.

DEFAULT_ENC_LAYERS = 8
DEFAULT_DEC_LAYERS = 4
SED_TARGET_DENSITY = 0.15  # T5 noise_density used by EDPretrainCollator


def _layer_ratio_factor(enc, dec):
    """Decoder-fraction = dec / (enc + dec). Used for cross-attn cost terms."""
    total = max(1, enc + dec)
    return dec / total


def compute_flops_dec(non_emb_params, tokens, enc=DEFAULT_ENC_LAYERS,
                      dec=DEFAULT_DEC_LAYERS):
    """DEC: every input token does one full pass through (enc+dec) causal
    layers; no cross-attn. Standard 6NT applies exactly."""
    return 6 * non_emb_params * tokens


def compute_flops_dd(non_emb_params, tokens, enc=DEFAULT_ENC_LAYERS,
                     dec=DEFAULT_DEC_LAYERS):
    """DD: encoder pass (enc layers) + decoder pass with combo attention
    that also attends to encoder positions. Combo cross-attn roughly doubles
    decoder cost vs a pure self-attn decoder.

    Effective FLOPs ≈ 6N·T · (1 + dec/(enc+dec))  [decoder fraction repeated
    once for the cross-attn term]. At enc=8, dec=4 → 1.33×.
    """
    dec_frac = _layer_ratio_factor(enc, dec)
    return 6 * non_emb_params * tokens * (1 + dec_frac)


def compute_flops_sed(non_emb_params, tokens, enc=DEFAULT_ENC_LAYERS,
                      dec=DEFAULT_DEC_LAYERS,
                      target_density=SED_TARGET_DENSITY):
    """SED: encoder runs over T input tokens; decoder runs over only
    target_density·T target tokens. Cross-attn keys come from the full
    encoder output, so the cross-attn term is sized by encoder T.

    Effective FLOPs ≈ 6N·T · [enc_frac + dec_frac·(target_density + enc_frac)]
    where enc_frac = enc/(enc+dec). At enc=8, dec=4, density=0.15 → ~0.81×.
    """
    enc_frac = 1.0 - _layer_ratio_factor(enc, dec)
    dec_frac = _layer_ratio_factor(enc, dec)
    multiplier = enc_frac + dec_frac * (target_density + enc_frac)
    return 6 * non_emb_params * tokens * multiplier


# Per-arch multiplier vs DEC at iso-tokens. Used by chinchilla_flop_match.
def arch_flop_multiplier(model_cls, enc=DEFAULT_ENC_LAYERS,
                         dec=DEFAULT_DEC_LAYERS,
                         target_density=SED_TARGET_DENSITY):
    """Returns k_x such that compute_flops_x(N, T) = k_x · 6NT."""
    if model_cls in ("dec", "DecoderOnly", "DecoderOnlyModel"):
        return 1.0
    if model_cls in ("dd", "Double_Decoder"):
        return 1.0 + _layer_ratio_factor(enc, dec)
    if model_cls in ("sed", "StandardEncDec"):
        enc_frac = 1.0 - _layer_ratio_factor(enc, dec)
        dec_frac = _layer_ratio_factor(enc, dec)
        return enc_frac + dec_frac * (target_density + enc_frac)
    raise ValueError(f"Unknown model_cls: {model_cls}")


def compute_flops_arch(model_cls, non_emb_params, tokens,
                       enc=DEFAULT_ENC_LAYERS, dec=DEFAULT_DEC_LAYERS,
                       target_density=SED_TARGET_DENSITY):
    """Dispatch FLOP computation by architecture string."""
    if model_cls in ("dec", "DecoderOnly", "DecoderOnlyModel"):
        return compute_flops_dec(non_emb_params, tokens, enc, dec)
    if model_cls in ("dd", "Double_Decoder"):
        return compute_flops_dd(non_emb_params, tokens, enc, dec)
    if model_cls in ("sed", "StandardEncDec"):
        return compute_flops_sed(non_emb_params, tokens, enc, dec, target_density)
    raise ValueError(f"Unknown model_cls: {model_cls}")


def chinchilla_flop_match(target_n, target_tokens, target_arch, source_arch,
                          chinchilla_ratio=20.0,
                          enc=DEFAULT_ENC_LAYERS, dec=DEFAULT_DEC_LAYERS):
    """Find (N, T) for source_arch that matches target_arch's FLOPs while
    holding T = chinchilla_ratio · N (Chinchilla-optimal frontier).

    Math:
        F_target = k_target · 6 · N_target · T_target
        F_source = k_source · 6 · N · T  with T = ρ·N
        Set equal: k_source · 6 · N · ρ·N = F_target
                   N² = F_target / (k_source · 6 · ρ)
                   N  = sqrt(F_target / (k_source · 6 · ρ))

    Equivalently: N_source = N_target · sqrt((k_target / k_source) · (T_target / (ρ·N_target)))
    For Chinchilla-anchored target (T_target = ρ·N_target), this collapses to
        N_source = N_target · sqrt(k_target / k_source)
        T_source = ρ · N_source

    Args:
        target_n: non-emb params of the reference run.
        target_tokens: token count of the reference run.
        target_arch: model_cls string of the reference (usually "dec").
        source_arch: model_cls string we want to FLOP-match against the target.
        chinchilla_ratio: T:N ratio to maintain for the source run (default 20).

    Returns:
        dict {"non_emb_params": int, "tokens": int, "target_flops": float,
              "source_flops": float, "matches_chinchilla": bool}
        — source_flops should equal target_flops to within rounding.
    """
    import math
    k_target = arch_flop_multiplier(target_arch, enc, dec)
    k_source = arch_flop_multiplier(source_arch, enc, dec)
    target_flops = k_target * 6 * target_n * target_tokens

    # Solve N² · k_source · 6 · ρ = F_target
    n_source = math.sqrt(target_flops / (k_source * 6 * chinchilla_ratio))
    t_source = chinchilla_ratio * n_source
    source_flops = k_source * 6 * n_source * t_source

    # Sanity: target's chinchilla ratio
    target_ratio = target_tokens / max(1, target_n)
    matches_chinchilla = abs(target_ratio - chinchilla_ratio) / chinchilla_ratio < 0.5

    return {
        "non_emb_params": int(round(n_source)),
        "tokens": int(round(t_source)),
        "target_flops": target_flops,
        "source_flops": source_flops,
        "target_chinchilla_ratio": target_ratio,
        "source_chinchilla_ratio": chinchilla_ratio,
        "matches_chinchilla": matches_chinchilla,
    }


def run_name_from_labels(plabel, tlabel):
    """Run name for a grid point: dd_0.5M_10Mtok."""
    return f"dd_{plabel}_{tlabel}tok"


def run_name_from_values(params, tokens):
    """Run name for an arbitrary point: dd_1000000p_40000000tok."""
    return f"dd_{params}p_{tokens}tok"


def lr_for_dim(dim):
    """Scale LR with model width: lr = 2e-3 * sqrt(64/dim)."""
    return round(0.002 * (64 / dim) ** 0.5, 6)


def eval_steps_for_tokens(total_tokens, batch_size=64, seq_len=SEQ_LEN):
    """~20 eval points per run.

    Returns step count in raw-batch units (pretrain.py divides by
    grad_accum_steps internally to convert to optimizer steps).
    """
    batches = total_tokens // (batch_size * seq_len)
    return max(4, batches // 20)


def save_steps_for_tokens(total_tokens, batch_size=64, seq_len=SEQ_LEN):
    """~5 checkpoints per run.

    Returns step count in raw-batch units (pretrain.py divides by
    grad_accum_steps internally to convert to optimizer steps).
    """
    batches = total_tokens // (batch_size * seq_len)
    return max(4, batches // 5)


# ── Architecture interpolation ─────────────────────────────────────────────

def interpolate_architecture(target_params):
    """Find the best architecture for a target non-embedding param count.

    Picks the layer configuration (enc, dec) from the nearest predefined
    architecture, then solves for dim (rounded to the nearest multiple of 64)
    to get as close to target_params as possible.

    Returns dict with dim, num_encoder_layers, num_decoder_layers,
    and actual_non_emb_params.
    """
    # Find nearest predefined architecture by param count (log-space distance,
    # since scaling laws are power-law relationships)
    best_label = None
    best_dist = float("inf")
    for label, arch in ARCHITECTURES.items():
        actual = non_emb_params(arch["dim"], arch["num_encoder_layers"], arch["num_decoder_layers"])
        dist = abs(math.log(actual) - math.log(target_params))
        if dist < best_dist:
            best_dist = dist
            best_label = label

    arch = ARCHITECTURES[best_label]
    enc = arch["num_encoder_layers"]
    dec = arch["num_decoder_layers"]
    total_layers = enc + dec

    # Solve: target = total_layers * 12 * dim²  →  dim = sqrt(target / (12 * total_layers))
    raw_dim = math.sqrt(target_params / (12 * total_layers))

    # Try both floor and ceil multiples of 64, pick whichever is closer
    # in log-space to the target param count
    dim_low = max(64, math.floor(raw_dim / 64) * 64)
    dim_high = dim_low + 64

    actual_low = non_emb_params(dim_low, enc, dec)
    actual_high = non_emb_params(dim_high, enc, dec)

    if abs(math.log(actual_low) - math.log(target_params)) <= abs(math.log(actual_high) - math.log(target_params)):
        dim = dim_low
    else:
        dim = dim_high

    actual = non_emb_params(dim, enc, dec)

    return {
        "dim": dim,
        "num_encoder_layers": enc,
        "num_decoder_layers": dec,
        "actual_non_emb_params": actual,
    }


def build_scaling_config(params, tokens, mup_base_dim=0, lr=None, run_name=None):
    """Build a config dict for arbitrary params/tokens, ready for build_config_from_dict.

    Args:
        params: target non-embedding parameter count (int)
        tokens: total training tokens (int)
        mup_base_dim: base width for μP scaling (0 = disabled)
        lr: override learning rate (None = auto from dim or μP base)
        run_name: override run name (None = auto from params/tokens)

    Returns:
        dict suitable for configs.build_config_from_dict()
    """
    arch = interpolate_architecture(params)
    name = run_name or run_name_from_values(params, tokens)

    # LR selection: explicit override > μP base LR > dim-scaled heuristic
    if lr is not None:
        effective_lr = lr
    elif mup_base_dim > 0:
        # With μP, use a fixed base LR (tuned at base width).
        # The optimizer handles per-param scaling internally.
        effective_lr = lr_for_dim(mup_base_dim)
    else:
        effective_lr = lr_for_dim(arch["dim"])

    return {
        "model_cls": "Double_Decoder",
        "dim": arch["dim"],
        "num_encoder_layers": arch["num_encoder_layers"],
        "num_decoder_layers": arch["num_decoder_layers"],
        "seq_len": SEQ_LEN,
        "shared": True,
        "logit_biases": False,
        "init_strategy": "xavier_uniform",
        "gradient_checkpointing": arch["dim"] >= 256,
        "use_compile": False,
        "mup_base_dim": mup_base_dim,
        "collator_cls": "DDPretrainCollator",
        "train_file": "data/Pretrain/slimpajama_6b_packed.jsonl",
        "eval_file": "data/Pretrain/slimpajama_6b_eval_packed.jsonl",
        "tokenizer_file": "tokenizer/tokenizer_32k.json",
        "auto_batch_size": True,
        "target_effective_batch": TARGET_EFFECTIVE_BATCH,
        "batch_size": 64,
        "grad_accum_steps": 1,
        "lr": effective_lr,
        "end_lr_ratio": 0.1,
        "total_tokens": tokens,
        "logging_steps": 10,
        "eval_steps": eval_steps_for_tokens(tokens, batch_size=64),
        "save_steps": save_steps_for_tokens(tokens, batch_size=64),
        "output_dir": "checkpoints/scaling",
        "output_file_name": name,
        "wandb_project": "dd-scaling-laws",
        "wandb_run_name": name,
        "wandb_entity": "block-based-double-decoders",
    }
