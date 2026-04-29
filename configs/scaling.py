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


# ── Helper functions ───────────────────────────────────────────────────────

def non_emb_params(dim, enc, dec):
    """Compute non-embedding parameter count (approximate, ignoring norms)."""
    return (enc + dec) * 12 * dim * dim


def compute_flops(non_emb_params, tokens):
    """Training FLOPs ≈ 6 * N * T (forward + backward)."""
    return 6 * non_emb_params * tokens


def run_name_from_labels(plabel, tlabel):
    """Run name for a grid point: dd_0.5M_10Mtok."""
    return f"dd_{plabel}_{tlabel}tok"


def run_name_from_values(params, tokens):
    """Run name for an arbitrary point: dd_1000000p_40000000tok."""
    return f"dd_{params}p_{tokens}tok"


def lr_for_dim(dim):
    """Scale LR with model width: lr = 2e-3 * sqrt(64/dim)."""
    return round(0.002 * (64 / dim) ** 0.5, 6)


def eval_steps_for_tokens(total_tokens):
    """~20 eval points per run."""
    est_steps = total_tokens // TOKENS_PER_STEP
    return max(4, est_steps // 20)


def save_steps_for_tokens(total_tokens):
    """~5 checkpoints per run."""
    est_steps = total_tokens // TOKENS_PER_STEP
    return max(4, est_steps // 5)


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


def build_scaling_config(params, tokens):
    """Build a config dict for arbitrary params/tokens, ready for build_config_from_dict.

    Args:
        params: target non-embedding parameter count (int)
        tokens: total training tokens (int)

    Returns:
        dict suitable for configs.build_config_from_dict()
    """
    arch = interpolate_architecture(params)
    name = run_name_from_values(params, tokens)

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
        "collator_cls": "BasicPretrainCollator",
        "train_file": "data/Pretrain/slimpajama_6b_packed.jsonl",
        "eval_file": "data/Pretrain/slimpajama_6b_eval_packed.jsonl",
        "tokenizer_file": "tokenizer/tokenizer_32k.json",
        "auto_batch_size": True,
        "target_effective_batch": TARGET_EFFECTIVE_BATCH,
        "batch_size": 64,
        "grad_accum_steps": 1,
        "lr": lr_for_dim(arch["dim"]),
        "end_lr_ratio": 0.1,
        "total_tokens": tokens,
        "logging_steps": 4,
        "eval_steps": eval_steps_for_tokens(tokens),
        "save_steps": save_steps_for_tokens(tokens),
        "output_dir": "checkpoints/scaling",
        "output_file_name": name,
        "wandb_project": "dd-scaling-laws",
        "wandb_run_name": name,
        "wandb_entity": "block-based-double-decoders",
    }
