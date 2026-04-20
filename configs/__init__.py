"""Training configuration for block-based double-decoder experiments."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    """All parameters needed for a pretraining or SFT run.

    Model parameters (passed through to Double_Decoder / DecoderOnlyModel):
        dim, num_encoder_layers, num_decoder_layers, seq_len, mlp_dim,
        shared, logit_biases, init_strategy, label_pad_token_id

    Training parameters (consumed by the training loop):
        batch_size, grad_accum_steps, lr, end_lr_ratio,
        logging_steps, eval_steps, save_steps,
        train_file, eval_file, output_dir, output_file_name,
        tokenizer_file, model_cls, collator_cls, input_model_name
    """

    # ── Model architecture ───────────────────────────────────────────────
    dim: int = 256
    num_encoder_layers: int = 6
    num_decoder_layers: int = 3
    seq_len: int = 2048
    mlp_dim: int = None  # defaults to 4 * dim in model
    shared: bool = True
    logit_biases: bool = False
    init_strategy: str = "xavier_uniform"
    label_pad_token_id: int = -100

    # ── Training loop ────────────────────────────────────────────────────
    batch_size: int = 64
    grad_accum_steps: int = 2
    lr: float = 6e-4
    end_lr_ratio: float = 0.1
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 2000

    # ── Data & I/O ───────────────────────────────────────────────────────
    train_file: str = "data/Pretrain/slimpajama_packed.jsonl"
    eval_file: str = "data/Pretrain/slimpajama_eval_packed.jsonl"
    output_dir: str = "checkpoints"
    output_file_name: str = "model"
    tokenizer_file: str = "tokenizer/tokenizer.json"
    input_model_name: str = ""

    # ── Class references (set by build_config_from_dict) ─────────────────
    model_cls: Any = None
    collator_cls: Any = None

    # ── Sink tokens (unused for now, but referenced in build_model) ──────
    self_sinks: int = 0
    cross_sinks: int = 0


def build_config_from_dict(cfg_dict) -> TrainingConfig:
    """Build a TrainingConfig from a Hydra DictConfig or plain dict.

    Resolves model_cls and collator_cls from string names to actual classes.
    """
    from omegaconf import OmegaConf

    # Convert Hydra DictConfig to plain dict
    if not isinstance(cfg_dict, dict):
        cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)

    # Resolve model class
    model_cls_name = cfg_dict.pop("model_cls", "Double_Decoder")
    if model_cls_name == "Double_Decoder":
        from models.double_decoder import Double_Decoder
        model_cls = Double_Decoder
    elif model_cls_name == "DecoderOnlyModel":
        from models.decoder import DecoderOnlyModel
        model_cls = DecoderOnlyModel
    else:
        raise ValueError(f"Unknown model_cls: {model_cls_name}")

    # Resolve collator class
    collator_cls_name = cfg_dict.pop("collator_cls", "BasicPretrainCollator")
    if collator_cls_name == "BasicPretrainCollator":
        from collators.double_decoder.pretrain import BasicPretrainCollator
        collator_cls = BasicPretrainCollator
    elif collator_cls_name == "DecoderPretrainCollator":
        from collators.decoder.pretrain import DecoderPretrainCollator
        collator_cls = DecoderPretrainCollator
    elif collator_cls_name == "BasicSFTCollator":
        from collators.double_decoder.sft import BasicSFTCollator
        collator_cls = BasicSFTCollator
    elif collator_cls_name == "DecoderSFTCollator":
        from collators.decoder.sft import DecoderSFTCollator
        collator_cls = DecoderSFTCollator
    else:
        raise ValueError(f"Unknown collator_cls: {collator_cls_name}")

    # Build config, ignoring unknown keys
    valid_keys = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    filtered["model_cls"] = model_cls
    filtered["collator_cls"] = collator_cls

    return TrainingConfig(**filtered)
