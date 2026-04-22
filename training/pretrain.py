import os
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, PreTrainedTokenizerFast
import torch
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import time

from models.double_decoder import Double_Decoder
from models.decoder import DecoderOnlyModel
from configs import TrainingConfig, build_config_from_dict

def _init_distributed() -> bool:
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend='nccl')
        return True
    return dist.is_available() and dist.is_initialized()


def _get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _get_eager_model(model):
    """Unwrap DDP and torch.compile to get the eager model for eval."""
    m = model.module if isinstance(model, DDP) else model
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def eval(model, eval_dataloader, device):
    eager_model = _get_eager_model(model)
    eager_model.eval()
    with torch.no_grad():
        local_loss_times_tokens = torch.zeros(1, dtype=torch.float64, device=device)
        local_tokens = torch.zeros(1, dtype=torch.float64, device=device)
        for eval_batch in eval_dataloader:
            eval_batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in eval_batch.items()
            }
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                eval_outputs = eager_model(**eval_batch)
            ntoks = (eval_batch["labels"] != -100).sum().to(dtype=torch.float64)
            local_loss_times_tokens += eval_outputs["loss"].detach().to(dtype=torch.float64) * ntoks
            local_tokens += ntoks

        loss_times_tokens = _all_reduce_sum(local_loss_times_tokens).item()
        total_tokens = int(_all_reduce_sum(local_tokens).item())
        avg_loss = loss_times_tokens / max(1, total_tokens)
        eval_ppl = float(torch.exp(torch.tensor(avg_loss)))

    eager_model.train()
    return avg_loss, eval_ppl


def build_model(cfg: TrainingConfig, vocab_size: int):
    non_model_attrs = {"batch_size", "grad_accum_steps", "logging_steps", "eval_steps", "save_steps",
                       "lr", "end_lr_ratio", "tokenizer_file", "output_dir", "output_file_name",
                       "train_file", "eval_file", "collator_cls", "model_cls", "input_model_name",
                       "self_sinks", "cross_sinks", "total_tokens", "max_steps",
                       "wandb_project", "wandb_run_name", "wandb_entity", "resume_from"}

    cfg_dict = dict(vars(cfg)) #Gets all setattr values
    for k in TrainingConfig.__dict__.keys():
        if not k.startswith('_') and k not in cfg_dict:
            cfg_dict[k] = getattr(cfg, k)

    hparams = {k: v for k, v in cfg_dict.items() if k not in non_model_attrs}
    hparams["vocab_size"] = vocab_size
    hparams["num_heads"] = cfg.dim // 64

    if issubclass(cfg.model_cls, DecoderOnlyModel):
        hparams["num_layers"] = cfg.num_decoder_layers + cfg.num_encoder_layers

    return cfg.model_cls(**hparams), hparams


def _compute_num_accumulations(cfg: TrainingConfig, world_size: int, fallback_num_lines: int = 0):
    """Compute total accumulation steps from total_tokens or fallback to file line count."""
    if cfg.total_tokens > 0:
        total_sequences = cfg.total_tokens // cfg.seq_len
        total_minibatches = total_sequences // cfg.batch_size
        return total_minibatches // cfg.grad_accum_steps
    else:
        # Fallback: count lines in file (only for small datasets)
        num_lines = fallback_num_lines or sum(1 for _ in open(cfg.train_file))
        total_minibatches = num_lines // cfg.batch_size
        return total_minibatches // cfg.grad_accum_steps


def _save_checkpoint(model, hparams, optimizer, scheduler, step, cfg, suffix=""):
    """Save full checkpoint with model, optimizer, scheduler, and step."""
    unwrapped = model.module if isinstance(model, DDP) else model
    # Get through torch.compile wrapper if present
    state_model = unwrapped._orig_mod if hasattr(unwrapped, '_orig_mod') else unwrapped
    filename = f"{cfg.output_dir}/{cfg.output_file_name}{suffix}.pt"
    torch.save(
        {
            "state_dict": state_model.state_dict(),
            "hparams": hparams,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
        },
        filename,
    )
    return filename


def pretrain(cfg: TrainingConfig, verbose=0) -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    _init_distributed()
    world_size = _get_world_size()
    rank = _get_rank()

    assert (cfg.grad_accum_steps % world_size == 0) and (cfg.grad_accum_steps >= world_size), \
        "grad_accum_steps must be divisible by and geq than world_size"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_file)
    bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    collator = cfg.collator_cls(bos_token_id=bos_token_id, eos_token_id=eos_token_id, max_seq_len=cfg.seq_len)

    model, hparams = build_model(cfg, tokenizer.vocab_size)
    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    model = torch.compile(model, fullgraph=False, dynamic=False)

    # ── Data loading ─────────────────────────────────────────────────────
    use_streaming = cfg.total_tokens > 0

    if use_streaming:
        ds = load_dataset("json", data_files=cfg.train_file, split="train", streaming=True)
        if world_size > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
    else:
        ds = load_dataset("json", data_files=cfg.train_file, split="train")

    eval_ds = load_dataset("json", data_files=cfg.eval_file, split="train")

    num_accumulations = _compute_num_accumulations(cfg, world_size)

    if not use_streaming:
        train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 else None
        if world_size > 1:
            train_sampler.set_epoch(0)
    else:
        train_sampler = None

    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 else None

    num_workers = min(6, max(24 // world_size, 1))
    dataloader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and not use_streaming),
        num_workers=num_workers if not use_streaming else 0,
        prefetch_factor=4 if not use_streaming else None,
        persistent_workers=(num_workers > 0 and not use_streaming),
        collate_fn=collator,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=4,
        collate_fn=collator,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.enable_flash_sdp(True)

    # ── Optimizer & scheduler ────────────────────────────────────────────
    lr = cfg.lr
    if rank == 0:
        print(f"Learning rate: {lr}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(k in n.lower() for k in ["bias","norm","ln","layernorm"]) else decay).append(p)

    optimizer = AdamW(
        [{"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True
    )

    lr_end = lr * cfg.end_lr_ratio
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_accumulations * 0.05),   # 5% warmup
        num_training_steps=num_accumulations,
        lr_end=lr_end,
        power=1.0
    )

    # ── Checkpoint resumption ────────────────────────────────────────────
    start_step = 0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        if rank == 0:
            print(f"Resuming from checkpoint: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        # Load into the unwrapped model (before DDP)
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(ckpt["state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "step" in ckpt:
            start_step = ckpt["step"]
        del ckpt
        if rank == 0:
            print(f"Resumed at step {start_step}")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    # ── Wandb ────────────────────────────────────────────────────────────
    use_wandb = bool(cfg.wandb_project) and rank == 0
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or None,
            entity=cfg.wandb_entity or None,
            config={**hparams, "lr": lr, "batch_size": cfg.batch_size,
                    "grad_accum_steps": cfg.grad_accum_steps, "total_tokens": cfg.total_tokens},
            resume="allow" if cfg.resume_from else None,
        )

    model.train()
    os.makedirs(cfg.output_dir, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    grad_norms = []
    train_steps_arr = []
    train_losses = []
    eval_steps_arr = []
    eval_losses = []

    steps_per_accum_per_gpu = cfg.grad_accum_steps // world_size
    logging_steps = cfg.logging_steps // cfg.grad_accum_steps
    eval_steps = cfg.eval_steps // cfg.grad_accum_steps
    save_steps = cfg.save_steps // cfg.grad_accum_steps

    max_steps = cfg.max_steps if cfg.max_steps > 0 else float('inf')
    effective_max = min(max_steps, num_accumulations)
    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len * world_size
    step_start_time = time.time()
    training_start_time = time.time()

    # Skip batches if resuming
    skip_batches = start_step * steps_per_accum_per_gpu

    if rank == 0:
        print(f"Training for {effective_max:,} accumulation steps ({tokens_per_step:,} tokens/step)")
        total_tok = effective_max * tokens_per_step
        print(f"Total tokens to process: {total_tok:,} ({total_tok / 1e9:.2f}B)")
        if cfg.max_steps > 0:
            print(f"Early stopping after {cfg.max_steps} steps")

    for batch_idx, batch in enumerate(dataloader):
        # Skip already-processed batches when resuming
        if batch_idx < skip_batches:
            continue

        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs["loss"]

        loss_value = loss.detach().item()
        (loss / steps_per_accum_per_gpu).backward()

        if (batch_idx + 1) % steps_per_accum_per_gpu == 0:
            grad = clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # ── Progress reporting ───────────────────────────────────────
            if rank == 0 and step % 10 == 0:
                elapsed_total = time.time() - training_start_time
                steps_done = step - start_step
                step_elapsed = time.time() - step_start_time
                toks_per_sec = tokens_per_step * steps_done / max(elapsed_total, 1e-6)
                pct = 100 * step / effective_max
                steps_remaining = effective_max - step
                eta_sec = steps_remaining * (elapsed_total / max(steps_done, 1))

                def _fmt(s):
                    if s < 60: return f"{s:.0f}s"
                    elif s < 3600: return f"{s/60:.1f}m"
                    else: return f"{s/3600:.1f}h"

                current_lr = scheduler.get_last_lr()[0]
                print(f"  [{pct:5.1f}%] Step {step:>6,}/{effective_max:,} | "
                      f"loss {loss_value:.4f} | grad {grad.item():.4f} | "
                      f"lr {current_lr:.2e} | {toks_per_sec:,.0f} tok/s | "
                      f"elapsed {_fmt(elapsed_total)} | ETA {_fmt(eta_sec)}")

            step_start_time = time.time()

            if logging_steps > 0 and step % logging_steps == 0 and rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                train_losses.append(loss_value)
                train_steps_arr.append(step)
                grad_norms.append(grad.item())
                if use_wandb:
                    elapsed_total = time.time() - training_start_time
                    steps_done = step - start_step
                    toks_per_sec = tokens_per_step * steps_done / max(elapsed_total, 1e-6)
                    wandb.log({
                        "train/loss": loss_value,
                        "train/grad_norm": grad.item(),
                        "train/lr": current_lr,
                        "train/step": step,
                        "train/tokens_per_sec": toks_per_sec,
                    }, step=step)

            if eval_steps > 0 and step % eval_steps == 0:
                eval_start = time.time()
                avg_loss, eval_ppl = eval(model, eval_dataloader, device)
                if rank == 0:
                    print(f"Eval time: {time.time() - eval_start:.2f}s - eval ppl: {eval_ppl:.4f}")
                    eval_losses.append(avg_loss)
                    eval_steps_arr.append(step)
                    if use_wandb:
                        wandb.log({
                            "eval/loss": avg_loss,
                            "eval/perplexity": eval_ppl,
                        }, step=step)

            if save_steps > 0 and step % save_steps == 0:
                if rank == 0:
                    fname = _save_checkpoint(model, hparams, optimizer, scheduler, step, cfg, suffix=f"_{step}")
                    print(f"Saved checkpoint: {fname}")
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            # Early stopping for micro runs
            if step >= max_steps:
                if rank == 0:
                    print(f"Reached max_steps={cfg.max_steps}, stopping.")
                break

    # ── End of training ──────────────────────────────────────────────────
    avg_loss, eval_ppl = eval(model, eval_dataloader, device)
    if verbose > 0 and rank == 0:
        print(f"End of training - eval perplexity: {eval_ppl:.4f}")

    if rank == 0:
        # Save plots
        if train_steps_arr:
            plt.plot(eval_steps_arr, eval_losses, label="Eval Losses")
            plt.plot(train_steps_arr, train_losses, label="Train Losses")
            plt.title("Losses")
            plt.legend()
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.savefig(f"{cfg.output_dir}/{cfg.output_file_name}_losses.png")
            plt.close()

            plt.plot(train_steps_arr, grad_norms, label="Grad Norms")
            plt.title("Grad Norms")
            plt.legend()
            plt.xlabel("Step")
            plt.ylabel("Grad Norm")
            plt.savefig(f"{cfg.output_dir}/{cfg.output_file_name}_grad_norms.png")
            plt.close()

        # Save final checkpoint
        fname = _save_checkpoint(model, hparams, optimizer, scheduler, step, cfg)
        print(f"Final checkpoint: {fname}")

        if use_wandb:
            wandb.log({"eval/final_perplexity": eval_ppl}, step=step)
            wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return eval_ppl, f"{cfg.output_dir}/{cfg.output_file_name}.pt"

@hydra.main(version_base=None, config_path="../configs", config_name="runs/pretrain")
def main(cfg: DictConfig):
    training_cfg = build_config_from_dict(cfg)
    pretrain(training_cfg, verbose=1)


if __name__ == "__main__":
    main()
