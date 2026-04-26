import os
import math
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, PreTrainedTokenizerFast
import torch
torch.set_float32_matmul_precision('high')
# Workaround for PyTorch 2.6.0 inductor bug: quantization pattern matcher
# crashes on non-quantized models. Disabling it is safe — we don't use quantization,
# and core inductor optimizations (kernel fusion, triton codegen) still apply.
import torch._inductor.config as _inductor_config
_inductor_config.pattern_matcher = False
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import time

from models.decoder import DecoderOnlyModel
from configs import TrainingConfig, build_config_from_dict
from training.dist_utils import init_distributed, get_world_size, get_rank, all_reduce_sum


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

        loss_times_tokens = all_reduce_sum(local_loss_times_tokens).item()
        total_tokens = int(all_reduce_sum(local_tokens).item())
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


def _try_batch_size(model, bs, seq_len, vocab_size, device):
    """Try a single forward+backward at the given batch size. Returns peak memory or -1 on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        input_ids = torch.randint(0, vocab_size, (bs, seq_len), device=device)
        labels = input_ids.clone()
        # DecoderOnlyModel doesn't accept blocks/sft kwargs
        import inspect
        sig = inspect.signature(model.forward)
        if 'blocks' in sig.parameters:
            blocks = torch.sort(torch.randperm(seq_len - 2, device=device)[:5] + 1)[0]
            kwargs = {"blocks": blocks, "sft": False}
        else:
            kwargs = {}
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels, **kwargs)
        out["loss"].backward()
        peak = torch.cuda.max_memory_allocated(device)
    except RuntimeError:
        peak = -1
    finally:
        # Clean up regardless of success or failure — use gc to handle
        # any partially-constructed tensors that might not be in local scope
        for p in model.parameters():
            p.grad = None
        import gc; gc.collect()
        torch.cuda.empty_cache()
    return peak


def _find_max_batch_size(model, seq_len, vocab_size, device, target_fraction=0.80):
    """Find the largest micro-batch that fits in GPU memory via linear ramp then binary search.

    Uses 80% of GPU memory to leave headroom for torch.compile overhead,
    optimizer states, and temporary buffers during training.

    Strategy: first ramp up by powers of 2 to find the approximate ceiling,
    then binary search within the last good range. This avoids starting the
    binary search at a huge batch size that could trigger a hard OOM.
    """
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * target_fraction)

    model.eval()

    # Phase 1: exponential ramp to find the ceiling
    last_good = 1
    probe = 1
    while probe <= 1024:
        peak = _try_batch_size(model, probe, seq_len, vocab_size, device)
        if peak < 0 or peak > target_mem:
            break
        last_good = probe
        probe *= 2

    # Phase 2: binary search between last_good and the failed probe
    lo, hi = last_good, min(probe, 1024)
    best = last_good
    while lo <= hi:
        mid = (lo + hi) // 2
        peak = _try_batch_size(model, mid, seq_len, vocab_size, device)
        if peak >= 0 and peak <= target_mem:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    model.train()
    for p in model.parameters():
        p.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return best


def pretrain(cfg: TrainingConfig, verbose=0) -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    init_distributed()
    world_size = get_world_size()
    rank = get_rank()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_file)
    bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    collator = cfg.collator_cls(bos_token_id=bos_token_id, eos_token_id=eos_token_id, max_seq_len=cfg.seq_len)

    model, hparams = build_model(cfg, tokenizer.vocab_size)
    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # ── Auto batch size ─────────────────────────────────────────────────
    if cfg.auto_batch_size:
        if rank == 0:
            print("Auto batch size: profiling GPU memory...")
        max_bs = _find_max_batch_size(model, cfg.seq_len, tokenizer.vocab_size, device)
        target = cfg.target_effective_batch
        if max_bs >= target:
            cfg.batch_size = target
            cfg.grad_accum_steps = 1
        else:
            cfg.batch_size = max_bs
            cfg.grad_accum_steps = max(1, target // max_bs)
        effective = cfg.batch_size * cfg.grad_accum_steps
        if rank == 0:
            total_gpu_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"Auto batch size: max_micro_batch={max_bs}, "
                  f"selected micro_batch={cfg.batch_size}, grad_accum={cfg.grad_accum_steps}, "
                  f"effective_batch={effective} seqs ({effective * cfg.seq_len:,} tokens/step) "
                  f"[GPU: {total_gpu_gb:.0f} GB]")

    assert (cfg.grad_accum_steps % world_size == 0) and (cfg.grad_accum_steps >= world_size), \
        "grad_accum_steps must be divisible by and geq than world_size"

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
        eager_model_for_watch = _get_eager_model(model)
        wandb_config = {
            **hparams,
            "lr": lr,
            "lr_end": lr_end,
            "batch_size": cfg.batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "total_tokens": cfg.total_tokens,
            "world_size": world_size,
            "seq_len": cfg.seq_len,
            "max_steps": cfg.max_steps,
            "warmup_steps": int(num_accumulations * 0.05),
            "total_accumulation_steps": num_accumulations,
            "tokens_per_step": cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len * world_size,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": world_size,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or None,
            entity=cfg.wandb_entity or None,
            config=wandb_config,
            resume="allow" if cfg.resume_from else None,
        )
        # Log gradient & parameter histograms every 500 steps
        wandb.watch(eager_model_for_watch, log="all", log_freq=500, log_graph=True)

    model.train()
    os.makedirs(cfg.output_dir, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    grad_norms = []
    train_steps_arr = []
    train_losses = []
    eval_steps_arr = []
    eval_losses = []

    # EMA loss for smoothed curve
    loss_ema = None
    loss_ema_alpha = 0.99

    steps_per_accum_per_gpu = cfg.grad_accum_steps // world_size
    logging_steps = cfg.logging_steps // cfg.grad_accum_steps
    eval_steps = cfg.eval_steps // cfg.grad_accum_steps
    save_steps = cfg.save_steps // cfg.grad_accum_steps

    max_steps = cfg.max_steps if cfg.max_steps > 0 else float('inf')
    # display_max is the actual number of steps we'll train for:
    # - If max_steps is set: use max_steps (early stopping / micro runs)
    # - Otherwise: use num_accumulations (token budget determines stop)
    display_max = max_steps if max_steps != float('inf') else num_accumulations
    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len * world_size
    step_start_time = time.time()
    training_start_time = time.time()
    tokens_seen = start_step * tokens_per_step

    # Track peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Skip batches if resuming
    skip_batches = start_step * steps_per_accum_per_gpu

    if rank == 0:
        print(f"Training for {display_max:,} steps ({tokens_per_step:,} tokens/step)")
        total_tok = display_max * tokens_per_step
        print(f"Total tokens to process: {total_tok:,} ({total_tok / 1e9:.2f}B)")

    # Accumulate micro-batch losses within an accumulation step
    accum_loss_sum = 0.0
    accum_loss_count = 0

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
        accum_loss_sum += loss_value
        accum_loss_count += 1
        (loss / steps_per_accum_per_gpu).backward()

        if (batch_idx + 1) % steps_per_accum_per_gpu == 0:
            grad = clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens_seen += tokens_per_step

            # Average loss across micro-batches in this accumulation
            avg_step_loss = accum_loss_sum / max(accum_loss_count, 1)
            accum_loss_sum = 0.0
            accum_loss_count = 0

            # Update EMA
            if loss_ema is None:
                loss_ema = avg_step_loss
            else:
                loss_ema = loss_ema_alpha * loss_ema + (1 - loss_ema_alpha) * avg_step_loss

            step_elapsed = time.time() - step_start_time
            current_lr = scheduler.get_last_lr()[0]

            # ── Per-step wandb logging (lightweight) ────────────────────
            if rank == 0 and use_wandb:
                elapsed_total = time.time() - training_start_time
                steps_done = step - start_step
                toks_per_sec = tokens_per_step / max(step_elapsed, 1e-6)
                avg_toks_per_sec = tokens_per_step * steps_done / max(elapsed_total, 1e-6)

                wandb.log({
                    # Core training metrics
                    "train/loss": avg_step_loss,
                    "train/loss_ema": loss_ema,
                    "train/perplexity": math.exp(min(avg_step_loss, 20)),  # clamp to avoid overflow
                    "train/grad_norm": grad.item(),
                    "train/lr": current_lr,
                    # Throughput
                    "throughput/tokens_per_sec": toks_per_sec,
                    "throughput/tokens_per_sec_avg": avg_toks_per_sec,
                    "throughput/step_time_ms": step_elapsed * 1000,
                    "throughput/samples_per_sec": cfg.batch_size * cfg.grad_accum_steps / max(step_elapsed, 1e-6),
                    # Progress
                    "progress/tokens_seen": tokens_seen,
                    "progress/tokens_seen_B": tokens_seen / 1e9,
                    "progress/pct_complete": 100 * step / display_max,
                    "progress/step": step,
                }, step=step)

            # ── Periodic heavy metrics (GPU memory, component grad norms) ─
            if rank == 0 and use_wandb and step % 50 == 0:
                heavy_metrics = {}

                # GPU memory
                if torch.cuda.is_available():
                    heavy_metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                    heavy_metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
                    heavy_metrics["gpu/memory_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    total_gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
                    heavy_metrics["gpu/memory_utilization_pct"] = 100 * torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory

                # Per-component gradient norms
                eager = _get_eager_model(model)
                component_grads = {}
                # Build component list dynamically — DecoderOnlyModel has
                # 'layers' instead of 'encoder_layers'/'decoder_layers'
                grad_components = [("embedding", eager.embedding)]
                if hasattr(eager, "encoder_layers"):
                    grad_components.append(("encoder", eager.encoder_layers))
                if hasattr(eager, "decoder_layers"):
                    grad_components.append(("decoder", eager.decoder_layers))
                if hasattr(eager, "layers"):
                    grad_components.append(("layers", eager.layers))
                if hasattr(eager, "encoder_norm"):
                    grad_components.append(("encoder_norm", eager.encoder_norm))
                grad_components.append(("output_norm", eager.output_norm))

                for comp_name, module in grad_components:
                    params_with_grad = [p for p in module.parameters() if p.grad is not None]
                    if params_with_grad:
                        total_norm = torch.norm(torch.stack([p.grad.detach().float().norm() for p in params_with_grad]))
                        heavy_metrics[f"grad_norm/{comp_name}"] = total_norm.item()

                # Per-component parameter norms (tracks weight growth/decay)
                weight_components = [("embedding", eager.embedding)]
                if hasattr(eager, "encoder_layers"):
                    weight_components.append(("encoder", eager.encoder_layers))
                if hasattr(eager, "decoder_layers"):
                    weight_components.append(("decoder", eager.decoder_layers))
                if hasattr(eager, "layers"):
                    weight_components.append(("layers", eager.layers))

                for comp_name, module in weight_components:
                    params = [p for p in module.parameters()]
                    if params:
                        total_norm = torch.norm(torch.stack([p.detach().float().norm() for p in params]))
                        heavy_metrics[f"weight_norm/{comp_name}"] = total_norm.item()

                wandb.log(heavy_metrics, step=step)

            # ── Console progress reporting (every step) ──────���─────────────
            if rank == 0:
                elapsed_total = time.time() - training_start_time
                steps_done = step - start_step
                avg_toks_per_sec = tokens_per_step * steps_done / max(elapsed_total, 1e-6)
                pct = 100 * step / display_max
                steps_remaining = display_max - step
                eta_sec = steps_remaining * (elapsed_total / max(steps_done, 1))

                def _fmt(s):
                    if s < 60: return f"{s:.0f}s"
                    elif s < 3600: return f"{s/60:.1f}m"
                    else: return f"{s/3600:.1f}h"

                print(f"  [{pct:5.1f}%] Step {step:>6,}/{display_max:,} | "
                      f"loss {avg_step_loss:.4f} (ema {loss_ema:.4f}) | grad {grad.item():.4f} | "
                      f"lr {current_lr:.2e} | {avg_toks_per_sec:,.0f} tok/s | "
                      f"elapsed {_fmt(elapsed_total)} | ETA {_fmt(eta_sec)}")

            step_start_time = time.time()

            if logging_steps > 0 and step % logging_steps == 0 and rank == 0:
                train_losses.append(avg_step_loss)
                train_steps_arr.append(step)
                grad_norms.append(grad.item())

            if eval_steps > 0 and step % eval_steps == 0:
                eval_start = time.time()
                avg_loss, eval_ppl = eval(model, eval_dataloader, device)
                eval_elapsed = time.time() - eval_start
                if rank == 0:
                    print(f"Eval time: {eval_elapsed:.2f}s - eval loss: {avg_loss:.4f} - eval ppl: {eval_ppl:.4f}")
                    eval_losses.append(avg_loss)
                    eval_steps_arr.append(step)
                    if use_wandb:
                        wandb.log({
                            "eval/loss": avg_loss,
                            "eval/perplexity": eval_ppl,
                            "eval/time_sec": eval_elapsed,
                            # Gap between train and eval (overfitting indicator)
                            "eval/train_eval_gap": avg_step_loss - avg_loss,
                        }, step=step)

            if save_steps > 0 and step % save_steps == 0:
                if rank == 0:
                    fname = _save_checkpoint(model, hparams, optimizer, scheduler, step, cfg, suffix=f"_{step}")
                    print(f"Saved checkpoint: {fname}")
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            # Stop when we've hit the target step count
            if step >= display_max:
                if rank == 0:
                    print(f"Reached {display_max:,} steps, stopping.")
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
            wandb.log({
                "eval/final_loss": avg_loss,
                "eval/final_perplexity": eval_ppl,
                "progress/total_tokens_seen": tokens_seen,
                "progress/total_steps": step,
                "progress/total_time_hours": (time.time() - training_start_time) / 3600,
            }, step=step)
            # Log final loss plot as artifact
            if train_steps_arr:
                wandb.log({
                    "plots/loss_curve": wandb.Image(f"{cfg.output_dir}/{cfg.output_file_name}_losses.png"),
                    "plots/grad_norms": wandb.Image(f"{cfg.output_dir}/{cfg.output_file_name}_grad_norms.png"),
                }, step=step)
            wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return eval_ppl, f"{cfg.output_dir}/{cfg.output_file_name}.pt"

def main():
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="runs/pretrain",
                        help="Config file name relative to configs/ (without .yaml)")
    args, overrides = parser.parse_known_args()

    # Load config YAML directly — no Hydra, no working-directory surprises
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", f"{args.config_name}.yaml")
    cfg = OmegaConf.load(config_path)

    # Apply any key=value overrides from the command line
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            OmegaConf.update(cfg, key, value)

    training_cfg = build_config_from_dict(cfg)
    pretrain(training_cfg, verbose=1)


if __name__ == "__main__":
    main()
