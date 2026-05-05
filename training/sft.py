from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, PreTrainedTokenizerFast
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from training.pretrain import build_model
from training.dist_utils import init_distributed, get_world_size, get_rank, all_reduce_sum
from configs import TrainingConfig, build_config_from_dict

def eval(model, eval_dataloader, device):
    model.eval()
    with torch.no_grad():
        local_loss_times_tokens = torch.zeros(1, dtype=torch.float64, device=device)
        local_tokens = torch.zeros(1, dtype=torch.float64, device=device)
        for eval_batch in eval_dataloader:
            eval_batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in eval_batch.items()
            }
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                eval_outputs = model(**eval_batch)
            ntoks = (eval_batch["labels"] != -100).sum().to(dtype=torch.float64)
            local_loss_times_tokens += eval_outputs["loss"].detach().to(dtype=torch.float64) * ntoks
            local_tokens += ntoks

        loss_times_tokens = all_reduce_sum(local_loss_times_tokens).item()
        total_tokens = int(all_reduce_sum(local_tokens).item())
        avg_loss = loss_times_tokens / max(1, total_tokens)
        eval_ppl = float(torch.exp(torch.tensor(avg_loss)))

    model.train()
    return avg_loss, eval_ppl

def sft(cfg: TrainingConfig, verbose = False) -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    init_distributed()
    world_size = get_world_size()
    rank = get_rank()

    assert (cfg.grad_accum_steps % world_size == 0) and (cfg.grad_accum_steps >= world_size), "grad_accum_steps must be divisible by and geq than world_size"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_file)
    bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    collator = cfg.collator_cls(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                pad_token_id=pad_token_id, max_seq_len=cfg.seq_len)

    if cfg.input_model_name:
        ckpt = torch.load(cfg.input_model_name, map_location=device)
        sd = ckpt["state_dict"]
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        hparams = ckpt["hparams"]

        model = cfg.model_cls(**hparams)
        model.load_state_dict(sd)
    else:
        model, hparams = build_model(cfg, tokenizer.vocab_size)

    model = model.to(device)
    model = torch.compile(model, fullgraph=False, dynamic=False)
    ds = load_dataset("json", data_files=cfg.train_file, split="train")
    eval_ds = load_dataset("json", data_files=cfg.eval_file, split="train")

    num_lines = sum(1 for _ in open(cfg.train_file))
    total_minibatches = int(num_lines / cfg.batch_size) #Number of minibatches in the dataset
    num_accumulations = int(total_minibatches / cfg.grad_accum_steps) #Number of times we accumulate

    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 else None
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 else None

    num_workers = min(6, max(24 // world_size, 1))
    dataloader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        sampler=eval_sampler,
        shuffle=eval_sampler is None,
        num_workers=num_workers,
        prefetch_factor=4,
        collate_fn=collator,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

    #Base learning rate refers to 64 batch, 2 accum = 128
    #We need to scale by number of accumations
    scale = cfg.grad_accum_steps * cfg.batch_size / 128
    lr = cfg.lr * scale

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

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_accumulations * 0.05),   # 2% warmup
        num_training_steps=num_accumulations,
        lr_end=lr * 0.1,
        power=1.0
    )

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    model.train()
    os.makedirs(cfg.output_dir, exist_ok=True)
    optimizer.zero_grad()
    step = 0
    train_steps_arr = []
    train_losses = []

    eval_steps_arr = []
    eval_losses = []

    accumulation_steps = cfg.grad_accum_steps // world_size
    logging_steps = cfg.logging_steps // cfg.grad_accum_steps
    eval_steps = cfg.eval_steps // cfg.grad_accum_steps
    save_steps = cfg.save_steps // cfg.grad_accum_steps

    # Token-weighted accumulation — see training/pretrain.py for the full
    # rationale. SFT especially benefits since prompt tokens are masked
    # (-100) so n_valid varies a lot across microbatches.
    accum_loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    accum_n_valid  = torch.zeros(1, dtype=torch.float64, device=device)

    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Training", disable=(rank != 0 or verbose <= 1))
    for batch_idx, batch in enumerate(progress_bar):
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs["loss"]

        n_valid = (batch["labels"] != -100).sum()
        loss_sum = loss * n_valid
        loss_sum.backward()

        accum_loss_sum += loss_sum.detach().to(dtype=torch.float64)
        accum_n_valid  += n_valid.to(dtype=torch.float64)

        if (batch_idx + 1) % accumulation_steps == 0:
            total_n = all_reduce_sum(accum_n_valid).item()
            if total_n > 0:
                scale = float(world_size) / total_n
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)

            clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            total_loss = all_reduce_sum(accum_loss_sum).item()
            avg_step_loss = total_loss / max(total_n, 1)
            accum_loss_sum.zero_()
            accum_n_valid.zero_()

            if logging_steps > 0 and step % logging_steps == 0 and rank == 0:
                if verbose > 1:
                    tqdm.write(f"Step {step}/{num_accumulations} - loss: {avg_step_loss:.4f}")
                train_losses.append(avg_step_loss)
                train_steps_arr.append(step)

            if eval_steps > 0 and step % eval_steps == 0:
                avg_loss, eval_ppl = eval(model, eval_dataloader, device)
                if rank == 0:
                    eval_losses.append(avg_loss)
                    eval_steps_arr.append(step)
                    if verbose:
                        tqdm.write(f"Step {step}/{num_accumulations} - eval perplexity: {eval_ppl:.4f}")

            if save_steps > 0 and step % save_steps == 0:
                if rank == 0:
                    tqdm.write(f"Saving model at step {step}")
                    unwrapped = model.module if isinstance(model, DDP) else model
                    state_dict = unwrapped.state_dict()
                    torch.save(
                        {
                            "state_dict": state_dict,
                            "hparams": hparams,
                        },
                        f"{cfg.output_dir}/{cfg.output_file_name}_{step}.pt",
                    )
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
    avg_loss, eval_ppl = eval(model, eval_dataloader, device)
    if verbose > 0 and rank == 0:
        tqdm.write(f"End of training - eval perplexity: {eval_ppl:.4f}")
    if rank == 0:
        plt.plot(eval_steps_arr, eval_losses, label="Eval Losses")
        plt.plot(train_steps_arr, train_losses, label="Train Losses")
        plt.title("Losses")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(f"{cfg.output_dir}/{cfg.output_file_name}_eval_losses.png")
        plt.close()
        unwrapped = model.module if isinstance(model, DDP) else model
        torch.save(eval_losses, f"{cfg.output_dir}/{cfg.output_file_name}_eval_losses.pt")
        torch.save(
            {"state_dict": unwrapped.state_dict(), "hparams": hparams},
            f"{cfg.output_dir}/{cfg.output_file_name}.pt",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return eval_ppl,f"{cfg.output_dir}/{cfg.output_file_name}.pt"

def main():
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="runs/sft",
                        help="Config file name relative to configs/ (without .yaml)")
    args, overrides = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", f"{args.config_name}.yaml")
    cfg = OmegaConf.load(config_path)

    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            OmegaConf.update(cfg, key, value)

    training_cfg = build_config_from_dict(cfg)
    sft(training_cfg, verbose=3)


if __name__ == "__main__":
    main()
