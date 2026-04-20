import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, PreTrainedTokenizerFast
from torch.optim.lr_scheduler import LambdaLR
import torch
torch.set_float32_matmul_precision('high')  
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
from tqdm import tqdm
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
                       "self_sinks", "cross_sinks"}
    
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


def pretrain(cfg: TrainingConfig, verbose = 0) -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    _init_distributed()
    world_size = _get_world_size()
    rank = _get_rank()

    assert (cfg.grad_accum_steps % world_size == 0) and (cfg.grad_accum_steps >= world_size), "grad_accum_steps must be divisible by and geq than world_size"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_file)
    bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    collator = cfg.collator_cls(bos_token_id=bos_token_id, eos_token_id=eos_token_id, max_seq_len=cfg.seq_len)

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

    if world_size > 1:
        train_sampler.set_epoch(0)
    
    num_workers = min(6, max(24 // world_size, 1))
    dataloader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        prefetch_factor=4,
        persistent_workers=num_workers > 0,
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

    #Base learning rate refers to 64 batch, 2 accum = 128
    #We need to scale by number of accumulations
    scale = cfg.grad_accum_steps * cfg.batch_size / 128
    lr = cfg.lr * scale
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

    #Old scheduler, want to test out this new one
    lr_end = lr * cfg.end_lr_ratio if lr > 1e-4 else lr * 0.99 # just for use when using 4e-5
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_accumulations * 0.05),   # 5% warmup
        num_training_steps=num_accumulations,
        lr_end=lr_end,                       # ~10% floor
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
    optimizer.zero_grad(set_to_none=True)
    
    step = 0
    grad_norms = []
    
    train_steps_arr = []
    train_losses = []

    eval_steps_arr = []
    eval_losses = []

    steps_per_accum_per_gpu = cfg.grad_accum_steps // world_size
    logging_steps = cfg.logging_steps // cfg.grad_accum_steps
    eval_steps = cfg.eval_steps // cfg.grad_accum_steps
    save_steps = cfg.save_steps // cfg.grad_accum_steps
    
    for batch_idx, batch in enumerate(dataloader):
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        #Create list 
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs["loss"]

        loss_value = loss.detach().item()        
        (loss / steps_per_accum_per_gpu).backward()

        if (batch_idx + 1) % steps_per_accum_per_gpu == 0:
            grad = clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if step % 100 == 0:
                print(f"Grad norm at accumulation step {step}: {grad.item()}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if logging_steps > 0 and step % logging_steps == 0 and rank == 0:
                if verbose > 1:
                    print(f"Step {step}/{num_accumulations} - loss: {loss_value:.4f}")
                train_losses.append(loss_value)
                train_steps_arr.append(step)
                grad_norms.append(grad.item())

            if eval_steps > 0 and step % eval_steps == 0:
                start_time = time.time()
                avg_loss, eval_ppl = eval(model, eval_dataloader, device)
                print(f"Eval time: {time.time() - start_time:.2f} seconds")
                if rank == 0:
                    eval_losses.append(avg_loss)
                    eval_steps_arr.append(step)
                    if verbose >= 1:
                        print(f"Step {step}/{num_accumulations} - eval perplexity: {eval_ppl:.4f}")

            if save_steps > 0 and step % save_steps == 0:
                if rank == 0:
                    print(f"Saving model at step {step}")
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
        print(f"End of training - eval perplexity: {eval_ppl:.4f}")
    if rank == 0:
        plt.plot(eval_steps_arr, eval_losses, label="Eval Losses")
        plt.plot(train_steps_arr, train_losses, label="Train Losses")
        plt.title("Losses")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(f"{cfg.output_dir}/{cfg.output_file_name}_eval_losses.png")
        plt.close()
        
        plt.plot(train_steps_arr, grad_norms, label="Grad Norms")
        plt.title("Grad Norms")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Grad Norm")
        plt.savefig(f"{cfg.output_dir}/{cfg.output_file_name}_grad_norms.png")
        plt.close()

        unwrapped = model.module if isinstance(model, DDP) else model
        torch.save(eval_losses, f"{cfg.output_dir}/{cfg.output_file_name}_eval_losses.pt")
        torch.save(
            {
                "state_dict": unwrapped.state_dict(),
                "hparams": hparams,
            },
            f"{cfg.output_dir}/{cfg.output_file_name}.pt",
        )
        
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
