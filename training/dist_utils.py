"""Distributed training utilities shared by pretrain and sft."""

import torch
import torch.distributed as dist
import os


def init_distributed() -> bool:
    """Initialize the NCCL process group if RANK/WORLD_SIZE env vars are set."""
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend='nccl')
        return True
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
