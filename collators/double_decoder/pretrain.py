import random
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

# Resolve repo root once so prompt_style strategy can find the cached histogram
# regardless of cwd. Walks up from .../collators/double_decoder/pretrain.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_HIST_PATH = _REPO_ROOT / "data" / "SFT" / "prompt_length_hist.npy"

# Boundary-sampling strategies for DD pretraining. The default keeps existing
# behavior bit-for-bit; new strategies are opt-in via the `boundary_strategy`
# constructor arg, exposed by parallel_scaling.py as --boundary-strategy.
BOUNDARY_STRATEGIES = (
    "random_uniform",  # current behavior: 2..max_blocks splits at random positions
    "logspace",        # use the existing self.possible_blocks log-spaced grid
    "single_middle",   # one boundary at seq_len // 2 (degenerate, ablation control)
    "prompt_style",    # one boundary sampled from the cached SFT prompt-length histogram
)

def _get_rank():
    try:
        return dist.get_rank() if dist.is_initialized() else 0
    except:
        return 0

def _get_rng(global_seed):
    wi = torch.utils.data.get_worker_info()
    wid = 0 if wi is None else wi.id
    rank = _get_rank()
    if global_seed is None:
        # nondeterministic but different per rank/worker
        ss = np.random.SeedSequence([np.random.SeedSequence().entropy, rank, wid])
    else:
        # fully reproducible across runs
        ss = np.random.SeedSequence([global_seed, rank, wid])
    return np.random.default_rng(ss)


# Module-level cache so we load the histogram once, not per-batch. Keyed by
# the file's mtime so a rebuilt histogram gets picked up on the next batch.
_PROMPT_HIST_CACHE = {"mtime": None, "samples": None}


def _load_prompt_length_samples():
    """Read the prompt-length histogram saved by data/SFT/build_prompt_hist.py.

    Returns a 1D numpy array of integer prompt lengths (sample-able with rng.choice).
    Returns None if the file doesn't exist — caller falls back to random_uniform.
    """
    if not _PROMPT_HIST_PATH.exists():
        return None
    mtime = _PROMPT_HIST_PATH.stat().st_mtime
    if _PROMPT_HIST_CACHE["mtime"] == mtime and _PROMPT_HIST_CACHE["samples"] is not None:
        return _PROMPT_HIST_CACHE["samples"]
    samples = np.load(_PROMPT_HIST_PATH).astype(np.int64)
    _PROMPT_HIST_CACHE["mtime"] = mtime
    _PROMPT_HIST_CACHE["samples"] = samples
    return samples


class DDPretrainCollator:
    """
    Data Collator for Double Decoder
    """
    def __init__(
        self,
        max_blocks = 8,
        max_seq_len = 1536,
        bos_token_id=1,
        eos_token_id=2,
        label_pad_token_id=-100,
        global_seed=None,
        boundary_strategy="random_uniform",
        **kwargs
    ):
        self.max_blocks = max_blocks
        #Creates log-spaced blocks to maximally vary lengths
        self.possible_blocks = np.logspace(0, np.log10(max_blocks), int(np.log2(max_blocks) + 1)).round().astype(int)
        self.label_pad_token_id = label_pad_token_id
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.global_seed = global_seed
        if boundary_strategy not in BOUNDARY_STRATEGIES:
            raise ValueError(
                f"boundary_strategy={boundary_strategy!r} not in {BOUNDARY_STRATEGIES}")
        self.boundary_strategy = boundary_strategy
        # Validate prompt_style data is present at construction time so we
        # fail fast at startup, not 500 batches into training.
        if boundary_strategy == "prompt_style" and _load_prompt_length_samples() is None:
            raise FileNotFoundError(
                f"boundary_strategy=prompt_style requires {_PROMPT_HIST_PATH}. "
                f"Build it with: python data/SFT/build_prompt_hist.py")
        self.rng = None # lazily created within the _init_

    def _sample_blocks(self):
        """Dispatch on self.boundary_strategy. Returns a sorted 1D torch.LongTensor
        of split positions in (1, max_seq_len-1)."""
        if self.boundary_strategy == "random_uniform":
            block_num = self.rng.integers(2, max(3, self.max_blocks))
            blocks = self.rng.choice(np.arange(2, self.max_seq_len - 1),
                                     size=block_num, replace=False)
        elif self.boundary_strategy == "logspace":
            # Sample a count from the precomputed log-spaced grid (same spirit
            # as the original `possible_blocks` field that was defined but
            # never used). Block positions are still uniform within [2, L-1].
            block_num = int(self.rng.choice(self.possible_blocks))
            block_num = max(2, min(block_num, self.max_seq_len - 2))
            blocks = self.rng.choice(np.arange(2, self.max_seq_len - 1),
                                     size=block_num, replace=False)
        elif self.boundary_strategy == "single_middle":
            blocks = np.array([self.max_seq_len // 2], dtype=np.int64)
        elif self.boundary_strategy == "prompt_style":
            samples = _load_prompt_length_samples()
            # Clip to the legal split range. Most SFT prompts are short
            # relative to seq_len so clipping mostly affects the long tail.
            valid = samples[(samples >= 2) & (samples < self.max_seq_len - 1)]
            if valid.size == 0:
                # All prompts longer than seq_len-1: degrade to single_middle
                # so the run doesn't crash. Should be rare (would mean the
                # SFT corpus has no examples shorter than the model's ctx).
                blocks = np.array([self.max_seq_len // 2], dtype=np.int64)
            else:
                pos = int(self.rng.choice(valid))
                blocks = np.array([pos], dtype=np.int64)
        else:
            raise AssertionError(f"unhandled strategy {self.boundary_strategy}")

        return torch.sort(torch.as_tensor(blocks, dtype=torch.long))[0]

    def __call__(self, batch):
        """
        batch: List[Dict[str, List[int]]]  each dict must have an "input_ids" key.
        Assumes that input_ids are already packed.
        Returns a dict with:
          - "encoder_input_ids": padded input sequences
          - "decoder_input_ids": padded input sequences
          - "labels":   padded label sequences
          - "blocks":  listing where to split in the combo attention
        """

        # 1) Get inputs and labels
        inputs = [example["input_ids"] for example in batch]
        labels = [example["input_ids"][1:] + [self.label_pad_token_id] for example in batch]

        #We don't want it to predict <bos> (when input is <eos>) OR the token after <bos> (when input is <bos>)
        labels = [
            [
                (label if input_id not in [self.bos_token_id, self.eos_token_id] else self.label_pad_token_id)
                for input_id, label in zip(input_seq, label_seq)
            ]
            for input_seq, label_seq in zip(inputs, labels)
        ]

        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)

        # 2) Create blocks
        if self.rng is None:
            self.rng = _get_rng(self.global_seed)

        blocks = self._sample_blocks()

        # 3) Return
        return {
            "input_ids": inputs,
            "labels": labels,
            "blocks": blocks,
            "sft": False,
        }
