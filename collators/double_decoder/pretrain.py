import random
import torch
import torch.distributed as dist
import numpy as np

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
        self.rng = None # lazily created within the _init_
        
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
        
        block_num = self.rng.integers(2, max(3, self.max_blocks))
        blocks = self.rng.choice(np.arange(2, self.max_seq_len - 1), size=block_num, replace=False)
        blocks = torch.sort(torch.tensor(blocks))[0]
        
        # 3) Return
        return {
            "input_ids": inputs,
            "labels": labels,
            "blocks": blocks,
            "sft": False, 
        }