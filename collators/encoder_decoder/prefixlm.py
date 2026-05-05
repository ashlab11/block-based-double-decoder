import torch
import numpy as np

class PrefixLMCollator:
    def __init__(self, 
        max_seq_len=2048,
        pad_token_id=3,
        eos_token_id=2,
        bos_token_id=1,
        label_pad_token_id=-100,
        global_seed=None,
        **kwargs
    ):
        self.rng = np.random.default_rng(global_seed)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.label_pad_token_id = label_pad_token_id
        self.max_seq_len = max_seq_len

    def _build_pretrain_example(self, input_ids, breakpoint):
        input_ids = input_ids[:self.max_seq_len] #Shouldn't be necessary since packed, but just in case
        encoder_ids = input_ids[:breakpoint]
        labels = input_ids[breakpoint:] + [self.eos_token_id]
        decoder_ids = [self.bos_token_id] + labels[:-1]
        return encoder_ids, decoder_ids, labels
    
    def __call__(self, batch):
        """
        Returns encoder ids, decoder ids, and blocks. Datasets are packed, so blocks are unnecessary
        (decoder and DD do the same)
        """
        breakpoint = self.rng.integers(1, self.max_seq_len - 1)
        examples = [self._build_pretrain_example(example["input_ids"], breakpoint) for example in batch]
        encoder_seqs, decoder_seqs, labels = zip(*examples)
        encoder_lens = [len(seq) for seq in encoder_seqs]
        decoder_positions = [
            [min(enc_len + pos, self.max_seq_len - 1) for pos in range(self.max_seq_len)]
            for enc_len in encoder_lens
        ]

        encoder_seqs = [seq + [self.pad_token_id] * (self.max_seq_len - len(seq)) for seq in encoder_seqs]
        decoder_seqs = [seq + [self.pad_token_id] * (self.max_seq_len - len(seq)) for seq in decoder_seqs]
        labels = [seq + [self.label_pad_token_id] * (self.max_seq_len - len(seq)) for seq in labels]
        
        return {
            "encoder_input_ids": torch.tensor(encoder_seqs, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_seqs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "decoder_input_positions": torch.tensor(decoder_positions, dtype=torch.long),
            "blocks": torch.tensor(encoder_lens, dtype=torch.long), #blocks here is a tensor = B, referring to length of encoder text at each sequence
            "sft": True,
        }
        
        
            