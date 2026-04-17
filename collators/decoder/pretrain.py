import random
import torch
import numpy as np

class DecoderPretrainCollator:
    """
    Data Collator for Decoder
    """
    def __init__(
        self,
        eos_token_id=2,
        bos_token_id=1,
        label_pad_token_id=-100,
        **kwargs
    ):
        self.bos_token_id = bos_token_id
        self.label_pad_token_id = label_pad_token_id
        self.eos_token_id = eos_token_id
        
    def __call__(self, batch):
        """
        batch: List[Dict[str, List[int]]]  each dict must have an "input_ids" key.
        Assumes that input_ids are already packed.
        Returns a dict with:
          - "input_ids": padded input sequences
          - "labels":   padded label sequences
        """
        
        # 1) Get inputs and labels
        inputs = [example["input_ids"] for example in batch]
        labels = [example["input_ids"][1:] + [self.label_pad_token_id] for example in batch]
        #Mask out the label corresponding to one after eos
        labels = [
            [
                (label if input_id not in [self.bos_token_id, self.eos_token_id] else self.label_pad_token_id)
                for input_id, label in zip(input_seq, label_seq)
            ]
            for input_seq, label_seq in zip(inputs, labels)
        ]
        
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        

        # 5) Return
        return {
            "input_ids": inputs,
            "labels": labels
        }