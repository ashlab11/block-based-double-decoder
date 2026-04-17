import random
import torch
import numpy as np

class BasicSFTCollator:
    """
    Data collator for conversations.
    Takes in a batch of conversations and returns a batch of padded input and label sequences.
    """
    def __init__(
        self,
        label_pad_token_id=-100,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        max_seq_len=2048,
        **kwargs
    ):
        self.eos_token_id = eos_token_id
        self.label_pad_token_id = label_pad_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.max_seq_len = max_seq_len
        
    def __call__(self, batch):
        """
        batch: List[Dict[str, List[int]]]  each dict must have an "input_ids" and "output_ids" key.
        Returns a dict with:
          - "input_ids": padded input sequences
          - "labels":   padded label sequences
        """
        
        contexts = []
        responses = []
        
        for example in batch:
            #Tokenizer doesn't auto put <bos> into conversation, we need to do so ourselves
            #NOTE: The response BOS token MUST have decoder_input_pos of 0 (this fits how it works in pretraining
            # Think I've solved this -- dec self DOESN'T look at <bos>, but it DOES have a sink token
            # Then, during SFT, it doesn't need to look at a <bos> as long as it has a sink token?
            contexts.append([self.bos_token_id] + example["input_ids"])
            responses.append(example["output_ids"] + [self.eos_token_id])
            

        return self._pad_sequences(contexts, responses)

    def _pad_sequences(self, contexts, responses):
        """
        Pads input and label sequences and returns:
          - padded_inputs
          - padded_labels
          - decoder_input_positions
          - blocks
        """
        # 1) Pad inputs
        context_lens = [len(seq) for seq in contexts] # Important for decoder input pos 
                
        padded_contexts = [
            seq + [self.pad_token_id] * (self.max_seq_len - len(seq))
            for seq in contexts
        ]
        
        padded_responses = [
            seq + [self.pad_token_id] * (self.max_seq_len - len(seq))
            for seq in responses
        ]
        
        # 3) Create decoder input positions
        decoder_input_pos = [[
            cxt_len + j for j in range(self.max_seq_len)
        ] for cxt_len in context_lens]
        
        #With this the dec pos will go > seq len, but we don't want this
        #Any individual decoder sentence will be <= seq len, it's because of padding
        decoder_input_pos = [[
            pos if pos < self.max_seq_len else self.max_seq_len - 1
            for pos in pos_seq
        ] for pos_seq in decoder_input_pos]
        
        
        # 3) Creating decoder inputs + labels
        base_labels = [padded_response[1:] + [self.label_pad_token_id] for padded_response in padded_responses]
        labels = [
            [
                label if label != self.pad_token_id else self.label_pad_token_id for label in label_seq
            ] for label_seq in base_labels
        ]
        
        
        return {
            "encoder_input_ids":         torch.tensor(padded_contexts, dtype=torch.long),
            "decoder_input_ids":         torch.tensor(padded_responses, dtype=torch.long),
            "labels":                    torch.tensor(labels, dtype=torch.long),
            "decoder_input_positions":         torch.tensor(decoder_input_pos, dtype=torch.long),
            "blocks":                    torch.tensor(context_lens, dtype=torch.long),
            "sft":                     True
        }
