import random
import torch

class DecoderSFTCollator:
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
        max_seq_len = 2048,
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
        
        seqs = []
        ctx_lens = []
        
        for example in batch:
            #Tokenizer doesn't auto put <bos> into conversation, we need to do so ourselves
            #NOTE: The response BOS token MUST have decoder_input_pos of 0 (this fits how it works in pretraining
            # Think I've solved this -- dec self DOESN'T look at <bos>, but it DOES have a sink token
            # Then, during SFT, it doesn't need to look at a <bos> as long as it has a sink token?
            seqs.append([self.bos_token_id] + example["input_ids"] + example['output_ids'] + [self.eos_token_id])
            ctx_lens.append(len(example["input_ids"]) + 1) # +1 for the bos token
            
        return self._pad_sequences(seqs, ctx_lens)

    def _pad_sequences(self, seqs, ctx_lens):
        """
        Pads input and label sequences and returns:
          - padded_inputs
          - padded_labels
        """
        # 1) Pad inputs
        padded_seqs = [
            seq + [self.pad_token_id] * (self.max_seq_len - len(seq))
            for seq in seqs
        ]
        
        # 2) Creating decoder inputs + labels
        #CONTEXT: [BOS, A, B, C, ASSISTANT, D, E, F, END]
        #LABELS: [A, B, C, ASSISTANT, D, E, F, END, <PAD>]
        #ctx_len = 4        
        base_labels = [padded_seq[1:] + [self.label_pad_token_id] for padded_seq in padded_seqs]
        labels = [[
            label if (pos >= ctx_lens[i] and label != self.pad_token_id) else self.label_pad_token_id
            for pos, label in enumerate(label_seq)
        ] for i, label_seq in enumerate(base_labels)]
        
        
        return {
            "input_ids":         torch.tensor(padded_seqs, dtype=torch.long),
            "labels":            torch.tensor(labels, dtype=torch.long)
        }
