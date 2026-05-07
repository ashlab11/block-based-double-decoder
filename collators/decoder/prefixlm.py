import numpy as np
import torch


class DecoderPrefixLMCollator:
    """PrefixLM-style collator for the decoder-only model.

    The decoder is trained with standard causal attention everywhere, but loss
    is masked out on the prefix (positions 0..breakpoint-1). This is "true"
    prefixLM minus the bidirectional prefix attention — the bidirectional half
    would require attention-mask plumbing in models/decoder.py that doesn't
    currently exist.

    For the apples-to-apples comparison we want (dd / sed / dec all post-hoc
    fine-tuned on the same data + same per-batch breakpoint), this is the
    architecturally-honest path: the dec model still has its native causal
    constraint, it just *learns* only from suffix-token prediction.

    Input: list of {"input_ids": [t0, t1, ..., t_{seq_len-1}]} (already packed
           via pack_dataset.py; assumes length == max_seq_len).

    Output: {"input_ids": [B, T], "labels": [B, T]}
            labels[i] = input_ids[i+1] (NTP shift), with -100 at:
              - positions before the breakpoint (prefix masking)
              - positions where the next token is BOS/EOS/PAD (existing convention)
    """

    def __init__(
        self,
        max_seq_len=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        label_pad_token_id=-100,
        global_seed=None,
        **kwargs,
    ):
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.rng = np.random.default_rng(global_seed)

    def __call__(self, batch):
        # One breakpoint per batch (matches PrefixLMCollator's behavior so the
        # three architectures see the same prefix/suffix split per batch step).
        # Range [1, max_seq_len-1) so we always have at least one prefix token
        # and at least one suffix token to predict.
        breakpoint = int(self.rng.integers(1, self.max_seq_len - 1))

        inputs = [ex["input_ids"][: self.max_seq_len] for ex in batch]
        # Standard NTP labels: shifted-right input.
        base_labels = [seq[1:] + [self.label_pad_token_id] for seq in inputs]

        labels = []
        for label_seq in base_labels:
            row = [
                self.label_pad_token_id
                if (pos < breakpoint - 1
                    or label in (self.bos_token_id, self.eos_token_id, self.pad_token_id))
                else label
                for pos, label in enumerate(label_seq)
            ]
            labels.append(row)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
