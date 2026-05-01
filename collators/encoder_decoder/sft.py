import torch

class EDSFTCollator:
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
        encoder_inputs = [[self.bos_token_id] + example["input_ids"] for example in batch]
        decoder_inputs = [[self.bos_token_id] + example["output_ids"] for example in batch]
        labels = [example["output_ids"] + [self.eos_token_id] for example in batch]
        encoder_lens = [min(len(seq), self.max_seq_len) for seq in encoder_inputs]

        padded_encoder = [seq[:self.max_seq_len] + [self.pad_token_id] * max(0, self.max_seq_len - len(seq)) for seq in encoder_inputs]
        padded_decoder = [seq[:self.max_seq_len] + [self.pad_token_id] * max(0, self.max_seq_len - len(seq)) for seq in decoder_inputs]
        padded_labels = [seq[:self.max_seq_len] + [self.label_pad_token_id] * max(0, self.max_seq_len - len(seq)) for seq in labels]

        decoder_input_pos = [
            [min(cxt_len + pos, self.max_seq_len - 1) for pos in range(self.max_seq_len)]
            for cxt_len in encoder_lens
        ]

        return {
            "encoder_input_ids": torch.tensor(padded_encoder, dtype=torch.long),
            "decoder_input_ids": torch.tensor(padded_decoder, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "decoder_input_positions": torch.tensor(decoder_input_pos, dtype=torch.long),
            "blocks": torch.tensor(encoder_lens, dtype=torch.long),
            "sft": True,
        }
