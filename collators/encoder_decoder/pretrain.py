import torch
import numpy as np

class EDPretrainCollator:
    def __init__(
        self,
        max_seq_len=2048,
        pad_token_id=3,
        eos_token_id=2,
        bos_token_id=1,
        label_pad_token_id=-100,
        sentinel_start_id=6,
        num_sentinel_tokens=100,
        noise_density=0.15,
        mean_span_length=4.0,
        global_seed=None,
        **kwargs
    ):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.label_pad_token_id = label_pad_token_id
        self.sentinel_start_id = sentinel_start_id
        self.num_sentinel_tokens = num_sentinel_tokens
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.rng = np.random.default_rng(global_seed)

    def _sample_mask_positions(self, input_ids):
        valid_positions = [
            idx
            for idx, token_id in enumerate(input_ids)
            if token_id not in {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        ]
        if not valid_positions:
            return []

        target_mask = max(1, int(round(len(valid_positions) * self.noise_density)))
        target_spans = max(1, int(round(target_mask / self.mean_span_length)))
        num_spans = min(len(valid_positions), self.num_sentinel_tokens, target_spans)

        span_starts = sorted(self.rng.choice(valid_positions, size=num_spans, replace=False).tolist())
        masked = set()

        for span_start in span_starts:
            span_len = max(1, int(self.rng.poisson(self.mean_span_length)))
            for pos in range(span_start, min(span_start + span_len, len(input_ids))):
                if pos not in valid_positions:
                    break
                masked.add(pos)
                if len(masked) >= target_mask:
                    return sorted(masked)

        return sorted(masked) if masked else [valid_positions[0]]

    def _mask_to_spans(self, mask_positions):
        if not mask_positions:
            return []
        spans = []
        span_start = mask_positions[0]
        prev = mask_positions[0]
        for pos in mask_positions[1:]:
            if pos == prev + 1:
                prev = pos
                continue
            spans.append((span_start, prev + 1))
            span_start = pos
            prev = pos
        spans.append((span_start, prev + 1))
        return spans[:self.num_sentinel_tokens]

    def _build_pretrain_example(self, input_ids):
        input_ids = input_ids[:self.max_seq_len]
        spans = self._mask_to_spans(self._sample_mask_positions(input_ids))

        encoder_ids = []
        decoder_labels = []
        idx = 0
        sentinel_offset = 0

        for start, end in spans:
            sentinel_id = self.sentinel_start_id + sentinel_offset
            encoder_ids.extend(input_ids[idx:start])
            encoder_ids.append(sentinel_id)
            decoder_labels.append(sentinel_id)
            decoder_labels.extend(input_ids[start:end])
            idx = end
            sentinel_offset += 1

        encoder_ids.extend(input_ids[idx:])
        decoder_labels.append(self.eos_token_id)

        decoder_input_ids = [self.bos_token_id] + decoder_labels[:-1]
        labels = decoder_labels

        return encoder_ids[:self.max_seq_len], decoder_input_ids[:self.max_seq_len], labels[:self.max_seq_len]

    def __call__(self, batch):
        examples = [self._build_pretrain_example(example["input_ids"]) for example in batch]
        encoder_inputs, decoder_inputs, labels = zip(*examples)
        encoder_lens = [len(seq) for seq in encoder_inputs]

        padded_encoder = [seq + [self.pad_token_id] * (self.max_seq_len - len(seq)) for seq in encoder_inputs]
        padded_decoder = [seq + [self.pad_token_id] * (self.max_seq_len - len(seq)) for seq in decoder_inputs]
        padded_labels = [seq + [self.label_pad_token_id] * (self.max_seq_len - len(seq)) for seq in labels]

        # Standard encoder-decoder uses an independent decoder position stream.
        # Offsetting by encoder length makes long packed examples clip many
        # supervised decoder tokens to the final RoPE position.
        decoder_positions = [list(range(self.max_seq_len)) for _ in encoder_lens]

        return {
            "encoder_input_ids": torch.tensor(padded_encoder, dtype=torch.long),
            "decoder_input_ids": torch.tensor(padded_decoder, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "decoder_input_positions": torch.tensor(decoder_positions, dtype=torch.long),
            "blocks": torch.tensor(encoder_lens, dtype=torch.long), #blocks here is a tensor = B, referring to length of encoder text at each sequence
            "sft": True,
        }
