"""Dense boolean attention masks for SDPA.

Each builder returns a dict of bool tensors where True = "attend to this
position". Shapes are chosen so they broadcast cleanly to the [B, H, L_q, L_kv]
shape SDPA's `attn_mask` expects:

  - [L_q, L_kv]              shared across batch and heads
  - [B, L_q, L_kv]           per-batch, shared across heads (unsqueeze H in caller)

Cross-attention adds a sink at kv=0 to avoid all-False rows (which would
produce -inf softmax). This only matters for the first cross-attn block of
the decoder, which never actually occurs in SFT/Inference (there is always
context).
"""

import torch


def _block_ids(blocks, seq_len, device):
    blocks = blocks.to(device=device, dtype=torch.int64)
    pos = torch.arange(seq_len, device=device, dtype=torch.int64)
    return torch.bucketize(pos, blocks, right=True)


def create_pretrain_masks(blocks, seq_len, device):
    """DD pretraining: per-token block boundaries shared across the batch.
    self_mask: causal AND same-block.
    cross_mask: kv in a strictly earlier block (with kv=0 sink).
    Both [L, L]."""
    block_ids = _block_ids(blocks, seq_len, device)
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    causal = q_idx >= kv_idx
    same_blk = block_ids.unsqueeze(1) == block_ids.unsqueeze(0)
    self_mask = causal & same_blk

    in_past_block = block_ids.unsqueeze(0) < block_ids.unsqueeze(1)
    first_token = (kv_idx == 0).expand(seq_len, seq_len)
    cross_mask = in_past_block | first_token

    return {"self_mask": self_mask, "cross_mask": cross_mask}


def create_sft_masks(batch_size, blocks, device, enc_len, dec_len):
    """DD SFT: causal self-attention + per-batch context-length cross gate.
    self_mask: [L_dec, L_dec] causal.
    cross_mask: [B, L_dec, L_enc]."""
    blocks = blocks.to(device=device, dtype=torch.int64)
    q_dec = torch.arange(dec_len, device=device).unsqueeze(1)
    kv_dec = torch.arange(dec_len, device=device).unsqueeze(0)
    self_mask = kv_dec <= q_dec

    kv_enc = torch.arange(enc_len, device=device).unsqueeze(0)
    pad_per_batch = kv_enc < blocks.unsqueeze(1)
    cross_mask = pad_per_batch.unsqueeze(1).expand(batch_size, dec_len, enc_len)

    return {"self_mask": self_mask, "cross_mask": cross_mask}


def create_inference_masks(device, enc_len, dec_len):
    """Inference-time: causal self, full cross. Both [L_q, L_kv]."""
    q_idx = torch.arange(dec_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(dec_len, device=device).unsqueeze(0)
    self_mask = kv_idx <= q_idx
    cross_mask = torch.ones(dec_len, enc_len, device=device, dtype=torch.bool)
    return {"self_mask": self_mask, "cross_mask": cross_mask}


def create_masks(batch_size, blocks, device, input_ids, encoder_input_ids, decoder_input_ids, sft):
    if sft:
        return create_sft_masks(
            batch_size, blocks, device,
            encoder_input_ids.shape[1], decoder_input_ids.shape[1],
        )
    return create_pretrain_masks(blocks, input_ids.shape[1], device)


def create_masks_ED(batch_size, blocks, device, encoder_input_ids, decoder_input_ids):
    """SED (T5-style): blocks[b] is a per-batch encoder length. Both encoder
    self-attention and decoder cross-attention just gate out padded encoder
    positions. Returns ({'self_mask'}, {'cross_mask'}) to match SED.forward's
    unpacking — the encoder dict only carries 'self_mask', the decoder dict
    only 'cross_mask'."""
    enc_lens = blocks.to(device=device, dtype=torch.int64)
    enc_len = encoder_input_ids.shape[1]
    dec_len = decoder_input_ids.shape[1]
    kv_idx = torch.arange(enc_len, device=device).unsqueeze(0)
    pad_mask = kv_idx < enc_lens.unsqueeze(1)  # [B, L_enc]

    enc_self = pad_mask.unsqueeze(1).expand(batch_size, enc_len, enc_len)
    cross = pad_mask.unsqueeze(1).expand(batch_size, dec_len, enc_len)

    return {"self_mask": enc_self}, {"cross_mask": cross}
