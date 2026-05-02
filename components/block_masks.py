import torch
from torch.nn.attention.flex_attention import create_block_mask

# To avoid -inf rows (a token that attends to nothing) the cross mask always
# admits the first token. Only matters for the very first cross-attn block of
# the decoder, which never actually occurs in SFT/Inference (there is always
# context).


def _precompute_block_ids(blocks, seq_len, device):
    blocks = blocks.to(device=device, dtype=torch.int64)
    pos = torch.arange(seq_len, device=device, dtype=torch.int64)
    return torch.bucketize(pos, blocks, right=True)


def causal_all_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def allow_all_mask(b, h, q_idx, kv_idx):
    return torch.ones_like(q_idx, dtype=torch.bool)


def create_pretrain_masks(blocks, seq_len, device):
    block_ids = _precompute_block_ids(blocks, seq_len, device)

    def pt_self_mask(b, h, q, kv):
        return (q >= kv) & (block_ids[q] == block_ids[kv])

    def pt_cross_mask(b, h, q, kv):
        first_token = kv == 0
        return (block_ids[kv] < block_ids[q]) | first_token

    self_mask = create_block_mask(
        pt_self_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )
    cross_mask = create_block_mask(
        pt_cross_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )
    return {"self_mask": self_mask, "cross_mask": cross_mask}


def create_sft_masks(batch_size, blocks, device, enc_len, dec_len):
    sft_block_ids = blocks

    def sft_cross_mask(b, h, q, kv):
        return kv < sft_block_ids[b]

    causal_mask = create_block_mask(
        causal_all_mask, B=batch_size, H=None, Q_LEN=dec_len, KV_LEN=dec_len, device=device,
    )
    cross_mask = create_block_mask(
        sft_cross_mask, B=batch_size, H=None, Q_LEN=dec_len, KV_LEN=enc_len, device=device,
    )
    return {"self_mask": causal_mask, "cross_mask": cross_mask}


def create_inference_masks(device, enc_len, dec_len):
    causal_mask = create_block_mask(
        causal_all_mask, B=None, H=None, Q_LEN=dec_len, KV_LEN=dec_len, device=device,
    )
    allow_mask = create_block_mask(
        allow_all_mask, B=None, H=None, Q_LEN=dec_len, KV_LEN=enc_len, device=device,
    )
    return {"self_mask": causal_mask, "cross_mask": allow_mask}


def create_masks(batch_size, blocks, device, input_ids, encoder_input_ids, decoder_input_ids, sft):
    if sft:
        return create_sft_masks(
            batch_size, blocks, device,
            encoder_input_ids.shape[1], decoder_input_ids.shape[1],
        )
    return create_pretrain_masks(blocks, input_ids.shape[1], device)


# blocks[b] is a per-batch encoder length here (not per-sequence split positions
# as in create_pretrain_masks). The mask only needs to gate out padded encoder
# positions on both encoder-self and decoder-cross paths.
def create_masks_ED(batch_size, blocks, device, encoder_input_ids, decoder_input_ids):
    enc_lens = blocks

    def mask(b, h, q_idx, kv_idx):
        return kv_idx < enc_lens[b]

    enc_mask = create_block_mask(
        mask, B=batch_size, H=None,
        Q_LEN=encoder_input_ids.shape[1], KV_LEN=encoder_input_ids.shape[1], device=device,
    )
    cross_mask = create_block_mask(
        mask, B=batch_size, H=None,
        Q_LEN=decoder_input_ids.shape[1], KV_LEN=encoder_input_ids.shape[1], device=device,
    )
    return {"self_mask": enc_mask}, {"cross_mask": cross_mask}
