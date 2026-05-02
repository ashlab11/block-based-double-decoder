import torch
from torch.nn.attention.flex_attention import create_block_mask

"""
NOTE: To avoid -inf (a token doesn't attend to any other tokens) we always attend to the first token on cross-attn
This only affects the first cross-attn block of the decoder, which never actually occurs in SFT/Inference (since there is always context)

DESIGN NOTE: The mask functions used to read module-level globals
(_pretrain_block_ids, _sft_block_ids). That was racy: every call to
create_pretrain_masks reassigned the global, but FlexAttention's compiled
mask kernels capture the global by name and re-read it at kernel-eval time.
Across many varying-seq_len calls (e.g. lambada / copy_retrieval evals),
dynamo's id-based guards plus single-slot caching let the kernel read a
stale tensor of the wrong length → out-of-bounds load → CUDA IMA.

The fix: make block_ids a *local closure variable* captured at mask-creation
time. Each BlockMask owns its own block_ids — no shared mutable state.
"""

@torch.compiler.disable()  # keep out of Inductor
def _precompute_block_ids(blocks, seq_len, device):
    blocks = blocks.to(device=device, dtype=torch.int64)
    pos = torch.arange(seq_len, device=device, dtype=torch.int64)
    block_ids = torch.bucketize(pos, blocks, right=True)
    return block_ids


# Inference masks (no blocks involved — fully stateless, safe at module level)
def causal_all_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx

def allow_all_mask(b, h, q_idx, kv_idx):
    return torch.ones_like(q_idx, dtype=torch.bool)


# --- BLOCK MASKS FOR PRETRAINING -----
@torch.compiler.disable()
def create_pretrain_masks(blocks, seq_len, device):
    block_ids = _precompute_block_ids(blocks, seq_len, device)

    def pt_self_mask(b, h, q, kv):
        is_causal = q >= kv
        same_blk = block_ids[q] == block_ids[kv]
        return is_causal & same_blk

    def pt_cross_mask(b, h, q, kv):
        first_token = kv == 0
        q_block = block_ids[q]
        kv_block = block_ids[kv]
        in_past_block = kv_block < q_block
        return in_past_block | first_token

    # NOTE: _compile=True is deprecated upstream and breaks torch.compile of the
    # outer model: it internally calls torch.compile(create_block_mask), which
    # builds BlockMask tensors under fake-tensor mode. Those fake/lazy fields
    # leak into the parent compile graph and crash inductor's get_attr→.item()
    # constant-folding path with "tensor has a non-zero number of elements but
    # its data is not allocated yet". Eager construction is fast enough here
    # (each unique (blocks, seq_len) is built once and cached via
    # parallel_scaling._cached_create_masks).
    self_mask_block = create_block_mask(pt_self_mask,
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )
    cross_mask_block = create_block_mask(pt_cross_mask,
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )
    return {"self_mask": self_mask_block, "cross_mask": cross_mask_block}


# --- BLOCK MASKS FOR SFT -----
@torch.compiler.disable()
def create_sft_masks(batch_size, blocks, device, enc_len, dec_len):
    sft_block_ids = blocks  # captured by closure below

    def sft_cross_mask(b, h, q, kv):
        return kv < sft_block_ids[b]

    causal_mask_block = create_block_mask(causal_all_mask,
        B=batch_size, H=None, Q_LEN=dec_len, KV_LEN=dec_len, device=device,
    )
    cross_mask_block = create_block_mask(sft_cross_mask,
        B=batch_size, H=None, Q_LEN=dec_len, KV_LEN=enc_len, device=device,
    )
    return {"self_mask": causal_mask_block, "cross_mask": cross_mask_block}


# --- BLOCK MASKS FOR INFERENCE -----
@torch.compiler.disable()
def create_inference_masks(device, enc_len, dec_len):
    causal_mask_block = create_block_mask(causal_all_mask,
        B=None, H=None, Q_LEN=dec_len, KV_LEN=dec_len, device=device,
    )
    allow_mask_block = create_block_mask(allow_all_mask,
        B=None, H=None, Q_LEN=dec_len, KV_LEN=enc_len, device=device,
    )
    return {"self_mask": causal_mask_block, "cross_mask": allow_mask_block}


#---- OVERARCHING MASK CREATOR ----
@torch.compiler.disable()
def create_masks(batch_size, blocks, device, input_ids, encoder_input_ids, decoder_input_ids, sft):
    if sft:
        return create_sft_masks(batch_size, blocks, device, encoder_input_ids.shape[1], decoder_input_ids.shape[1])
    else:
        return create_pretrain_masks(blocks, input_ids.shape[1], device)


#---- BLOCKS FOR ED (T5-style) ----
# blocks[b] is a per-batch encoder length, NOT the per-sequence split positions
# used by create_pretrain_masks. The mask only needs to gate out padded encoder
# positions on both encoder-self and decoder-cross paths.
@torch.compiler.disable()
def create_masks_ED(batch_size, blocks, device, encoder_input_ids, decoder_input_ids):
    enc_lens = blocks  # captured by closure below

    def mask(b, h, q_idx, kv_idx):
        return kv_idx < enc_lens[b]

    enc_mask_block = create_block_mask(mask,
        B=batch_size, H=None, Q_LEN=encoder_input_ids.shape[1], KV_LEN=encoder_input_ids.shape[1], device=device,
    )
    cross_mask_block = create_block_mask(mask,
        B=batch_size, H=None, Q_LEN=decoder_input_ids.shape[1], KV_LEN=encoder_input_ids.shape[1], device=device,
    )
    # Returns a tuple of two single-key dicts to match StandardEncDec.forward's
    # `causal_mask, cross_mask = create_masks_ED(...)` unpacking.
    return {"self_mask": enc_mask_block}, {"cross_mask": cross_mask_block}
