import torch
from torch.nn.attention.flex_attention import create_block_mask

"""
NOTE: To avoid -inf (a token doesn't attend to any other tokens) we always attend to the first token on cross-attn
This only affects the first cross-attn block of the decoder, which never actually occurs in SFT/Inference (since there is always context)
"""

# Global variables
_pretrain_block_ids = None
_sft_block_ids = None

# --- BLOCK MASKS FOR PRETRAINING ----
@torch.compiler.disable()  # keep out of Inductor
def _precompute_block_ids(blocks, seq_len, device):
    #Has corresponding buckets for each index
    blocks = blocks.to(device=device, dtype=torch.int64)
    pos = torch.arange(seq_len, device=device, dtype=torch.int64)
    block_ids = torch.bucketize(pos, blocks, right=True) #block_ids[i] = torch.bucketize(i, blocks)
    return block_ids

def pt_self_mask(b, h, q, kv):
    is_causal = q >= kv
    same_blk  = (_pretrain_block_ids[q] == _pretrain_block_ids[kv])
    return is_causal & same_blk

def pt_cross_mask(b, h, q, kv):
    first_token = kv == 0
    q_block = _pretrain_block_ids[q]
    kv_block = _pretrain_block_ids[kv]
    in_past_block = kv_block < q_block
    return in_past_block | first_token

def sft_cross_mask(b, h, q, kv):
    return kv < _sft_block_ids[b] #Blocks represent the length so [A, B, C, D] would be 4, so <=3 works
    
# Inference masks (no blocks involved)
def causal_all_mask(b, h, q_idx, kv_idx):   # decoder self: full causal
    return kv_idx <= q_idx

def allow_all_mask(b, h, q_idx, kv_idx):    # encoder cross: see everything
    return torch.ones_like(q_idx, dtype=torch.bool)

# --- BLOCK MASKS FOR PRETRAINING -----
@torch.compiler.disable()  # keep the orchestration eager; only the mask kernels compile
def create_pretrain_masks(blocks, seq_len, device):
    block_ids = _precompute_block_ids(blocks, seq_len, device)
    global _pretrain_block_ids
    _pretrain_block_ids = block_ids
    
    self_mask_block = create_block_mask(pt_self_mask,
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True
    )
    cross_mask_block = create_block_mask(pt_cross_mask,
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True
    )
    return {"self_mask": self_mask_block, "cross_mask": cross_mask_block}


# --- BLOCK MASKS FOR SFT -----
@torch.compiler.disable()  # keep out of Inductor
def create_sft_masks(batch_size, blocks, device, enc_len, dec_len):
    causal_mask_block = create_block_mask(causal_all_mask,
        B=batch_size, H=None, Q_LEN=dec_len, KV_LEN=dec_len, device=device, _compile=True
    )
    
    global _sft_block_ids
    _sft_block_ids = blocks
    
    cross_mask_block = create_block_mask(sft_cross_mask,
        B=batch_size, H=None, Q_LEN=dec_len, KV_LEN=enc_len, device=device, _compile=True
    )
    
    return {"self_mask": causal_mask_block, "cross_mask": cross_mask_block}

# --- BLOCK MASKS FOR INFERENCE -----
@torch.compiler.disable()  # keep out of Inductor
def create_inference_masks(device, enc_len, dec_len):
    causal_mask_block = create_block_mask(causal_all_mask,
        B=None, H=None, Q_LEN=dec_len, KV_LEN=dec_len, device=device, _compile=True
    )
    allow_mask_block = create_block_mask(allow_all_mask,
        B=None, H=None, Q_LEN=dec_len, KV_LEN=enc_len, device=device, _compile=True
    )
    return {"self_mask": causal_mask_block, "cross_mask": allow_mask_block}

#---- OVERARCHING MASK CREATOR ----
@torch.compiler.disable()  # keep out of Inductor
def create_masks(batch_size, blocks, device, input_ids, encoder_input_ids, decoder_input_ids, sft):
    if sft:
        return create_sft_masks(batch_size, blocks, device, encoder_input_ids.shape[1], decoder_input_ids.shape[1])
    else:
        return create_pretrain_masks(blocks, input_ids.shape[1], device)
    
    
#---- BLOCKS FOR ED ----
@torch.compiler.disable()
def create_masks_ED(batch_size, blocks, device, encoder_input_ids, decoder_input_ids):
    def mask(b, h, q_idx, kv_idx):    # encoder cross: see everything
        return kv_idx < blocks[b]
    
    enc_mask_block = create_block_mask(mask,
        B=batch_size, H=None, Q_LEN=encoder_input_ids.shape[1], KV_LEN=encoder_input_ids.shape[1], device=device, _compile=True
    )
    cross_mask_block = create_block_mask(mask,
        B=batch_size, H=None, Q_LEN=decoder_input_ids.shape[1], KV_LEN=encoder_input_ids.shape[1], device=device, _compile=True
    )
    #Needs to create two masks, overloading self mask -- the self mask occurs on the ENCODER side
    return {"self_mask": enc_mask_block}, {"cross_mask": cross_mask_block}
