"""
Module for RoPE (Rotary Positional Embeddings) Attention Layer
This module implements a combo-attention and self-attention layer with RoPE.
"""

import torch
import torch.nn as nn
import numpy as np
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.attention.flex_attention import flex_attention as _flex_attention_eager
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# flex_attention's eager fallback is a slow Python reference impl; the docs
# treat compile as the supported path. We compile at import time with
# dynamic=True so one graph covers all (B, H, L) shapes — this is the right
# granularity for compile in this codebase: the kernel is the smallest stable
# unit, while the surrounding model has data-dependent control flow that
# should stay in eager.
flex_attention = torch.compile(_flex_attention_eager, dynamic=True)

class CrossAttention(nn.Module):
    """Standard cross-attention: Q from decoder, K/V from encoder output."""
    def __init__(self, dim, num_heads, seq_len=1024, base_head_dim=0):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        # μP attention scale: √base_head_dim / head_dim. Collapses to standard
        # 1/√head_dim when base_head_dim == head_dim (head_dim held constant
        # across widths) — so existing base-width LR tunings transfer untouched.
        # When head_dim grows past base_head_dim, the extra 1/√head_dim damping
        # is what canonical μP requires for stable softmax across widths.
        # Tensor Programs V (Yang & Hu 2022) §B. Falls back to SDPA's default
        # 1/√head_dim when base_head_dim=0 (μP off, the safe default that also
        # avoids sqrt(0) via short-circuit).
        if base_head_dim:
            self.attn_scale = np.sqrt(base_head_dim) / self.head_dim
        else:
            self.attn_scale = None

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.query_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)
        self.key_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)

    def forward(self, x, encoder_output, block_masks=None, decoder_input_positions=None):
        B, L, D = x.size()
        _, L_enc, _ = encoder_output.size()

        query = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        kv = self.kv_proj(encoder_output)
        key, value = torch.split(kv, [self.dim, self.dim], dim=-1)
        key = key.reshape(B, L_enc, self.num_heads, self.head_dim)
        value = value.reshape(B, L_enc, self.num_heads, self.head_dim)

        query = self.query_rotary(query, input_pos=decoder_input_positions)
        key = self.key_rotary(key)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        cross_mask = None if block_masks is None else block_masks.get('cross_mask')
        if cross_mask is not None:
            output = flex_attention(query, key, value, block_mask=cross_mask, scale=self.attn_scale)
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, scale=self.attn_scale)

        output = output.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(output)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, seq_len = 1024, gating = False, causal=True,
                 base_head_dim=0):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.seq_len = seq_len
        self.causal = causal #Causal only affects when block masks are NOT used
        self.gating = gating
        # See CrossAttention.__init__ for the μP attention scale rationale.
        if base_head_dim:
            self.attn_scale = np.sqrt(base_head_dim) / self.head_dim
        else:
            self.attn_scale = None
        if self.gating:
            self.gater = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid()) 
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)

    def forward(self, x, block_masks=None, input_pos=None, **kwargs):
        B, L, D = x.size()  # [batch_size, seq_length, embed_dim]

        qkv = self.qkv(x) # [B, L, 3 * dim]
        query, key, value = torch.split(qkv, [self.dim, self.dim, self.dim], dim=-1)
        query = query.reshape(B, L, self.num_heads, self.head_dim)
        key = key.reshape(B, L, self.num_heads, self.head_dim)
        value = value.reshape(B, L, self.num_heads, self.head_dim)

        # Apply RoPE (input_pos used by decoder layers to continue from encoder positions)
        query = self.rotary_emb(query, input_pos=input_pos)
        key = self.rotary_emb(key, input_pos=input_pos)
                    
        query = query.transpose(1, 2) # [B, num_heads, L, head_dim]
        key = key.transpose(1, 2) # [B, num_heads, L, head_dim]
        value = value.transpose(1, 2) # [B, num_heads, L, head_dim]
        
        #Dealing with attn mask
        self_mask = None if block_masks is None else block_masks['self_mask']
        
        if self_mask is None:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                is_causal=self.causal, scale=self.attn_scale)
        else:
           output = flex_attention(query, key, value, block_mask=self_mask, scale=self.attn_scale)
        
        if self.gating:
            gating_modulator = self.gater(x).reshape(B, self.num_heads, L, self.head_dim)
            output = output * gating_modulator

        output = output.transpose(1, 2).reshape(B, L, D)  # [B, L, embed_dim]
        output = self.out_proj(output)
        return output
        
class ComboAttention(nn.Module):
    def __init__(self, dim, num_heads, seq_len = 1024, shared = True, logit_biases = False,
                 base_head_dim=0):
        super(ComboAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.seq_len = seq_len
        self.shared = shared
        self.logit_biases = logit_biases
        # See CrossAttention.__init__ for the μP attention scale rationale.
        if base_head_dim:
            self.attn_scale = np.sqrt(base_head_dim) / self.head_dim
        else:
            self.attn_scale = None
        
        if shared:
            self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        else: #Not shared, so we use an additional 2d^2 parameters 
            self.enc_kv_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.dec_kv_proj = nn.Linear(dim, 2 * dim, bias=False)
            
        if logit_biases:
            self.logit_bias_proj = nn.Parameter(torch.zeros(num_heads, 2)) # [:, 0] = self, [:, 1] = cross
            self.mix_temp = nn.Parameter(torch.ones(num_heads))
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.query_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)
        self.key_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)
        
    def forward(self, x, encoder_inputs, block_masks=None, decoder_input_positions=None):
        B, L, D = x.size()
        _, L_enc, _ = encoder_inputs.size()
                
        #--- DECODER ---
        query = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim) # Query shared over encoder/decoder regardless
        
        dec_kv = self.kv_proj(x) if self.shared else self.dec_kv_proj(x)
        dec_key, dec_value = torch.split(dec_kv, [self.dim, self.dim], dim=-1)
        dec_key = dec_key.reshape(B, L, self.num_heads, self.head_dim)
        dec_value = dec_value.reshape(B, L, self.num_heads, self.head_dim)
        
        query = self.query_rotary(query, input_pos=decoder_input_positions)
        dec_key = self.key_rotary(dec_key, input_pos=decoder_input_positions)
        
        query = query.transpose(1, 2)
        dec_key = dec_key.transpose(1, 2)
        dec_value = dec_value.transpose(1, 2)
        
        #--- ENCODER (no need for query) ---
        enc_kv = self.kv_proj(encoder_inputs) if self.shared else self.enc_kv_proj(encoder_inputs)
        enc_key, enc_value = torch.split(enc_kv, [self.dim, self.dim], dim=-1)
        enc_key = enc_key.reshape(B, L_enc, self.num_heads, self.head_dim)
        enc_value = enc_value.reshape(B, L_enc, self.num_heads, self.head_dim)
        
        enc_key = self.key_rotary(enc_key)
        
        enc_key = enc_key.transpose(1, 2)
        enc_value = enc_value.transpose(1, 2)
        
        # --- ATTENTION ---
        use_flash = block_masks is None and HAS_FLASH_ATTN  #Only happens if we're in inference mode AND on a gpu that can handle flash attn
        if use_flash:
            #Flash attention expects [B, L, N_h, D_h]
            query, enc_key, dec_key, enc_value, dec_value = query.transpose(1, 2), enc_key.transpose(1, 2), dec_key.transpose(1, 2), enc_value.transpose(1, 2), dec_value.transpose(1, 2)

            dec_output, dec_lse, _ = flash_attn_func(query, dec_key, dec_value, causal=True, return_attn_probs=True, softmax_scale=self.attn_scale)
            enc_output, enc_lse, _ = flash_attn_func(query, enc_key, enc_value, causal=False, return_attn_probs=True, softmax_scale=self.attn_scale)

        elif block_masks is None:
            # SDPA fallback: no flash-attn, no block masks (inference mode).
            # We need LSE for the sigmoid gate, so compute attention manually.
            scale = self.attn_scale or 1.0 / (self.head_dim ** 0.5)

            # Decoder self-attention (causal)
            dec_scores = torch.matmul(query, dec_key.transpose(-2, -1)) * scale
            causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            dec_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            dec_lse = torch.logsumexp(dec_scores, dim=-1)  # [B, H, L]
            dec_attn = torch.softmax(dec_scores, dim=-1)
            dec_output = torch.matmul(dec_attn, dec_value)

            # Encoder cross-attention (full, non-causal)
            enc_scores = torch.matmul(query, enc_key.transpose(-2, -1)) * scale
            enc_lse = torch.logsumexp(enc_scores, dim=-1)  # [B, H, L]
            enc_attn = torch.softmax(enc_scores, dim=-1)
            enc_output = torch.matmul(enc_attn, enc_value)
        else:
            self_mask = block_masks.get('self_mask')
            cross_mask = block_masks.get('cross_mask')
            
            assert self_mask is not None and cross_mask is not None, "Self and cross masks must both be not None"

            dec_output, dec_lse = flex_attention(query, dec_key, dec_value, block_mask=self_mask, return_lse=True, scale=self.attn_scale)
            enc_output, enc_lse = flex_attention(query, enc_key, enc_value, block_mask=cross_mask, return_lse=True, scale=self.attn_scale)
            
        if self.logit_biases:
            #Temperatures and logit biases help to self-stabilize during training -- they will hopefully be close to 0/1 respectively by the end
            dec_lse = (dec_lse + self.logit_bias_proj[:, 0].reshape(1, self.num_heads, 1)) * self.mix_temp.reshape(1, self.num_heads, 1)
            enc_lse = (enc_lse + self.logit_bias_proj[:, 1].reshape(1, self.num_heads, 1)) * self.mix_temp.reshape(1, self.num_heads, 1)

        dec_w = torch.sigmoid(dec_lse - enc_lse).unsqueeze(-1) # Want this to be [B, L, N_h, 1]
        dec_w = dec_w.transpose(1, 2) if use_flash else dec_w # Flash uses weird dimensions
        output = dec_w * dec_output + (1 - dec_w) * enc_output
        
        output = output.reshape(B, L, D) if use_flash else output.transpose(1, 2).reshape(B, L, D)
        output = self.out_proj(output)
        return output
        