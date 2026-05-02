"""
RoPE (Rotary Positional Embeddings) attention layers.

Uses torch.nn.functional.scaled_dot_product_attention with dense bool
attn_mask tensors throughout — no flex_attention, no torch.compile HOP
machinery. ComboAttention computes attention manually because it needs
the per-row logsumexp of attention scores for the sigmoid gate (SDPA's
public API doesn't expose LSE); compile fuses this into roughly the same
kernel SDPA's math backend produces.
"""

import torch
import torch.nn as nn
import numpy as np
from torchtune.modules import RotaryPositionalEmbeddings


import torch._inductor.config as _ic
_ic.pattern_matcher = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def _to_attn_mask(mask):
    """Reshape a bool mask to something SDPA's attn_mask= will broadcast.
    [L_q, L_kv]   -> [1, 1, L_q, L_kv]
    [B, L_q, L_kv] -> [B, 1, L_q, L_kv]
    """
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    return mask


class CrossAttention(nn.Module):
    """Standard cross-attention: Q from decoder, K/V from encoder output."""
    def __init__(self, dim, num_heads, seq_len=1024, base_head_dim=0):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
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
        attn_mask = _to_attn_mask(cross_mask)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.attn_scale,
        )

        output = output.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(output)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, seq_len=1024, gating=False, causal=True,
                 base_head_dim=0):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.seq_len = seq_len
        self.causal = causal  # only used when no mask is supplied
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
        B, L, D = x.size()

        qkv = self.qkv(x)
        query, key, value = torch.split(qkv, [self.dim, self.dim, self.dim], dim=-1)
        query = query.reshape(B, L, self.num_heads, self.head_dim)
        key = key.reshape(B, L, self.num_heads, self.head_dim)
        value = value.reshape(B, L, self.num_heads, self.head_dim)

        query = self.rotary_emb(query, input_pos=input_pos)
        key = self.rotary_emb(key, input_pos=input_pos)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        self_mask = None if block_masks is None else block_masks.get('self_mask')
        attn_mask = _to_attn_mask(self_mask)
        # SDPA disallows passing both attn_mask and is_causal=True.
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None and self.causal),
            scale=self.attn_scale,
        )

        if self.gating:
            gating_modulator = self.gater(x).reshape(B, self.num_heads, L, self.head_dim)
            output = output * gating_modulator

        output = output.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(output)


class ComboAttention(nn.Module):
    def __init__(self, dim, num_heads, seq_len=1024, shared=True, logit_biases=False,
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
        else:
            self.enc_kv_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.dec_kv_proj = nn.Linear(dim, 2 * dim, bias=False)

        if logit_biases:
            self.logit_bias_proj = nn.Parameter(torch.zeros(num_heads, 2))  # [:, 0] = self, [:, 1] = cross
            self.mix_temp = nn.Parameter(torch.ones(num_heads))

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.query_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)
        self.key_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)

    def _attn_with_lse(self, query, key, value, mask, default_causal):
        """Manual matmul → masked softmax → @V. Returns (output, lse).
        mask: bool [B, H, L_q, L_kv] (already broadcastable) or None.
        default_causal: if mask is None, apply a triangular causal mask.
        """
        scale = self.attn_scale if self.attn_scale is not None else 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B, H, L_q, L_kv]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        elif default_causal:
            L = scores.size(-2)
            causal = torch.triu(
                torch.ones(L, L, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        lse = torch.logsumexp(scores, dim=-1)  # [B, H, L_q]
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, value)  # [B, H, L_q, D_h]
        return out, lse

    def forward(self, x, encoder_inputs, block_masks=None, decoder_input_positions=None):
        B, L, D = x.size()
        _, L_enc, _ = encoder_inputs.size()

        # --- DECODER ---
        query = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)

        dec_kv = self.kv_proj(x) if self.shared else self.dec_kv_proj(x)
        dec_key, dec_value = torch.split(dec_kv, [self.dim, self.dim], dim=-1)
        dec_key = dec_key.reshape(B, L, self.num_heads, self.head_dim)
        dec_value = dec_value.reshape(B, L, self.num_heads, self.head_dim)

        query = self.query_rotary(query, input_pos=decoder_input_positions)
        dec_key = self.key_rotary(dec_key, input_pos=decoder_input_positions)

        query = query.transpose(1, 2)
        dec_key = dec_key.transpose(1, 2)
        dec_value = dec_value.transpose(1, 2)

        # --- ENCODER (no Q) ---
        enc_kv = self.kv_proj(encoder_inputs) if self.shared else self.enc_kv_proj(encoder_inputs)
        enc_key, enc_value = torch.split(enc_kv, [self.dim, self.dim], dim=-1)
        enc_key = enc_key.reshape(B, L_enc, self.num_heads, self.head_dim)
        enc_value = enc_value.reshape(B, L_enc, self.num_heads, self.head_dim)

        enc_key = self.key_rotary(enc_key)

        enc_key = enc_key.transpose(1, 2)
        enc_value = enc_value.transpose(1, 2)

        # --- ATTENTION ---
        # Inference fast path: no masks + flash_attn available. flash_attn
        # returns LSE natively, so no manual matmul is needed.
        use_flash = block_masks is None and HAS_FLASH_ATTN
        if use_flash:
            # Flash attention expects [B, L, N_h, D_h]
            query_f = query.transpose(1, 2)
            dec_key_f = dec_key.transpose(1, 2)
            dec_value_f = dec_value.transpose(1, 2)
            enc_key_f = enc_key.transpose(1, 2)
            enc_value_f = enc_value.transpose(1, 2)

            dec_output, dec_lse, _ = flash_attn_func(
                query_f, dec_key_f, dec_value_f,
                causal=True, return_attn_probs=True, softmax_scale=self.attn_scale,
            )
            enc_output, enc_lse, _ = flash_attn_func(
                query_f, enc_key_f, enc_value_f,
                causal=False, return_attn_probs=True, softmax_scale=self.attn_scale,
            )
        else:
            self_mask = None if block_masks is None else block_masks.get('self_mask')
            cross_mask = None if block_masks is None else block_masks.get('cross_mask')
            dec_output, dec_lse = self._attn_with_lse(
                query, dec_key, dec_value,
                _to_attn_mask(self_mask),
                default_causal=True,
            )
            enc_output, enc_lse = self._attn_with_lse(
                query, enc_key, enc_value,
                _to_attn_mask(cross_mask),
                default_causal=False,
            )

        if self.logit_biases:
            # Temperatures and logit biases help self-stabilize during training —
            # they will hopefully be close to 0/1 respectively by the end.
            dec_lse = (dec_lse + self.logit_bias_proj[:, 0].reshape(1, self.num_heads, 1)) * self.mix_temp.reshape(1, self.num_heads, 1)
            enc_lse = (enc_lse + self.logit_bias_proj[:, 1].reshape(1, self.num_heads, 1)) * self.mix_temp.reshape(1, self.num_heads, 1)

        dec_w = torch.sigmoid(dec_lse - enc_lse).unsqueeze(-1)  # [B, H, L, 1]
        dec_w = dec_w.transpose(1, 2) if use_flash else dec_w  # flash uses [B, L, H, ·]
        output = dec_w * dec_output + (1 - dec_w) * enc_output

        output = output.reshape(B, L, D) if use_flash else output.transpose(1, 2).reshape(B, L, D)
        output = self.out_proj(output)
        return output
