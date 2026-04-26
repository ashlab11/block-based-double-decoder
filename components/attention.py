"""
Module for RoPE (Rotary Positional Embeddings) Attention Layer
This module implements a combo-attention and self-attention layer with RoPE.
"""

import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.attention.flex_attention import flex_attention, AuxRequest
from flash_attn import flash_attn_func

class BaseAttention(nn.Module):
    def __init__(self, dim, num_heads, seq_len = 1024):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.rotary_emb = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=seq_len)
        
        #QKV differ between methods, out proj doesn't
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def attend(self, q, k, v, causal = False, block_mask = None, return_lse=False):
        """Basic attention function, saves substantial space in later functions"""
        if block_mask is None:
            assert not return_lse, "Can't return LSE in base attn"
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                is_causal=causal)
        else:
            aux = AuxRequest(lse=True) if return_lse else None
            return flex_attention(q, k, v, block_mask = block_mask, return_aux=aux)
    def reshape_to_heads(self, *xs):
        return (x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim) for x in xs)
    def reshape_to_attend(self, *xs):
        return (x.transpose(1, 2) for x in xs)  # [B, num_heads, L, head_dim]
class SelfAttention(BaseAttention):
    def __init__(self, dim, num_heads, seq_len = 1024):
        super().__init__(dim, num_heads, seq_len)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x, block_masks=None, **kwargs):
        B, L, D = x.size()  # [batch_size, seq_length, embed_dim]
        
        qkv = self.qkv(x) # [B, L, 3 * dim]
        query, key, value = torch.split(qkv, [self.dim, self.dim, self.dim], dim=-1)
        query, key, value = self.reshape_to_heads(query, key, value)
        
        # Apply RoPE, transposing comes after
        query = self.rotary_emb(query)
        key = self.rotary_emb(key)
        query, key, value = self.reshape_to_attend(query, key, value)
        
        #Dealing with attn mask
        self_mask = None if block_masks is None else block_masks['self_mask']
        output = self.attend(query, key, value, causal=True, block_mask=self_mask) #Causality ignored if mask exists

        output = output.transpose(1, 2).reshape(B, L, D)  # [B, L, embed_dim]
        output = self.out_proj(output)
        return output
        
class CrossAttention(BaseAttention):
    def __init__(self, dim, num_heads, seq_len = 1024):
        super().__init__(dim, num_heads, seq_len)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x, encoder_inputs, **kwargs):
        B, L, D = x.size()  # [batch_size, seq_length, embed_dim]
        
        query = self.q_proj(x)
        kv = self.kv_proj(encoder_inputs)
        key, value = torch.split(kv, [self.dim, self.dim], dim=-1)
        query, key, value = self.reshape_to_heads(query, key, value)
        
        # Apply RoPE, transposing comes after
        query = self.rotary_emb(query)
        key = self.rotary_emb(key)
                    
        query, key, value = self.reshape_to_attend(query, key, value)
        
        #Always full
        output = self.attend(query, key, value, causal=False)

        output = output.transpose(1, 2).reshape(B, L, D)  # [B, L, embed_dim]
        output = self.out_proj(output)
        return output
class ComboAttention(BaseAttention):
    def __init__(self, dim, num_heads, seq_len = 1024, shared = True, logit_biases = False):
        super(ComboAttention, self).__init__(dim, num_heads, seq_len)
        self.seq_len = seq_len
        self.shared = shared
        self.logit_biases = logit_biases
        
        if shared:
            self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        else: #Not shared, so we use an additional 2d^2 parameters 
            self.enc_kv_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.dec_kv_proj = nn.Linear(dim, 2 * dim, bias=False)
            
        if logit_biases:
            self.logit_bias_proj = nn.Parameter(torch.zeros(num_heads, 2)) # [:, 0] = self, [:, 1] = cross
            self.mix_temp = nn.Parameter(torch.ones(num_heads))
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x, encoder_inputs, block_masks=None, decoder_input_positions=None):
        B, L, D = x.size()
                
        #--- Projections ---
        query = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim) # Query shared over encoder/decoder regardless
        
        dec_kv = self.kv_proj(x) if self.shared else self.dec_kv_proj(x)
        dec_key, dec_value = torch.split(dec_kv, [self.dim, self.dim], dim=-1)
        enc_kv = self.kv_proj(encoder_inputs) if self.shared else self.enc_kv_proj(encoder_inputs)
        enc_key, enc_value = torch.split(enc_kv, [self.dim, self.dim], dim=-1)
        
        #Reshaping and rotary embeddings
        dec_key, dec_value, enc_key, enc_value = self.reshape_to_heads(dec_key, dec_value, enc_key, enc_value)
        
        query = self.rotary_emb(query, input_pos=decoder_input_positions)
        dec_key = self.rotary_emb(dec_key, input_pos=decoder_input_positions)
        enc_key = self.rotary_emb(enc_key)
        
        query, dec_key, dec_value, enc_key, enc_value = self.reshape_to_attend(query, dec_key, dec_value, enc_key, enc_value)
        
        # --- ATTENTION ---
        use_flash = block_masks is None  #Only happens if we're in inference mode AND on a gpu that can handle flash attn
        if use_flash:
            #Flash attention expects [B, L, N_h, D_h]
            query, enc_key, dec_key, enc_value, dec_value = self.reshape_to_attend(query, enc_key, dec_key, enc_value, dec_value)
            
            dec_output, dec_lse, _ = flash_attn_func(query, dec_key, dec_value, causal=True, return_attn_probs = True)
            enc_output, enc_lse, _ = flash_attn_func(query, enc_key, enc_value, causal=False, return_attn_probs = True)
        else:
            dec_output, dec_aux = self.attend(query, dec_key, dec_value, block_mask = block_masks.get('self_mask'), return_lse=True)
            enc_output, enc_aux = self.attend(query, enc_key, enc_value, block_mask=block_masks.get('cross_mask'), return_lse=True) #[B, L, N_h]
            dec_lse, enc_lse = dec_aux.lse, enc_aux.lse
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
        