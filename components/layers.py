"""
Layers for causal and combo-attn 
"""

import torch
import torch.nn as nn
from .attention import SelfAttention, ComboAttention

class CausalLayer(nn.Module):
    def __init__(self, dim, num_heads, seq_len, mlp_dim=None, scale = 1, attn_gating = False):
        super(CausalLayer, self).__init__()
        self.dim = dim
        self.scale = scale
        mlp_dim = mlp_dim or 4 * dim 
        
        # Self-attention block
        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads, seq_len=seq_len, gating = attn_gating)
        self.norm1 = nn.RMSNorm(dim)
        
        # MLP block
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim, bias=False))
                
    def forward(self, x):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        x = self.self_attn(x)
        x = residual + self.scale * x
        
        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.scale * x
        
        return x

class ComboDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, seq_len, shared = True, mlp_dim=None, logit_biases = False):
        super(ComboDecoderLayer, self).__init__()
        self.dim = dim
        mlp_dim = mlp_dim or 4 * dim 
        
        # Self-attention block
        self.combo_attn = ComboAttention(dim=dim, num_heads=num_heads, seq_len=seq_len, shared=shared, logit_biases=logit_biases, adapter_rank=adapter_rank)
        self.norm1 = nn.RMSNorm(dim)
        
        # MLP block
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim, bias=False))
        
    def forward(self, x, encoder_output, block_masks, decoder_input_positions=None):
        # Self-attention with residual
        residual = x
        
        #Combo attention
        x = self.norm1(x)
        x = self.combo_attn(x, encoder_output, block_masks=block_masks, decoder_input_positions=decoder_input_positions)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x