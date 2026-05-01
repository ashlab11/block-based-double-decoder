"""
Layers for causal and combo-attn 
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .attention import SelfAttention, ComboAttention, CrossAttention

class BasicLayer(nn.Module):
    def __init__(self, dim, num_heads, seq_len, mlp_dim=None, scale = 1, attn_gating = False, use_checkpoint = False, mup = False, 
                 causal = True):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.scale = scale
        self.use_checkpoint = use_checkpoint
        mlp_dim = mlp_dim or 4 * dim

        # Self-attention block
        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads, seq_len=seq_len, gating=attn_gating, mup=mup, 
                                       causal=causal)
        self.norm1 = nn.RMSNorm(dim)

        # MLP block
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim, bias=False))

    def _forward(self, x, block_masks=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        x = self.self_attn(x, block_masks)
        x = residual + self.scale * x

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.scale * x

        return x

    def forward(self, x, block_masks=None):
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward, x, block_masks, use_reentrant=False)
        return self._forward(x, block_masks)

class StandardDecoderLayer(nn.Module):
    """Standard transformer decoder layer: self-attn → cross-attn → MLP (sequential)."""
    def __init__(self, dim, num_heads, seq_len, mlp_dim=None, use_checkpoint=False, mup=False):
        super(StandardDecoderLayer, self).__init__()
        self.use_checkpoint = use_checkpoint
        mlp_dim = mlp_dim or 4 * dim

        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads, seq_len=seq_len, mup=mup, causal=True)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, seq_len=seq_len, mup=mup)

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.norm3 = nn.RMSNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim, bias=False))

    def _forward(self, x, encoder_output, block_masks, decoder_input_positions=None):
        #Causal for decoder
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, input_pos=decoder_input_positions)
        x = residual + x

        # Cross-attention (to encoder output)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, encoder_output, block_masks=block_masks,
                            decoder_input_positions=decoder_input_positions)
        x = residual + x

        # MLP
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = residual + x

        return x

    def forward(self, x, encoder_output, block_masks, decoder_input_positions=None):
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward, x, encoder_output, block_masks,
                                   decoder_input_positions, use_reentrant=False)
        return self._forward(x, encoder_output, block_masks, decoder_input_positions)


class ComboDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, seq_len, shared = True, mlp_dim=None, logit_biases = False, use_checkpoint = False, mup = False):
        super(ComboDecoderLayer, self).__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        mlp_dim = mlp_dim or 4 * dim

        # Self-attention block
        self.combo_attn = ComboAttention(dim=dim, num_heads=num_heads, seq_len=seq_len, shared=shared, logit_biases=logit_biases, mup=mup)
        self.norm1 = nn.RMSNorm(dim)

        # MLP block
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim, bias=False))

    def _forward(self, x, encoder_output, block_masks, decoder_input_positions=None):
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

    def forward(self, x, encoder_output, block_masks, decoder_input_positions=None):
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward, x, encoder_output, block_masks, decoder_input_positions, use_reentrant=False)
        return self._forward(x, encoder_output, block_masks, decoder_input_positions)