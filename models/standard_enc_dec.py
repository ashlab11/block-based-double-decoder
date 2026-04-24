"""
Standard encoder-decoder transformer with separate self-attention and
cross-attention in the decoder (no combo/sigmoid gating).

This serves as the baseline enc-dec model against which the Double Decoder's
combo attention mechanism is compared.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.layers import CausalLayer, StandardDecoderLayer
from components.block_masks import create_sft_masks, create_pretrain_masks, create_inference_masks, create_masks
from components.initialization import initialize_model


class StandardEncDec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 3,
        seq_len: int = 2048,
        mlp_dim: int = None,
        label_pad_token_id: int = -100,
        init_strategy: str = "xavier_uniform",
        gradient_checkpointing: bool = False,
        **kwargs  # absorb DD-specific args like shared, logit_biases
    ):
        super(StandardEncDec, self).__init__()
        self.dim = dim
        self.label_pad_token_id = label_pad_token_id
        self.seq_len = seq_len

        # Shared token embeddings
        self.embedding = nn.Embedding(vocab_size, dim)

        # Encoder (standard causal self-attention)
        self.encoder_layers = nn.ModuleList([
            CausalLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, seq_len=seq_len,
                        use_checkpoint=gradient_checkpointing)
            for _ in range(num_encoder_layers)
        ])

        # Decoder (standard self-attn + cross-attn + MLP)
        self.decoder_layers = nn.ModuleList([
            StandardDecoderLayer(dim=dim, num_heads=num_heads, seq_len=seq_len,
                                mlp_dim=mlp_dim, use_checkpoint=gradient_checkpointing)
            for _ in range(num_decoder_layers)
        ])

        # Norms and output
        self.encoder_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.output_projection.weight = self.embedding.weight

        initialize_model(self, init_strategy)

    def encode(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        return x

    def decode(self, input_ids, encoder_output, block_masks=None, decoder_input_positions=None):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, block_masks,
                      decoder_input_positions=decoder_input_positions)
        x = self.output_norm(x)
        logits = self.output_projection(x)
        return logits

    def forward(
        self,
        input_ids=None,
        labels=None,
        blocks=None,
        decoder_input_positions=None,
        sft=False,
        encoder_input_ids=None,
        decoder_input_ids=None
    ):
        assert blocks is not None, "Blocks are required for PT and SFT"

        if sft:
            assert encoder_input_ids is not None and decoder_input_ids is not None
        else:
            assert input_ids is not None

        encoder_input_ids = encoder_input_ids if sft else input_ids
        decoder_input_ids = decoder_input_ids if sft else input_ids
        batch_size = encoder_input_ids.shape[0]

        block_masks = create_masks(
            batch_size=batch_size, blocks=blocks, device=encoder_input_ids.device,
            input_ids=input_ids, encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids, sft=sft)

        encoder_output = self.encode(encoder_input_ids)
        logits = self.decode(decoder_input_ids, encoder_output, block_masks,
                             decoder_input_positions)

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.label_pad_token_id)
        else:
            loss = None

        return {"loss": loss, "logits": logits}
