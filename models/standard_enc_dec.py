"""
Standard encoder-decoder transformer with separate self-attention and
cross-attention in the decoder (no combo/sigmoid gating).

This serves as the baseline enc-dec model against which the Double Decoder's
combo attention mechanism is compared.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from components.layers import BasicLayer, StandardDecoderLayer
from components.block_masks import create_masks, create_masks_ED, create_inference_masks
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
        mup_base_dim: int = 0,
        **kwargs  # absorb DD-specific args like shared, logit_biases
    ):
        super(StandardEncDec, self).__init__()
        self.dim = dim
        self.label_pad_token_id = label_pad_token_id
        self.seq_len = seq_len
        self.mup_base_dim = mup_base_dim
        mup = mup_base_dim > 0
        self.mup_mult = mup_base_dim / dim if mup else 1.0
        # With tied embedding (constant std init), readout grows like √dim from
        # the dim-wide dot product, not like dim — so the readout multiplier is
        # √(base/dim), not base/dim. Verified by mup_full_check.py Check 6.
        self.mup_readout_mult = math.sqrt(self.mup_mult)

        # Shared token embeddings
        self.embedding = nn.Embedding(vocab_size, dim)

        # Encoder (standard causal self-attention)
        self.encoder_layers = nn.ModuleList([
            BasicLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, seq_len=seq_len,
                        use_checkpoint=gradient_checkpointing, mup=mup, causal=False)
            for _ in range(num_encoder_layers)
        ])

        # Decoder (standard self-attn + cross-attn + MLP)
        self.decoder_layers = nn.ModuleList([
            StandardDecoderLayer(dim=dim, num_heads=num_heads, seq_len=seq_len,
                                mlp_dim=mlp_dim, use_checkpoint=gradient_checkpointing, mup=mup)
            for _ in range(num_decoder_layers)
        ])

        # Norms and output
        self.encoder_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.output_projection.weight = self.embedding.weight

        initialize_model(self, init_strategy)

    def encode(self, input_ids, block_masks = None):
        x = self.embedding(input_ids)
        for layer in self.encoder_layers:
            x = layer(x, block_masks)
        x = self.encoder_norm(x)
        return x

    def decode(self, input_ids, encoder_output, block_masks=None, decoder_input_positions=None):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, block_masks,
                      decoder_input_positions=decoder_input_positions)
        x = self.output_norm(x)
        logits = self.output_projection(x) * self.mup_readout_mult
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
        assert blocks is not None, "Masks are required for PT/SFT"

        if sft:
            assert encoder_input_ids is not None and decoder_input_ids is not None
        else:
            assert input_ids is not None

        encoder_input_ids = encoder_input_ids if sft else input_ids
        decoder_input_ids = decoder_input_ids if sft else input_ids
        batch_size = encoder_input_ids.shape[0]
        
        causal_mask, cross_mask = create_masks_ED(batch_size=batch_size, blocks=blocks, device=encoder_input_ids.device, 
                                      encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)

        encoder_output = self.encode(encoder_input_ids, causal_mask)
        logits = self.decode(decoder_input_ids, encoder_output, cross_mask,
                             decoder_input_positions)

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.label_pad_token_id)
        else:
            loss = None

        return {"loss": loss, "logits": logits}
    
    def generate(self, input, max_new_tokens, tokenizer, temp=1.0, top_k=50, top_p=0.9, device=torch.device('cuda:0')):
        """Given input prompt ids, generate a response"""
        self.eval()
        assistant_token_id = tokenizer.token_to_id("<assistant>")
        eos_token_id = tokenizer.token_to_id("</s>")
        
        input_ids = tokenizer.encode(input).ids # [L]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        #If we're in inference mode and on B200+ (for now), create block masks
        with torch.inference_mode():
            encoder_output = self.encode(input_ids) # [L, D]
        return_ids = torch.tensor([assistant_token_id]).to(device).unsqueeze(0)
        num_incoming_tokens = input_ids.shape[1]
        num_new_tokens = 1  # Assistant token
        
        decoder_input_positions = torch.tensor([num_incoming_tokens]).to(device).unsqueeze(0)
        
        while return_ids[0][-1] != eos_token_id and num_new_tokens < max_new_tokens: 
            #Create block masks if we're on B200 (for now)  
            if torch.cuda.get_device_capability()[0] >= 10:
                block_masks = create_inference_masks(device = device, enc_len = num_incoming_tokens, dec_len = num_new_tokens)
            else:
                block_masks = None
         
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = self.decode(input_ids = return_ids, encoder_output = encoder_output, decoder_input_positions = decoder_input_positions, 
                                     block_masks = block_masks)
            logits = logits[:, -1, :] / temp
            logits = logits.squeeze()
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                threshold = values[-1]
                logits[logits < threshold] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return_ids = torch.cat([return_ids, next_token.unsqueeze(0)], dim=-1)
            decoder_input_positions = torch.cat([decoder_input_positions, decoder_input_positions[:, -1].unsqueeze(1) + 1], dim=1)
            num_new_tokens += 1
        print(f"Is assistant token in return ids? {assistant_token_id in return_ids[0].tolist()}")
        return tokenizer.decode(return_ids[0].tolist())