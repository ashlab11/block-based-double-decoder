import torch
import torch.nn as nn
from components.layers import CausalLayer
import torch.nn.functional as F
from components.initialization import initialize_model

class DecoderOnlyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_heads: int = 9,
        num_layers: int = 14,
        seq_len: int = 1536,
        mlp_dim: int = None,
        label_pad_token_id: int = -100,
        init_strategy = "xavier_uniform", 
        **kwargs
    ):
        super(DecoderOnlyModel, self).__init__()
        self.dim = dim
        self.label_pad_token_id = label_pad_token_id
        # Token embeddings  
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Encoder
        self.layers = nn.ModuleList([
            CausalLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, seq_len=seq_len)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.output_projection.weight = self.embedding.weight
        
        # Initialize weights for fast convergence! 
        initialize_model(self, init_strategy)
        
    def forward(
        self,
        input_ids,
        labels = None):
        # Encode
        x = self.embedding(input_ids)
    
        for layer in self.layers:
            x = layer(x)
                
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # Computing loss here
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.label_pad_token_id
            )
        else:
            loss = None
        return {"loss": loss, "logits": logits}
    
    def generate(self, tokenizer, input, max_new_tokens, temp=1.0, top_k=50, top_p=0.9, device=torch.device('cpu')):
        """Given an input of a prompt, generate a response"""
        self.eval()
        bos_token_id = tokenizer.token_to_id("<s>")
        assistant_token_id = tokenizer.token_to_id("<assistant>")
        
        bos_token = torch.tensor([bos_token_id]).to(device).unsqueeze(0)
        assistant_token = torch.tensor([assistant_token_id]).to(device).unsqueeze(0)
        
        eos_token_id = tokenizer.token_to_id("</s>")
        
        input_ids = tokenizer.encode(input).ids # [L]
        original_len = len(input_ids) + 2 #+1 for assistant, bos
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        input_ids = torch.cat([bos_token, input_ids, assistant_token], dim=-1)
        
        while input_ids[0][-1] != eos_token_id and len(input_ids[0]) < max_new_tokens:
            with torch.inference_mode():
                logits = self.forward(input_ids)['logits']
            logits = logits[:, -1, :] / temp
            logits = logits.squeeze()
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                threshold = values[-1]
                logits[logits < threshold] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        return tokenizer.decode(input_ids[0].tolist()[original_len:])