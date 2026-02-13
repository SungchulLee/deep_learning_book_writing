"""
GPT-style Model
"""
import torch
import torch.nn as nn
from transformer_decoder import TransformerDecoder

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072):
        super().__init__()
        self.decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    def forward(self, x, mask=None):
        return self.decoder(x, mask)
    
    def generate(self, start_tokens, max_len=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(start_tokens)
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                start_tokens = torch.cat([start_tokens, next_token], dim=1)
        return start_tokens
