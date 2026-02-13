"""
BERT-style Model
"""
import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder

class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff)
        self.pooler = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        encoded = self.encoder(x, mask)
        pooled = torch.tanh(self.pooler(encoded[:, 0, :]))  # [CLS] token
        return encoded, pooled
