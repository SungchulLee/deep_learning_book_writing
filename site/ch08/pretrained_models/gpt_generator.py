"""
GPT Text Generator
"""
import torch
import torch.nn as nn

class GPTGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        x = x.transpose(0, 1)
        x = self.transformer(x, x, tgt_mask=mask)
        x = x.transpose(0, 1)
        
        return self.fc_out(x)
