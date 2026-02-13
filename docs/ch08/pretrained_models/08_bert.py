#!/usr/bin/env python3
'''
BERT - Bidirectional Encoder Representations from Transformers
Paper: "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
Key: Masked language modeling, next sentence prediction
'''
import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size=30000, d_model=768, max_len=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)
        self.segment = nn.Embedding(3, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, segment_label):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        embedding = self.token(x) + self.position(pos) + self.segment(segment_label)
        return self.norm(embedding)

class BERT(nn.Module):
    def __init__(self, vocab_size=30000, d_model=768, n_layers=12, heads=12):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.nsp = nn.Linear(d_model, 2)
    
    def forward(self, x, segment_label):
        x = self.embedding(x, segment_label)
        x = self.transformer(x)
        return self.fc(x), self.nsp(x[:, 0])

if __name__ == "__main__":
    model = BERT()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
