"""
Transformer Model for Comparison
"""
import torch
import torch.nn as nn

class TransformerForComparison(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8, num_layers=6, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        return self.classifier(x)
