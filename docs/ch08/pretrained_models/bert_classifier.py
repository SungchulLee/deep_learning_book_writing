"""
BERT for Text Classification
"""
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.positional_encoding import PositionalEncoding

class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=768, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [seq, batch, dim]
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x[0]  # [CLS] token
        return self.classifier(x)
