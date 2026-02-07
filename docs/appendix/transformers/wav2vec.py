#!/usr/bin/env python3
'''
Wav2Vec 2.0 - Self-Supervised Learning for Speech Recognition
Paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)
Key: Contrastive learning for speech, quantization, transformer encoder
'''
import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 512, 10, stride=5, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class Wav2Vec2(nn.Module):
    def __init__(self, d_model=768, n_layers=12, n_heads=12):
        super().__init__()
        self.feature_extractor = FeatureEncoder()
        self.feature_projection = nn.Linear(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.quantizer = nn.Linear(d_model, 320)
    
    def forward(self, x):
        # x: [batch, 1, time]
        features = self.feature_extractor(x)
        features = features.transpose(1, 2)
        features = self.feature_projection(features)
        
        context = self.transformer(features)
        
        quantized = self.quantizer(context)
        
        return context, quantized

if __name__ == "__main__":
    model = Wav2Vec2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 1, 16000)
    print(f"Input: {x.shape}")
