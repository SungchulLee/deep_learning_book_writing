#!/usr/bin/env python3
'''
EfficientNet - Rethinking Model Scaling
Paper: "EfficientNet: Rethinking Model Scaling for CNNs" (2019)
Key: Compound scaling (depth, width, resolution), highly efficient
'''
import torch
import torch.nn as nn

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            MBConv(32, 16, expand_ratio=1),
            MBConv(16, 24, stride=2),
            MBConv(24, 40, stride=2),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(40, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

if __name__ == "__main__":
    model = EfficientNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
