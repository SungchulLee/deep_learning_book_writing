#!/usr/bin/env python3
'''
ConvNeXt - A ConvNet for the 2020s
Paper: "A ConvNet for the 2020s" (2022)
Key: Modernized ResNet with design choices from Vision Transformers
'''
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.LayerNorm([96, 56, 56])
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean([2, 3])
        x = self.norm(x)
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
