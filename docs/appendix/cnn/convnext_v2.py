#!/usr/bin/env python3
'''
ConvNeXt V2 - Modern ConvNet with Improved Design
Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (2023)
Key: Global Response Normalization (GRN), improved training strategies
'''
import torch
import torch.nn as nn

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(96) for _ in range(3)])
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean([2, 3])
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXtV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
