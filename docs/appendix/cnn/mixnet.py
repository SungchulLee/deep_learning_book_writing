#!/usr/bin/env python3
'''
MixNet - Mixed Depthwise Convolutional Kernels
Paper: "MixConv: Mixed Depthwise Convolutional Kernels" (2019)
Key: Multiple kernel sizes in a single depthwise convolution layer
'''
import torch
import torch.nn as nn

class MixConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.groups = len(kernel_sizes)
        assert out_channels % self.groups == 0
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels // self.groups, out_channels // self.groups, 
                     k, padding=k//2, groups=in_channels // self.groups)
            for k in kernel_sizes
        ])
    
    def forward(self, x):
        chunks = torch.chunk(x, self.groups, dim=1)
        outs = [conv(chunk) for conv, chunk in zip(self.convs, chunks)]
        return torch.cat(outs, dim=1)

class MixNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.mixconv = MixConv2d(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = torch.nn.functional.relu6(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu6(self.bn2(self.mixconv(out)))
        out = self.bn3(self.conv2(out))
        if x.shape == out.shape:
            out = out + x
        return out

class MixNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.blocks = nn.Sequential(
            MixNetBlock(16, 24),
            MixNetBlock(24, 24),
        )
        self.conv_head = nn.Conv2d(24, 1536, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1536)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu6(self.bn1(self.stem(x)))
        x = self.blocks(x)
        x = torch.nn.functional.relu6(self.bn2(self.conv_head(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = MixNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
