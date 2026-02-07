#!/usr/bin/env python3
'''
GhostNet - More Features from Cheap Operations
Paper: "GhostNet: More Features from Cheap Operations" (2020)
Key: Ghost modules generate more features with fewer parameters
'''
import torch
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, out_channels)
        
        if stride > 1:
            self.conv_dw = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(out_channels)
        self.stride = stride
        
        self.ghost2 = GhostModule(out_channels, out_channels, ratio=1)
        
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
        x = self.ghost2(x)
        return x + self.shortcut(residual)

class GhostNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        
        self.blocks = nn.Sequential(
            GhostBottleneck(16, 16),
            GhostBottleneck(16, 24, 2),
            GhostBottleneck(24, 24),
        )
        
        self.conv_head = nn.Conv2d(24, 960, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.act2 = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(960, num_classes)
    
    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act2(self.bn2(self.conv_head(x)))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

if __name__ == "__main__":
    model = GhostNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
