#!/usr/bin/env python3
'''
PSPNet - Pyramid Scene Parsing Network
Paper: "Pyramid Scene Parsing Network" (2017)
Key: Pyramid pooling module to aggregate multi-scale context
'''
import torch
import torch.nn as nn

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
    
    def forward(self, x):
        h, w = x.shape[2:]
        out = [x]
        for stage in self.stages:
            out.append(torch.nn.functional.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(out, dim=1)

class PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.ppm = PyramidPooling(64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64 + 64, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1(x)
        x = self.ppm(x)
        x = self.final(x)
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    model = PSPNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
