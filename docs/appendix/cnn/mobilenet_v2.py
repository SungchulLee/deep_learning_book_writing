#!/usr/bin/env python3
'''
MobileNetV2 - Efficient CNN for Mobile Devices
Paper: "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
Key: Inverted residual blocks, depthwise separable convolutions
'''
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        # Implementation details...
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)

if __name__ == "__main__":
    model = MobileNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
