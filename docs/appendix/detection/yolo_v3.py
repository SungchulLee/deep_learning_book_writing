#!/usr/bin/env python3
'''
YOLOv3 - You Only Look Once v3
Paper: "YOLOv3: An Incremental Improvement" (2018)
Key: Single-shot object detection, predicts at 3 scales
'''
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1)
        self.conv2 = ConvBlock(channels // 2, channels, 3)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # Darknet-53 backbone (simplified)
        self.conv1 = ConvBlock(3, 32, 3)
        self.conv2 = ConvBlock(32, 64, 3, stride=2)
        self.res1 = ResidualBlock(64)
        
        self.conv3 = ConvBlock(64, 128, 3, stride=2)
        self.res2 = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])
        
        self.conv4 = ConvBlock(128, 256, 3, stride=2)
        self.res3 = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])
        
        # Detection heads at different scales
        self.detect1 = nn.Conv2d(256, (5 + num_classes) * 3, 1)  # 3 anchors per scale
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        
        detections = self.detect1(x)
        return detections

if __name__ == "__main__":
    model = YOLOv3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
