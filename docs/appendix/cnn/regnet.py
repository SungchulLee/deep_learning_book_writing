#!/usr/bin/env python3
'''
RegNet - Designing Network Design Spaces
Paper: "Designing Network Design Spaces" (2020)
Key: Quantized linear parametrization of network width/depth
'''
import torch
import torch.nn as nn

class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class RegNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.layer1 = self._make_layer(32, 64, 2)
        self.head = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            layers.append(RegNetBlock(in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

if __name__ == "__main__":
    model = RegNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
