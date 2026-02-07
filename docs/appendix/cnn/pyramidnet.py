#!/usr/bin/env python3
'''
PyramidNet - Deep Pyramidal Residual Networks
Paper: "Deep Pyramidal Residual Networks" (2017)
Key: Gradually increases feature map dimensions
'''
import torch
import torch.nn as nn

class PyramidBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
            )
    
    def forward(self, x):
        out = self.conv1(torch.nn.functional.relu(self.bn1(x)))
        out = self.conv2(torch.nn.functional.relu(self.bn2(out)))
        out = self.bn3(out)
        
        # Zero padding for dimension matching
        shortcut = self.shortcut(x)
        if shortcut.size(1) != out.size(1):
            pad_size = out.size(1) - shortcut.size(1)
            shortcut = torch.nn.functional.pad(shortcut, (0, 0, 0, 0, 0, pad_size))
        
        out += shortcut
        return out

class PyramidNet(nn.Module):
    def __init__(self, num_classes=1000, alpha=48, depth=110):
        super().__init__()
        n = (depth - 2) // 6
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        
        add_rate = alpha / (3 * n)
        in_channels = 16
        
        layers = []
        for i in range(3 * n):
            out_channels = round(16 + add_rate * (i + 1))
            stride = 2 if i == n or i == 2 * n else 1
            layers.append(PyramidBasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = torch.nn.functional.relu(self.bn(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = PyramidNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
