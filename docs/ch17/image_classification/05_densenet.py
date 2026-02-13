#!/usr/bin/env python3
'''
DenseNet-121 - Densely Connected Networks
Paper: "Densely Connected Convolutional Networks" (2017)
Key: Each layer connects to all subsequent layers, feature reuse
'''
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim=1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        return self.pool(self.conv(torch.relu(self.bn(x))))

class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        num_channels = 64
        self.dense1 = DenseBlock(6, num_channels, growth_rate)
        num_channels += 6 * growth_rate
        self.trans1 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense2 = DenseBlock(12, num_channels, growth_rate)
        num_channels += 12 * growth_rate
        self.trans2 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense3 = DenseBlock(24, num_channels, growth_rate)
        num_channels += 24 * growth_rate
        self.trans3 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense4 = DenseBlock(16, num_channels, growth_rate)
        num_channels += 16 * growth_rate
        
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)
    
    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = torch.relu(self.bn_final(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = DenseNet121()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
