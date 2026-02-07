#!/usr/bin/env python3
'''
ResNeSt - Split-Attention Networks
Paper: "ResNeSt: Split-Attention Networks" (2020)
Key: Split-attention blocks with channel-wise attention across feature groups
'''
import torch
import torch.nn as nn

class SplitAttention(nn.Module):
    def __init__(self, channels, radix=2, groups=1):
        super().__init__()
        self.radix = radix
        self.groups = groups
        inter_channels = max(channels * radix // 4, 32)
        
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=groups)
        self.rsoftmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, self.radix, -1, *x.shape[2:])
        gap = x.sum(dim=1)
        gap = nn.functional.adaptive_avg_pool2d(gap, 1)
        
        atten = self.fc1(gap)
        atten = self.bn1(atten)
        atten = self.relu(atten)
        atten = self.fc2(atten)
        atten = atten.view(batch, self.radix, -1)
        atten = self.rsoftmax(atten)
        atten = atten.view(batch, self.radix, -1, 1, 1)
        
        out = (x * atten).sum(dim=1)
        return out

class ResNeStBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, radix=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * radix, 3, stride, 1, groups=radix, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * radix)
        self.split_attention = SplitAttention(out_channels, radix)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.split_attention(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNeSt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 and in_channels != 64 else 1
            layers.append(ResNeStBlock(in_channels if i == 0 else out_channels * 4, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = ResNeSt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
