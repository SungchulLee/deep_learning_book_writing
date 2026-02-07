#!/usr/bin/env python3
'''
ShuffleNet - Efficient CNN with Channel Shuffle Operation
Paper: "ShuffleNet: An Extremely Efficient CNN for Mobile Devices" (2017)
Key: Channel shuffle operation, group convolutions for efficiency
'''
import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 4
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        
        self.expand = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, 2, 1)
    
    def forward(self, x):
        out = self.bottleneck(x)
        out = channel_shuffle(out, 2)
        out = self.depthwise(out)
        out = self.expand(out)
        
        if self.stride == 2:
            out = torch.cat([out, self.shortcut(x)], 1)
        else:
            out += x
        return nn.functional.relu(out)

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

if __name__ == "__main__":
    model = ShuffleNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
