#!/usr/bin/env python3
'''
CoordConv - Adding Spatial Coordinates to CNNs
Paper: "An Intriguing Failing of Convolutional Neural Networks" (2018)
Key: Adds coordinate channels to help with spatial reasoning
'''
import torch
import torch.nn as nn

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
    
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Create coordinate grids
        xx_channel = torch.arange(width, dtype=x.dtype, device=x.device).repeat(1, height, 1)
        yy_channel = torch.arange(height, dtype=x.dtype, device=x.device).repeat(1, width, 1).transpose(1, 2)
        
        xx_channel = xx_channel / (width - 1)
        yy_channel = yy_channel / (height - 1)
        
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
        
        ret = torch.cat([x, xx_channel, yy_channel], dim=1)
        
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)
        
        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

class CoordConvNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = CoordConv(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.conv2 = CoordConv(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CoordConvNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
