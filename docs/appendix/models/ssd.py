#!/usr/bin/env python3
'''
SSD - Single Shot MultiBox Detector
Paper: "SSD: Single Shot MultiBox Detector" (2016)
Key: Multi-scale feature maps for detection, single forward pass
'''
import torch
import torch.nn as nn

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Base network (VGG-like)
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Extra layers for multi-scale detection
        self.extras = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Detection heads
        self.loc = nn.Conv2d(256, 4 * 4, 3, padding=1)  # 4 default boxes * 4 coords
        self.conf = nn.Conv2d(256, 4 * num_classes, 3, padding=1)  # 4 boxes * num_classes
    
    def forward(self, x):
        x = self.base(x)
        x = self.extras(x)
        
        loc = self.loc(x)
        conf = self.conf(x)
        
        return {'loc': loc, 'conf': conf}

if __name__ == "__main__":
    model = SSD()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
