#!/usr/bin/env python3
'''
RetinaNet - Focal Loss for Dense Object Detection
Paper: "Focal Loss for Dense Object Detection" (2017)
Key: Focal loss to handle class imbalance, Feature Pyramid Network (FPN)
'''
import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Bottom-up pathway
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Lateral connections
        self.lateral3 = nn.Conv2d(64, 256, 1)
        
        # Top-down pathway
        self.smooth = nn.Conv2d(256, 256, 3, 1, 1)
    
    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        
        # Lateral connection
        p3 = self.lateral3(c1)
        
        # Smooth
        p3 = self.smooth(p3)
        
        return [p3]

class RetinaNet(nn.Module):
    def __init__(self, num_classes=80, num_anchors=9):
        super().__init__()
        self.fpn = FPN()
        
        # Classification subnet
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * num_classes, 3, 1, 1)
        )
        
        # Box regression subnet
        self.box_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * 4, 3, 1, 1)
        )
    
    def forward(self, x):
        features = self.fpn(x)
        
        cls_outputs = []
        box_outputs = []
        
        for feat in features:
            cls_outputs.append(self.cls_subnet(feat))
            box_outputs.append(self.box_subnet(feat))
        
        return {'classifications': cls_outputs, 'regressions': box_outputs}

if __name__ == "__main__":
    model = RetinaNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
