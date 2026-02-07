#!/usr/bin/env python3
'''
Mask R-CNN - Instance Segmentation Framework
Paper: "Mask R-CNN" (2017)
Key: Extends Faster R-CNN by adding mask prediction branch
'''
import torch
import torch.nn as nn

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        # Simplified backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # RPN (Region Proposal Network)
        self.rpn_conv = nn.Conv2d(64, 512, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(512, 2 * 9, 1)  # 9 anchors, 2 classes (obj/not)
        self.rpn_reg = nn.Conv2d(512, 4 * 9, 1)  # 9 anchors, 4 coords
        
        # ROI Head
        self.roi_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        
        # Classification and bounding box regression
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # Mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # RPN
        rpn_feat = torch.nn.functional.relu(self.rpn_conv(features))
        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        
        # Simplified output
        return {'features': features, 'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg}

if __name__ == "__main__":
    model = MaskRCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
