#!/usr/bin/env python3
'''
Faster R-CNN - Towards Real-Time Object Detection with Region Proposal Networks
Paper: "Faster R-CNN: Towards Real-Time Object Detection" (2015)
Key: RPN for region proposals, end-to-end trainable
'''
import torch
import torch.nn as nn

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # RPN
        self.rpn = nn.Sequential(
            nn.Conv2d(64, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.rpn_cls = nn.Conv2d(512, 2 * 9, 1)  # 2 classes * 9 anchors
        self.rpn_reg = nn.Conv2d(512, 4 * 9, 1)  # 4 coords * 9 anchors
        
        # ROI pooling and classification
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
    
    def forward(self, x):
        feat = self.features(x)
        
        # RPN
        rpn_feat = self.rpn(feat)
        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        
        return {'features': feat, 'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg}

if __name__ == "__main__":
    model = FasterRCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
