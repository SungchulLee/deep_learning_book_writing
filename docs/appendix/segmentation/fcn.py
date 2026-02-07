#!/usr/bin/env python3
'''
FCN - Fully Convolutional Networks for Semantic Segmentation
Paper: "Fully Convolutional Networks for Semantic Segmentation" (2015)
Key: First end-to-end CNN for semantic segmentation, replaces FC with conv layers
'''
import torch
import torch.nn as nn

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # VGG-style encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Score layers for skip connections
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        
        # Upsampling
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        score = self.score_pool3(conv3)
        upscore = self.upscore(score)
        
        return upscore

if __name__ == "__main__":
    model = FCN8s()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
