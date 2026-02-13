#!/usr/bin/env python3
"""
============================================================
GoogLeNet (Inception v1) - Going Deeper with Convolutions
============================================================

Paper: "Going Deeper with Convolutions" (2014)
Authors: Christian Szegedy et al. (Google)
Link: https://arxiv.org/abs/1409.4842

Key Innovations:
- Introduced Inception modules: parallel conv operations with different filter sizes
- Won ILSVRC 2014 with 6.67% error rate
- 22 layers deep but only ~6.8 million parameters (12x fewer than AlexNet)
- Efficient use of computational resources through 1x1 convolutions
- Auxiliary classifiers during training to combat vanishing gradients
- No fully connected layers except the final classifier
- Global Average Pooling instead of large FC layers

Inception Module Concept:
Instead of choosing one filter size (1x1, 3x3, or 5x5), apply all in parallel
and concatenate the results. This allows the network to learn which scale
of features is most important for a given task.

Architecture:
- 9 Inception modules
- 2 auxiliary classifiers (removed during inference)
- Global average pooling
- Minimal FC layers
- ~6.8 million parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    Inception Module (Naive version with dimensionality reduction)
    
    Applies 1x1, 3x3, 5x5 convolutions and 3x3 maxpooling in parallel,
    then concatenates the outputs.
    
    Uses 1x1 convolutions before 3x3 and 5x5 for dimensionality reduction
    to reduce computational cost.
    
    Args:
        in_channels: Number of input channels
        ch1x1: Number of 1x1 conv filters
        ch3x3_reduce: Number of 1x1 conv filters before 3x3 conv (reduction)
        ch3x3: Number of 3x3 conv filters
        ch5x5_reduce: Number of 1x1 conv filters before 5x5 conv (reduction)
        ch5x5: Number of 5x5 conv filters
        pool_proj: Number of 1x1 conv filters after pooling
    """
    
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # Branch 1: 1x1 convolution
        # Captures features at the pixel level
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 conv -> 3x3 conv
        # 1x1 reduces dimensions, then 3x3 captures local spatial features
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 conv -> 5x5 conv
        # Captures larger spatial patterns
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 max pool -> 1x1 conv
        # Provides alternative local evidence
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Apply all branches in parallel
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate along channel dimension
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) Implementation
    
    Args:
        num_classes (int): Number of output classes. Default: 1000
        aux_logits (bool): Whether to use auxiliary classifiers during training
        init_weights (bool): Whether to initialize weights
    """
    
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        
        # Initial convolutional layers
        # These are more traditional conv layers before inception modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Inception modules
        #                   in,  1x1, 3x3_red, 3x3, 5x5_red, 5x5, pool_proj
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Auxiliary classifiers (used during training to help with gradient flow)
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # Initial conv layers
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        
        # Inception blocks
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        
        # Auxiliary classifier 1
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        # Auxiliary classifier 2
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        # Return auxiliary outputs during training
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class InceptionAux(nn.Module):
    """
    Auxiliary classifier to combat vanishing gradients in deep networks.
    Provides additional gradient signal during training.
    """
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = GoogLeNet(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    print("GoogLeNet (Inception v1) Model Summary")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)
    
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print("=" * 60)
