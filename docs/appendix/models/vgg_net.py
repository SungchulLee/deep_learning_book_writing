#!/usr/bin/env python3
"""
============================================================
VGGNet - Very Deep Convolutional Networks
============================================================

Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014)
Authors: Karen Simonyan, Andrew Zisserman (Visual Geometry Group, Oxford)
Link: https://arxiv.org/abs/1409.1556

Key Innovations:
- Demonstrated that network depth is critical for performance
- Used only 3x3 conv filters throughout (instead of varying sizes)
- Stacked multiple 3x3 convs to get effective receptive field of larger filters
- Simple and homogeneous architecture
- Runner-up in ILSVRC 2014 (7.3% error)

Architecture Variants:
- VGG11: 11 weight layers (8 conv + 3 fc)
- VGG13: 13 weight layers (10 conv + 3 fc)
- VGG16: 16 weight layers (13 conv + 3 fc) - Most popular
- VGG19: 19 weight layers (16 conv + 3 fc) - Deepest

This implementation: VGG16
- ~138 million parameters
- Input: 224x224x3
- Output: 1000 classes

Design Principle:
Two 3x3 conv layers = effective 5x5 receptive field with fewer parameters
Three 3x3 conv layers = effective 7x7 receptive field with fewer parameters
"""

import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    VGG16 Implementation for Image Classification
    
    Architecture: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    Where numbers are conv filters and 'M' indicates MaxPooling
    
    Args:
        num_classes (int): Number of output classes. Default: 1000 (ImageNet)
        init_weights (bool): Whether to initialize weights. Default: True
    """
    
    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super(VGG16, self).__init__()
        
        # ============================================================
        # Feature Extraction (Convolutional Blocks)
        # ============================================================
        
        # Block 1: 224x224x3 -> 112x112x64
        # Two 3x3 convs give effective 5x5 receptive field
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 224x224x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 224x224x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112x64
        )
        
        # Block 2: 112x112x64 -> 56x56x128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 112x112x128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 112x112x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56x128
        )
        
        # Block 3: 56x56x128 -> 28x28x256
        # Three 3x3 convs give effective 7x7 receptive field
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 56x56x256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 56x56x256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 56x56x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28x256
        )
        
        # Block 4: 28x28x256 -> 14x14x512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 28x28x512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 28x28x512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 28x28x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x512
        )
        
        # Block 5: 14x14x512 -> 7x7x512
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 14x14x512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 14x14x512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 14x14x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7x512
        )
        
        # Adaptive pooling for flexibility with input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # ============================================================
        # Classification Head (Fully Connected Layers)
        # ============================================================
        self.classifier = nn.Sequential(
            # FC1: 7x7x512 = 25088 -> 4096
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout for regularization
            
            # FC2: 4096 -> 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            # FC3 (Output): 4096 -> num_classes
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights if specified
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VGG16
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Pass through all convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Adaptive pooling
        x = self.avgpool(x)
        
        # Flatten: (batch_size, 512, 7, 7) -> (batch_size, 512*7*7)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming (He) initialization
        This is better for ReLU activations than Xavier initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal initialization for FC layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ============================================================
# Demo and Testing
# ============================================================
if __name__ == "__main__":
    # Create model instance
    model = VGG16(num_classes=1000)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("VGG16 Model Summary")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    print(f"Output logits (first sample, first 10 classes): {logits[0, :10]}")
    print("=" * 60)
