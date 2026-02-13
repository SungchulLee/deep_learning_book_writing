#!/usr/bin/env python3
"""
============================================================
AlexNet - Deep Convolutional Neural Network
============================================================

Paper: "ImageNet Classification with Deep Convolutional Neural Networks" (2012)
Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
Link: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Key Innovations:
- Won ImageNet ILSVRC 2012 with 15.3% error rate (vs 26.2% second place)
- First deep CNN to win ImageNet, sparking the deep learning revolution
- Used ReLU activation instead of tanh/sigmoid for faster training
- Introduced Dropout for regularization
- Used Data Augmentation extensively
- GPU training with CUDA
- Local Response Normalization (LRN) - not included in this minimal version

Architecture Overview:
- 5 Convolutional layers
- 3 Fully connected layers
- ReLU activations
- Max pooling
- Dropout (0.5) in FC layers
- ~60 million parameters

Input: 224x224x3 RGB images
Output: 1000 class logits (ImageNet)
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet Implementation for Image Classification
    
    Args:
        num_classes (int): Number of output classes. Default: 1000 (ImageNet)
    """
    
    def __init__(self, num_classes: int = 1000):
        super(AlexNet, self).__init__()
        
        # ============================================================
        # Feature Extraction Layers (Convolutional Part)
        # ============================================================
        self.features = nn.Sequential(
            # Conv Layer 1: 224x224x3 -> 55x55x64
            # Large 11x11 kernel to capture broader features in the first layer
            # Stride of 4 reduces spatial dimensions quickly
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # ReLU is faster than tanh/sigmoid
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55x55x64 -> 27x27x64
            
            # Conv Layer 2: 27x27x64 -> 27x27x192
            # Increases depth to capture more complex features
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x192 -> 13x13x192
            
            # Conv Layer 3: 13x13x192 -> 13x13x384
            # First layer without pooling - captures higher-level features
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Layer 4: 13x13x384 -> 13x13x256
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Layer 5: 13x13x256 -> 13x13x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13x256 -> 6x6x256
        )
        
        # Adaptive pooling to handle slight variations in input size
        # Ensures output is always 6x6 regardless of exact input dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # ============================================================
        # Classification Layers (Fully Connected Part)
        # ============================================================
        self.classifier = nn.Sequential(
            # Dropout for regularization - randomly drops 50% of neurons during training
            # Prevents co-adaptation of neurons and reduces overfitting
            nn.Dropout(p=0.5),
            
            # FC Layer 1: 6x6x256 = 9216 -> 4096
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            
            # FC Layer 2: 4096 -> 4096
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            
            # FC Layer 3 (Output): 4096 -> num_classes
            # No activation here - outputs raw logits for CrossEntropyLoss
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through AlexNet
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract features through convolutional layers
        x = self.features(x)
        
        # Apply adaptive pooling
        x = self.avgpool(x)
        
        # Flatten the tensor: (batch_size, 256, 6, 6) -> (batch_size, 256*6*6)
        x = torch.flatten(x, start_dim=1)
        
        # Pass through classifier
        x = self.classifier(x)
        
        return x


# ============================================================
# Demo and Testing
# ============================================================
if __name__ == "__main__":
    # Create model instance
    model = AlexNet(num_classes=1000)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("AlexNet Model Summary")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    # Set model to evaluation mode for inference
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    print(f"Output logits (first sample, first 10 classes): {logits[0, :10]}")
    print("=" * 60)
