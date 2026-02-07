#!/usr/bin/env python3
"""
================================================================================
GoogLeNet (Inception v1) - Going Deeper with Convolutions
================================================================================

Paper: "Going Deeper with Convolutions" (CVPR 2015)
Authors: Christian Szegedy et al. (Google)
Link: https://arxiv.org/abs/1409.4842

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
GoogLeNet introduced the Inception module, which runs multiple convolution
operations in parallel and concatenates results. This allowed for much deeper
and wider networks while keeping computational cost manageable.

- **ILSVRC 2014 Winner**: 6.67% top-5 error (beat VGG)
- **22 layers deep** but only ~6.8M parameters (12× fewer than AlexNet!)
- **No large FC layers**: Used Global Average Pooling

================================================================================
THE KEY INSIGHT: INCEPTION MODULE
================================================================================

Problem: What filter size should we use? 
- 1×1: Point-wise operations, channel mixing
- 3×3: Local features
- 5×5: Larger receptive field
- MaxPool: Alternative local evidence

Solution: Use ALL of them in parallel!

Naive Inception Module:
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Input Feature Map                                  │
│                                   │                                           │
│           ┌───────────┬───────────┼───────────┬───────────┐                  │
│           │           │           │           │           │                  │
│           ▼           ▼           ▼           ▼           │                  │
│       ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐       │                  │
│       │ 1×1   │   │ 3×3   │   │ 5×5   │   │ 3×3   │       │                  │
│       │ Conv  │   │ Conv  │   │ Conv  │   │MaxPool│       │                  │
│       └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘       │                  │
│           │           │           │           │           │                  │
│           └───────────┴───────────┴───────────┴───────────┘                  │
│                                   │                                           │
│                                   ▼                                           │
│                       Filter Concatenation (along channels)                   │
└──────────────────────────────────────────────────────────────────────────────┘

Problem: 5×5 convolutions are expensive!
- 192 input → 5×5×192×32 = 153,600 parameters per layer!

================================================================================
DIMENSIONALITY REDUCTION WITH 1×1 CONVOLUTIONS
================================================================================

Solution: Use 1×1 convolutions to reduce dimensions before expensive operations.

Optimized Inception Module:
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Input (192 channels)                               │
│                                   │                                           │
│     ┌─────────────┬───────────────┼───────────────┬─────────────┐            │
│     │             │               │               │             │            │
│     ▼             ▼               ▼               ▼             │            │
│ ┌───────┐     ┌───────┐       ┌───────┐      ┌───────┐         │            │
│ │ 1×1   │     │ 1×1   │       │ 1×1   │      │ 3×3   │         │            │
│ │ Conv  │     │ Conv  │       │ Conv  │      │MaxPool│         │            │
│ │ (64)  │     │ (96)  │       │ (16)  │      │       │         │            │
│ └───┬───┘     └───┬───┘       └───┬───┘      └───┬───┘         │            │
│     │             │               │              │              │            │
│     │             ▼               ▼              │              │            │
│     │         ┌───────┐       ┌───────┐         │              │            │
│     │         │ 3×3   │       │ 5×5   │         │              │            │
│     │         │ Conv  │       │ Conv  │         ▼              │            │
│     │         │ (128) │       │ (32)  │     ┌───────┐          │            │
│     │         └───┬───┘       └───┬───┘     │ 1×1   │          │            │
│     │             │               │         │ Conv  │          │            │
│     │             │               │         │ (32)  │          │            │
│     │             │               │         └───┬───┘          │            │
│     │             │               │             │              │            │
│     └─────────────┴───────────────┴─────────────┴──────────────┘            │
│                                   │                                          │
│                                   ▼                                          │
│                   Output: 64 + 128 + 32 + 32 = 256 channels                  │
└──────────────────────────────────────────────────────────────────────────────┘

Parameter savings:
- Without 1×1: 192×5×5×32 = 153,600
- With 1×1:    192×1×1×16 + 16×5×5×32 = 3,072 + 12,800 = 15,872
- ~10× reduction!

================================================================================
AUXILIARY CLASSIFIERS
================================================================================

Problem: Vanishing gradients in very deep networks
Solution: Add auxiliary classifiers at intermediate layers

┌──────────────────────────────────────────────────────────────────────────────┐
│  Input → ... → Inception_4a → AuxClassifier₁ → Inception_4b → ...           │
│                     │                                                        │
│                     └→ Loss₁ (weighted 0.3)                                  │
│                                                                              │
│  ... → Inception_4d → AuxClassifier₂ → Inception_4e → ... → MainClassifier  │
│              │                                                    │          │
│              └→ Loss₂ (weighted 0.3)                    Loss_main (1.0)     │
│                                                                              │
│  Total Loss = Loss_main + 0.3 × Loss₁ + 0.3 × Loss₂                         │
└──────────────────────────────────────────────────────────────────────────────┘

Benefits:
- Provides gradient signal at intermediate layers
- Acts as regularization
- Removed during inference (only main classifier used)

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: CNN (parallel convolutions, 1×1 convolutions)
- Ch04: Image Classification (Inception architecture)
- Ch21: Model Compression (efficient architecture design)

Related: resnet.py (skip connections), densenet.py (dense connections)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InceptionModule(nn.Module):
    """
    Inception Module with dimensionality reduction
    
    Applies 1×1, 3×3, 5×5 convolutions and 3×3 maxpool in parallel,
    then concatenates outputs.
    
    Args:
        in_channels: Number of input channels
        ch1x1: Output channels for 1×1 conv branch
        ch3x3_reduce: Channels after 1×1 reduction before 3×3
        ch3x3: Output channels for 3×3 conv branch
        ch5x5_reduce: Channels after 1×1 reduction before 5×5
        ch5x5: Output channels for 5×5 conv branch
        pool_proj: Output channels after pooling projection
    
    Output channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
    """
    
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int
    ):
        super(InceptionModule, self).__init__()
        
        # Branch 1: 1×1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1×1 → 3×3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1×1 → 5×5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: MaxPool → 1×1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all branch outputs"""
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionAux(nn.Module):
    """
    Auxiliary Classifier for GoogLeNet
    
    Provides additional gradient signal during training.
    Removed during inference.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
    """
    
    def __init__(self, in_channels: int, num_classes: int):
        super(InceptionAux, self).__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) Implementation
    
    Args:
        num_classes: Number of output classes. Default: 1000
        aux_logits: Whether to use auxiliary classifiers. Default: True
        init_weights: Whether to initialize weights. Default: True
    
    Example:
        >>> model = GoogLeNet(num_classes=1000, aux_logits=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> if model.training:
        ...     output, aux1, aux2 = model(x)
        ... else:
        ...     output = model(x)
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        init_weights: bool = True
    ):
        super(GoogLeNet, self).__init__()
        
        self.aux_logits = aux_logits
        
        # ====================================================================
        # STEM (Initial convolutions)
        # ====================================================================
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ====================================================================
        # INCEPTION MODULES
        # ====================================================================
        # Inception 3a, 3b
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)  # 256
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)  # 480
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Inception 4a, 4b, 4c, 4d, 4e
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)  # 512
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)  # 512
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)  # 512
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)  # 528
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)  # 832
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Inception 5a, 5b
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)  # 832
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)  # 1024
        
        # ====================================================================
        # AUXILIARY CLASSIFIERS
        # ====================================================================
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)  # After 4a
            self.aux2 = InceptionAux(528, num_classes)  # After 4d
        
        # ====================================================================
        # CLASSIFIER
        # ====================================================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through GoogLeNet
        
        Returns:
            If training with aux_logits: (main_output, aux1_output, aux2_output)
            Otherwise: main_output
        """
        # Stem
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        # Inception 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        # Inception 4
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
        
        # Inception 5
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# DEMO AND TESTING
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GoogLeNet (Inception v1) Model Summary")
    print("=" * 70)
    
    model = GoogLeNet(num_classes=1000, aux_logits=True)
    total_params, _ = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)
    
    x = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    # Training mode (returns auxiliary outputs)
    model.train()
    output, aux1, aux2 = model(x)
    print(f"Training - Main output: {output.shape}, Aux1: {aux1.shape}, Aux2: {aux2.shape}")
    
    # Evaluation mode (returns only main output)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Eval - Output: {output.shape}")
    print("=" * 70)
