"""
Module 33.4: EfficientNet - Compound Scaling (ADVANCED)

EfficientNet introduces compound scaling that uniformly scales network depth, width, and resolution
with a set of fixed scaling coefficients, achieving better accuracy and efficiency.

Key Concepts:
1. Compound scaling method (depth, width, resolution)
2. MBConv blocks with squeeze-and-excitation
3. Neural Architecture Search (NAS) baseline
4. Efficient mobile-friendly architecture

Paper: Tan & Le, 2019 - "EfficientNet: Rethinking Model Scaling"
"""

import torch
import torch.nn as nn
import math

class SwishActivation(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    1. Global average pooling (squeeze)
    2. Two FC layers with ReLU (excitation)
    3. Sigmoid activation
    4. Multiply with input (scale)
    """
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),  # Swish
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv) block.
    
    Architecture:
    1. Expansion: 1x1 conv to expand channels
    2. Depthwise: 3x3 or 5x5 depthwise conv
    3. Squeeze-Excitation
    4. Projection: 1x1 conv to project back
    5. Skip connection if stride=1 and in_ch=out_ch
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        # Squeeze-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(hidden_dim, se_channels)
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x

class EfficientNet(nn.Module):
    """
    EfficientNet architecture with compound scaling.
    
    Scaling parameters:
    - depth: number of layers
    - width: number of channels  
    - resolution: input image size
    
    Compound scaling: depth = α^φ, width = β^φ, resolution = γ^φ
    where α, β, γ are constants and φ is compound coefficient
    """
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=10, dropout=0.2):
        super().__init__()
        
        # Base configuration: [expand_ratio, channels, num_layers, stride, kernel_size]
        base_config = [
            [1, 16, 1, 1, 3],   # Stage 1
            [6, 24, 2, 2, 3],   # Stage 2
            [6, 40, 2, 2, 5],   # Stage 3
            [6, 80, 3, 2, 3],   # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]
        
        # Stem
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        # Build MBConv blocks
        in_channels = out_channels
        blocks = []
        for expand, channels, num_layers, stride, kernel in base_config:
            out_channels = self._round_filters(channels, width_mult)
            num_layers = self._round_repeats(num_layers, depth_mult)
            
            for i in range(num_layers):
                blocks.append(MBConvBlock(
                    in_channels, out_channels, kernel,
                    stride if i == 0 else 1, expand
                ))
                in_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes)
        )
    
    def _round_filters(self, filters, width_mult):
        """Round number of filters based on width multiplier."""
        filters *= width_mult
        new_filters = int(filters + 4) // 8 * 8
        new_filters = max(8, new_filters)
        if new_filters < 0.9 * filters:
            new_filters += 8
        return int(new_filters)
    
    def _round_repeats(self, repeats, depth_mult):
        """Round number of layers based on depth multiplier."""
        return int(math.ceil(depth_mult * repeats))
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

def efficientnet_b0(num_classes=10):
    """EfficientNet-B0: baseline model"""
    return EfficientNet(width_mult=1.0, depth_mult=1.0, num_classes=num_classes)

def efficientnet_b1(num_classes=10):
    """EfficientNet-B1: slightly wider and deeper"""
    return EfficientNet(width_mult=1.0, depth_mult=1.1, num_classes=num_classes)

def efficientnet_b2(num_classes=10):
    """EfficientNet-B2"""
    return EfficientNet(width_mult=1.1, depth_mult=1.2, num_classes=num_classes)

# Training code follows similar pattern to previous modules
# Key differences:
# - Use RMSprop optimizer with decay 0.9
# - Learning rate warmup for first few epochs
# - Exponential moving average of weights
# - Strong data augmentation (RandAugment, Mixup)

# EXERCISES:
# 1. Implement compound scaling search to find optimal α, β, γ
# 2. Compare EfficientNet-B0 to ResNet-50 (similar accuracy, fewer params)
# 3. Visualize how compound scaling affects network architecture
# 4. Implement EfficientNetV2 improvements (Fused-MBConv)
# 5. Apply EfficientNet to different image sizes (96x96, 224x224, 384x384)
