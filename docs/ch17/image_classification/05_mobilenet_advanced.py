"""
Module 33.5: MobileNet - Efficient Mobile Vision (ADVANCED)

MobileNet uses depthwise separable convolutions to build lightweight deep neural networks
for mobile and embedded vision applications.

Key Concepts:
1. Depthwise separable convolutions
2. Width and resolution multipliers
3. Inverted residuals (MobileNetV2)
4. Linear bottlenecks

Paper: Howard et al., 2017 - "MobileNets: Efficient CNNs for Mobile Vision"
"""

import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution = Depthwise + Pointwise
    
    Standard conv: K×K×C_in×C_out parameters
    Depthwise sep: K×K×C_in + C_in×C_out parameters
    
    Reduction: (K² + 1/C_out) ≈ 8-9x fewer parameters for 3x3
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        # Depthwise: applies single filter per input channel
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class InvertedResidual(nn.Module):
    """
    Inverted Residual block (MobileNetV2).
    
    Standard residual: wide -> narrow -> wide
    Inverted residual: narrow -> wide -> narrow
    
    Architecture:
    1. Expansion: 1x1 conv to expand channels (×6)
    2. Depthwise: 3x3 depthwise conv
    3. Projection: 1x1 conv to project back (linear)
    4. Skip connection if stride=1
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Projection (linear bottleneck - no ReLU!)
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2(nn.Module):
    """
    MobileNetV2 with inverted residuals and linear bottlenecks.
    
    Configuration: [expand_ratio, channels, num_blocks, stride]
    """
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        
        # Configuration
        config = [
            # expand, channels, blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Stem
        in_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks
        blocks = []
        for expand, channels, num_blocks, stride in config:
            out_channels = int(channels * width_mult)
            for i in range(num_blocks):
                blocks.append(InvertedResidual(
                    in_channels, out_channels,
                    stride if i == 0 else 1, expand
                ))
                in_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(final_channels, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

def mobilenet_v2(num_classes=10, width_mult=1.0):
    """
    MobileNetV2 with width multiplier.
    
    width_mult=1.0: 3.5M parameters
    width_mult=0.75: ~2.6M parameters  
    width_mult=0.5: ~1.9M parameters
    width_mult=0.35: ~1.66M parameters
    """
    return MobileNetV2(num_classes, width_mult)

# EXERCISES:
# 1. Compare standard conv vs depthwise separable conv on FLOPs and parameters
# 2. Train MobileNetV2 with different width multipliers (0.5, 0.75, 1.0, 1.4)
# 3. Measure actual inference time on CPU vs GPU
# 4. Implement MobileNetV3 with h-swish and squeeze-excitation
# 5. Apply quantization-aware training for deployment
