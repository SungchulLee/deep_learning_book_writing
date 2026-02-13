#!/usr/bin/env python3
"""
================================================================================
DenseNet - Densely Connected Convolutional Networks
================================================================================

Paper: "Densely Connected Convolutional Networks" (CVPR 2017 Best Paper)
Authors: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
Link: https://arxiv.org/abs/1608.06993

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
DenseNet introduced an elegant connection pattern where each layer receives
feature maps from ALL preceding layers. This led to:

- **CVPR 2017 Best Paper Award**
- **State-of-the-art** performance with fewer parameters than ResNet
- **Improved gradient flow** throughout the network
- **Feature reuse** reduces redundancy and parameters

================================================================================
KEY INSIGHT: DENSE CONNECTIONS
================================================================================

Each layer l receives feature maps from ALL preceding layers:
    x_l = H_l([x_0, x_1, ..., x_{l-1}])
    
Where [·] denotes concatenation along the channel dimension.

Comparison:
- Traditional: L1 → L2 → L3 → L4
- ResNet:      L1 ──→ L2 ──→ L3 ──→ L4  (with additive skip connections)
- DenseNet:    L1 ──→ L2 ──→ L3 ──→ L4  (ALL layers connected via concatenation)

================================================================================
BENEFITS OF DENSE CONNECTIONS
================================================================================

1. **Strong Gradient Flow**: Direct paths from loss to all layers
2. **Feature Reuse**: Later layers access low-level features directly
3. **Parameter Efficiency**: 3× fewer parameters than ResNet for similar accuracy
4. **Implicit Deep Supervision**: Short paths for gradient flow
5. **Regularization Effect**: Works well on small datasets

================================================================================
ARCHITECTURE CONFIGURATIONS
================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│ Model        │ Layers per Block    │ Growth Rate │ Parameters │             │
├──────────────────────────────────────────────────────────────────────────────┤
│ DenseNet-121 │ [6, 12, 24, 16]     │     32      │    8.0M    │             │
│ DenseNet-169 │ [6, 12, 32, 32]     │     32      │   14.1M    │             │
│ DenseNet-201 │ [6, 12, 48, 32]     │     32      │   20.0M    │             │
│ DenseNet-264 │ [6, 12, 64, 48]     │     32      │   33.3M    │             │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: Residual Connections (comparison with skip connections)
- Ch04: Image Classification (DenseNet architecture)  
- Ch08: Transfer Learning (efficient feature extraction)

Related: resnet.py, efficientnet.py
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DenseLayer(nn.Module):
    """
    Single Dense Layer within a Dense Block
    
    Implements BN-ReLU-Conv pattern with bottleneck:
    Input → BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3 → Output
    
    Args:
        in_channels: Number of input channels
        growth_rate: Number of output feature maps (k)
        bn_size: Bottleneck multiplier for 1×1 conv. Default: 4
        drop_rate: Dropout rate. Default: 0
    """
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0
    ):
        super(DenseLayer, self).__init__()
        
        # Bottleneck: reduce to bn_size × growth_rate channels
        bottleneck_channels = bn_size * growth_rate
        
        # First BN-ReLU-Conv (1×1): Dimensionality Reduction
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels,
            kernel_size=1, stride=1, bias=False
        )
        
        # Second BN-ReLU-Conv (3×3): Feature Extraction
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.drop_rate = drop_rate
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass: concatenate inputs, apply bottleneck, produce k features"""
        if isinstance(inputs, torch.Tensor):
            prev_features = inputs
        else:
            prev_features = torch.cat(inputs, dim=1)
        
        # Bottleneck: reduce channels
        out = self.conv1(F.relu(self.bn1(prev_features)))
        
        # Produce new features
        out = self.conv2(F.relu(self.bn2(out)))
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        
        return out


class DenseBlock(nn.Module):
    """
    Dense Block: A sequence of densely connected layers
    
    Each layer receives features from ALL previous layers.
    Output channels = in_channels + num_layers × growth_rate
    
    Args:
        num_layers: Number of dense layers in this block
        in_channels: Number of input channels
        growth_rate: Number of feature maps each layer produces
        bn_size: Bottleneck multiplier. Default: 4
        drop_rate: Dropout rate. Default: 0
    """
    
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0
    ):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_channels = in_channels + i * growth_rate
            layer = DenseLayer(layer_input_channels, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: each layer receives ALL previous features (concatenated)"""
        features = [x]
        
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    """
    Transition Layer: Between dense blocks
    
    Compresses channels and downsamples spatial dimensions.
    Structure: BN → ReLU → Conv1×1 → AvgPool2×2
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (= θ × in_channels)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Transition, self).__init__()
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress channels and downsample"""
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    """
    DenseNet (Densely Connected Convolutional Network)
    
    Args:
        growth_rate: Number of filters to add per layer (k). Default: 32
        block_config: Number of layers in each dense block. Default: (6, 12, 24, 16)
        num_init_features: Number of filters in initial conv. Default: 64
        bn_size: Bottleneck multiplier. Default: 4
        drop_rate: Dropout rate. Default: 0
        num_classes: Number of output classes. Default: 1000
        compression: Channel compression in transitions (θ). Default: 0.5
    
    Example:
        >>> model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16))
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)
    """
    
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        compression: float = 0.5
    ):
        super(DenseNet, self).__init__()
        
        # ====================================================================
        # STEM: Initial Convolution
        # ====================================================================
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(
            3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False
        ))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # ====================================================================
        # DENSE BLOCKS + TRANSITIONS
        # ====================================================================
        num_features = num_init_features
        
        for i, num_layers in enumerate(block_config):
            # Dense Block
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Transition (except after last block)
            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = Transition(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features
        
        # Final BN
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DenseNet"""
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier"""
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def DenseNet121(num_classes: int = 1000, **kwargs) -> DenseNet:
    """DenseNet-121: ~8M parameters"""
    return DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes, **kwargs)

def DenseNet169(num_classes: int = 1000, **kwargs) -> DenseNet:
    """DenseNet-169: ~14M parameters"""
    return DenseNet(32, (6, 12, 32, 32), 64, num_classes=num_classes, **kwargs)

def DenseNet201(num_classes: int = 1000, **kwargs) -> DenseNet:
    """DenseNet-201: ~20M parameters"""
    return DenseNet(32, (6, 12, 48, 32), 64, num_classes=num_classes, **kwargs)


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
    print("DenseNet Family - Parameter Comparison")
    print("=" * 70)
    
    for name, fn in [('DenseNet-121', DenseNet121), ('DenseNet-169', DenseNet169), ('DenseNet-201', DenseNet201)]:
        model = fn(num_classes=1000)
        total, _ = count_parameters(model)
        print(f"{name}: {total:>12,} parameters")
    
    print("=" * 70)
    
    model = DenseNet121(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"Input: {x.shape}, Output: {logits.shape}")
    print("=" * 70)
