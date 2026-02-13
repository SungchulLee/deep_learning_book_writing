"""
Residual Connection Variants
============================
Different types and variations of residual connections:
1. Pre-activation ResNet
2. Wide ResNet
3. ResNeXt (Aggregated Residual Transformations)
4. DenseNet connections
5. Squeeze-and-Excitation (SE) blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. PRE-ACTIVATION RESIDUAL BLOCK
# ============================================================================

class PreActivationBlock(nn.Module):
    """
    Pre-activation Residual Block (Identity Mappings in Deep Residual Networks)
    
    Differences from original ResNet:
    - BN and ReLU come BEFORE convolution (not after)
    - Cleaner gradient flow through the skip connection
    - Better performance, especially for very deep networks
    
    Original: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    Pre-act:  BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Add
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActivationBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        
        out += shortcut
        return out


# ============================================================================
# 2. WIDE RESIDUAL BLOCK
# ============================================================================

class WideResidualBlock(nn.Module):
    """
    Wide Residual Block (Wide Residual Networks)
    
    Key idea: Increase width (channels) instead of depth
    - Uses a width multiplier (k) to make layers wider
    - Can be more efficient than very deep networks
    - Better parallelization on GPUs
    
    Example: k=10 means 10x more channels than standard ResNet
    """
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3, width_factor=1):
        super(WideResidualBlock, self).__init__()
        
        # Apply width factor
        wide_channels = int(out_channels * width_factor)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, wide_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(wide_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.conv2 = nn.Conv2d(wide_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out += self.shortcut(x)
        
        return out


# ============================================================================
# 3. RESNEXT BLOCK (AGGREGATED RESIDUAL TRANSFORMATIONS)
# ============================================================================

class ResNeXtBlock(nn.Module):
    """
    ResNeXt Block (Aggregated Residual Transformations for Deep Neural Networks)
    
    Key idea: Split-transform-merge strategy with cardinality
    - Instead of single transformation, use multiple parallel paths
    - Cardinality (C) = number of parallel transformations
    - More effective than increasing depth or width
    
    Architecture:
    - Split input into C groups
    - Apply same transformation to each group
    - Aggregate results
    """
    expansion = 2
    
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4):
        super(ResNeXtBlock, self).__init__()
        
        # Calculate group width
        width = int(out_channels * (base_width / 64.)) * cardinality
        
        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        # 3x3 grouped conv (the key innovation)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


# ============================================================================
# 4. DENSE CONNECTION (INSPIRED BY DENSENET)
# ============================================================================

class DenseBlock(nn.Module):
    """
    Dense Connection Block (inspired by DenseNet)
    
    Key idea: Connect each layer to ALL previous layers
    - Not a pure residual connection (concatenation instead of addition)
    - Better feature reuse
    - Reduces number of parameters
    
    Each layer receives feature maps from all preceding layers
    """
    
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_layers):
            self.layers.append(self._make_layer(current_channels, growth_rate))
            current_channels += growth_rate
    
    def _make_layer(self, in_channels, growth_rate):
        """Single dense layer"""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            # Concatenate all previous features
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        # Return concatenated features
        return torch.cat(features, dim=1)


# ============================================================================
# 5. SQUEEZE-AND-EXCITATION (SE) BLOCK
# ============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Key idea: Channel-wise attention mechanism
    - Squeeze: Global average pooling to get channel statistics
    - Excitation: Learn channel-wise weights
    - Recalibrate features by channel importance
    
    Can be added to any residual block for better performance
    """
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        reduced_channels = max(channels // reduction, 1)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        
        # Squeeze: Global pooling
        squeeze = self.squeeze(x).view(batch, channels)
        
        # Excitation: Channel attention
        excitation = self.excitation(squeeze).view(batch, channels, 1, 1)
        
        # Recalibrate
        return x * excitation.expand_as(x)


class SEResidualBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation
    Combines residual connections with channel attention
    """
    
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE block
        out = self.se(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


# ============================================================================
# TESTING AND COMPARISON
# ============================================================================

def test_variants():
    """
    Test all residual connection variants
    """
    print("=" * 80)
    print("Residual Connection Variants Comparison")
    print("=" * 80)
    
    batch_size = 2
    x = torch.randn(batch_size, 64, 32, 32)
    
    variants = {
        "Pre-activation": PreActivationBlock(64, 64),
        "Wide ResNet (k=2)": WideResidualBlock(64, 64, width_factor=2),
        "ResNeXt (C=32)": ResNeXtBlock(64, 64, cardinality=32),
        "SE-ResNet": SEResidualBlock(64, 64),
        "Dense Block": DenseBlock(64, growth_rate=32, num_layers=4)
    }
    
    print(f"\nInput shape: {x.shape}")
    print("-" * 80)
    
    for name, block in variants.items():
        with torch.no_grad():
            output = block(x)
        
        params = sum(p.numel() for p in block.parameters())
        print(f"{name:20} -> Output: {str(output.shape):25} Parameters: {params:>8,}")
    
    print("=" * 80)


def compare_performance_characteristics():
    """
    Compare key characteristics of different variants
    """
    print("\n" + "=" * 80)
    print("Performance Characteristics Comparison")
    print("=" * 80)
    
    characteristics = {
        "Original ResNet": {
            "Gradient Flow": "Good",
            "Parameter Efficiency": "Medium",
            "Training Speed": "Fast",
            "Best For": "General purpose"
        },
        "Pre-activation": {
            "Gradient Flow": "Excellent",
            "Parameter Efficiency": "Medium",
            "Training Speed": "Fast",
            "Best For": "Very deep networks (1000+ layers)"
        },
        "Wide ResNet": {
            "Gradient Flow": "Good",
            "Parameter Efficiency": "Low",
            "Training Speed": "Medium",
            "Best For": "When compute > memory constraint"
        },
        "ResNeXt": {
            "Gradient Flow": "Good",
            "Parameter Efficiency": "High",
            "Training Speed": "Fast",
            "Best For": "Better accuracy with similar complexity"
        },
        "SE-ResNet": {
            "Gradient Flow": "Good",
            "Parameter Efficiency": "High",
            "Training Speed": "Medium",
            "Best For": "Fine-grained classification"
        },
        "DenseNet": {
            "Gradient Flow": "Excellent",
            "Parameter Efficiency": "Very High",
            "Training Speed": "Slow",
            "Best For": "Small datasets, medical imaging"
        }
    }
    
    # Print table
    headers = ["Variant", "Gradient Flow", "Param Efficiency", "Speed", "Best For"]
    print(f"\n{headers[0]:20} {headers[1]:15} {headers[2]:17} {headers[3]:10} {headers[4]}")
    print("-" * 80)
    
    for variant, chars in characteristics.items():
        print(f"{variant:20} {chars['Gradient Flow']:15} {chars['Parameter Efficiency']:17} "
              f"{chars['Training Speed']:10} {chars['Best For']}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RESIDUAL CONNECTION VARIANTS")
    print("=" * 80)
    
    print("\nThis module covers:")
    print("1. Pre-activation ResNet (better gradient flow)")
    print("2. Wide ResNet (wider instead of deeper)")
    print("3. ResNeXt (aggregated transformations)")
    print("4. DenseNet connections (concatenation vs addition)")
    print("5. SE blocks (channel-wise attention)")
    
    # Test implementations
    test_variants()
    
    # Compare characteristics
    compare_performance_characteristics()
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("1. Pre-activation: Best for very deep networks (cleaner gradients)")
    print("2. Wide ResNet: Trade depth for width (better GPU utilization)")
    print("3. ResNeXt: Cardinality as a new dimension (parallel paths)")
    print("4. DenseNet: Maximum feature reuse (concatenate all previous layers)")
    print("5. SE blocks: Channel attention (learn feature importance)")
    print("=" * 80 + "\n")
