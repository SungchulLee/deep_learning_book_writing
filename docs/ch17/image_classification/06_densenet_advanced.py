"""
Module 33.6: DenseNet - Dense Connectivity (ADVANCED)

DenseNet connects each layer to every other layer in a feed-forward fashion,
enabling maximum information flow and feature reuse.

Key Concepts:
1. Dense connectivity pattern
2. Feature concatenation instead of addition
3. Transition layers for dimensionality reduction
4. Growth rate hyperparameter

Paper: Huang et al., 2017 - "Densely Connected Convolutional Networks"
"""

import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    """
    Single dense layer: BN -> ReLU -> Conv(1x1) -> BN -> ReLU -> Conv(3x3)
    
    Bottleneck design: 1x1 conv reduces channels before expensive 3x3 conv
    Output is concatenated with input along channel dimension
    """
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        
        # Bottleneck layer
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        
        # 3x3 convolution
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))
        
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, 
                                                training=self.training)
        
        # Concatenate input and output
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    """
    Dense block with multiple dense layers.
    
    Each layer receives feature maps from all preceding layers.
    If block has L layers with growth rate k:
    - Layer l has k₀ + k×(l-1) input channels
    - Each layer produces k feature maps
    - Block output has k₀ + k×L channels
    """
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate, bn_size, drop_rate
            ))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class TransitionLayer(nn.Module):
    """
    Transition layer between dense blocks.
    
    Reduces spatial dimensions (2x2 avg pooling) and 
    optionally reduces channels (compression factor)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    """
    DenseNet architecture.
    
    Args:
        growth_rate: Number of filters each layer adds (k)
        block_config: Number of layers in each dense block
        compression: Reduction factor in transition layers (0 < θ ≤ 1)
        num_classes: Number of output classes
    """
    def __init__(self, growth_rate=12, block_config=(6,12,24,16), 
                 compression=0.5, num_classes=10, drop_rate=0.0):
        super().__init__()
        
        # Initial convolution
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(
                num_layers, num_features, growth_rate, 
                bn_size=4, drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            # Transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        
        # Final batch norm
        self.features.add_module('bn_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def densenet121(num_classes=10):
    """DenseNet-121 (k=32, blocks=[6,12,24,16])"""
    return DenseNet(32, (6,12,24,16), 0.5, num_classes)

def densenet169(num_classes=10):
    """DenseNet-169 (k=32, blocks=[6,12,32,32])"""
    return DenseNet(32, (6,12,32,32), 0.5, num_classes)

def densenet_small(num_classes=10):
    """Small DenseNet for CIFAR-10 (k=12, blocks=[6,12,24,16])"""
    return DenseNet(12, (6,12,24,16), 0.5, num_classes)

# EXERCISES:
# 1. Visualize dense connectivity: how many connections in a block with L layers?
# 2. Compare memory usage during training vs ResNet
# 3. Ablation study: vary growth rate k (8, 12, 24, 32)
# 4. Measure feature reuse: how much do early features contribute to final layers?
# 5. Implement DenseNet-BC variant with compression and bottleneck
