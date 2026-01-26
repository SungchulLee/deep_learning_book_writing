# DenseNet: Dense Connectivity

## Overview

DenseNet connects each layer to every subsequent layer in a feed-forward fashion, enabling **maximum information flow** and **feature reuse**. This dense connectivity pattern leads to substantially fewer parameters than ResNet while achieving competitive accuracy.

!!! info "Key Paper"
    Huang et al., 2017 - "Densely Connected Convolutional Networks" ([arXiv:1608.06993](https://arxiv.org/abs/1608.06993))

## Learning Objectives

1. Understand dense connectivity and feature concatenation
2. Implement DenseNet with growth rate and transition layers
3. Analyze feature reuse patterns
4. Compare DenseNet vs ResNet efficiency

## Dense Connectivity

### Key Innovation

Instead of adding features (ResNet), DenseNet **concatenates** all preceding features:

```
ResNet Block:                    DenseNet Block:

x₀ → [Conv] → x₁                 x₀ → [Conv] → x₁
      ↓                                ↓
x₁ + x₀ → [Conv] → x₂            [x₀, x₁] → [Conv] → x₂
      ↓                                ↓
x₂ + x₁ → [Conv] → x₃            [x₀, x₁, x₂] → [Conv] → x₃

(Addition)                       (Concatenation)
```

### Benefits

1. **Gradient flow**: Direct connections from loss to all layers
2. **Feature reuse**: Earlier features accessible to later layers
3. **Fewer parameters**: No need to relearn redundant features
4. **Implicit deep supervision**: Loss has direct paths to early layers

### Mathematical Formulation

For layer $l$ receiving inputs from all preceding layers:

$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

where $[...]$ denotes concatenation and $H_l$ is BN-ReLU-Conv.

## Architecture Components

### Growth Rate (k)

Each layer produces $k$ feature maps. After $l$ layers:

$$\text{channels} = k_0 + k \times l$$

where $k_0$ is initial channel count.

**Typical values**: k = 12, 24, 32

```python
# With k=32 and 6 layers:
# Input: 64 channels
# After layer 1: 64 + 32 = 96
# After layer 2: 96 + 32 = 128
# After layer 6: 64 + 6×32 = 256 channels
```

### Dense Block

```python
class DenseLayer(nn.Module):
    """Single dense layer: BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3"""
    
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        
        # Bottleneck: reduce channels before 3×3
        inter_channels = bn_size * growth_rate
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, growth_rate, 3, padding=1, bias=False)
        )
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = self.layers(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    """Dense block with n layers."""
    
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            ))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
```

### Transition Layer

Between dense blocks, reduce spatial dimensions and channels:

```python
class TransitionLayer(nn.Module):
    """Transition: BN → ReLU → Conv1×1 → AvgPool"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
    
    def forward(self, x):
        return self.trans(x)
```

### Compression Factor (θ)

Reduce channels at transitions:

$$\text{out\_channels} = \lfloor \theta \times \text{in\_channels} \rfloor$$

**Typical value**: θ = 0.5 (halve channels)

## Full DenseNet Implementation

```python
class DenseNet(nn.Module):
    """
    DenseNet architecture.
    
    Args:
        growth_rate: Channels added per layer (k)
        block_config: Layers in each dense block
        compression: Channel reduction at transitions (θ)
        num_classes: Output classes
        drop_rate: Dropout rate in dense layers
    """
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 compression=0.5, num_classes=1000, drop_rate=0.0):
        super().__init__()
        
        # Initial convolution
        num_init = 2 * growth_rate  # 64 for k=32
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Dense blocks with transitions
        num_features = num_init
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(num_layers, num_features, growth_rate, 
                             bn_size=4, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            # Transition (except after last block)
            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = TransitionLayer(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Factory functions
def densenet121(num_classes=1000):
    """DenseNet-121: k=32, blocks=[6, 12, 24, 16]"""
    return DenseNet(32, (6, 12, 24, 16), 0.5, num_classes)

def densenet169(num_classes=1000):
    """DenseNet-169: k=32, blocks=[6, 12, 32, 32]"""
    return DenseNet(32, (6, 12, 32, 32), 0.5, num_classes)

def densenet201(num_classes=1000):
    """DenseNet-201: k=32, blocks=[6, 12, 48, 32]"""
    return DenseNet(32, (6, 12, 48, 32), 0.5, num_classes)

def densenet_cifar(num_classes=10):
    """Small DenseNet for CIFAR: k=12"""
    return DenseNet(12, (6, 12, 24, 16), 0.5, num_classes)
```

## Connectivity Analysis

### Number of Connections

In a dense block with L layers:

$$\text{connections} = \frac{L(L+1)}{2}$$

For L=12: 78 connections (vs 12 in plain network)

### Feature Reuse Visualization

```
Layer outputs used by subsequent layers:

        Layer →  1   2   3   4   5   6
        Uses ↓
          1      -   ✓   ✓   ✓   ✓   ✓
          2          -   ✓   ✓   ✓   ✓
          3              -   ✓   ✓   ✓
          4                  -   ✓   ✓
          5                      -   ✓
          6                          -

Each layer receives ALL preceding features!
```

### Implicit Deep Supervision

Loss gradient reaches layer $l$ through:
- Direct path: $\frac{\partial L}{\partial x_l}$
- Paths through all subsequent layers

## Memory Considerations

### Challenge: Quadratic Memory

Naive implementation stores all intermediate features:

```
Memory for block with L layers, growth rate k:
Layer 1: k₀ features
Layer 2: k₀ + k features  
Layer 3: k₀ + 2k features
...
Layer L: k₀ + (L-1)k features

Total stored: L×k₀ + k×(0+1+...+(L-1)) = L×k₀ + k×L(L-1)/2
```

### Solution: Memory-Efficient Implementation

Recompute concat during backward pass:

```python
class EfficientDenseBlock(nn.Module):
    """Memory-efficient dense block with checkpointing."""
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Checkpoint: recompute during backward
            new_features = torch.utils.checkpoint.checkpoint(
                layer, torch.cat(features, 1),
                use_reentrant=False
            )
            features.append(new_features)
        return torch.cat(features, 1)
```

## Comparison with ResNet

| Aspect | ResNet | DenseNet |
|--------|--------|----------|
| Connection | Skip (addition) | Dense (concat) |
| Information | Summation | Preserved |
| Parameters | More | Fewer |
| Memory (train) | Less | More |
| Feature reuse | Implicit | Explicit |

### Parameter Efficiency

| Model | Params | Top-1 | FLOPs |
|-------|--------|-------|-------|
| ResNet-50 | 25.6M | 76.1% | 4.1B |
| ResNet-101 | 44.5M | 77.4% | 7.9B |
| DenseNet-121 | 8.0M | 75.0% | 2.9B |
| DenseNet-169 | 14.1M | 76.2% | 3.4B |
| DenseNet-201 | 20.0M | 77.4% | 4.3B |

**DenseNet-201 matches ResNet-101 with 55% fewer parameters!**

## Exercises

1. Visualize how many connections exist in a 6-layer dense block
2. Implement DenseNet-BC (with bottleneck and compression)
3. Compare memory usage vs ResNet during training
4. Ablation study: vary growth rate k (8, 12, 24, 32)

## Key Takeaways

1. **Dense connectivity** maximizes information flow
2. **Feature concatenation** enables explicit feature reuse
3. **Growth rate** controls capacity (small k works well)
4. **Compression** reduces parameters between blocks
5. **Memory-efficient training** is important for deep DenseNets

## References

1. Huang et al. (2017). Densely Connected Convolutional Networks. CVPR.
2. Pleiss et al. (2017). Memory-Efficient Implementation of DenseNets.

---

**Previous**: [MobileNet](mobilenet.md) | **Next**: [ConvNeXt](convnext.md)
