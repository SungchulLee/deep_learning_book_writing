# ResNet: Residual Networks

## Overview

Residual Networks (ResNet) introduced **skip connections** that enable training of very deep networks (100+ layers) by solving the degradation problem where deeper networks paradoxically performed worse than shallower ones. ResNet won the ImageNet 2015 competition and remains one of the most influential architectures in deep learning.

!!! info "Key Paper"
    He et al., 2015 - "Deep Residual Learning for Image Recognition" ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))

## Learning Objectives

After completing this section, you will be able to:

1. Explain why skip connections solve the degradation problem
2. Implement BasicBlock and BottleneckBlock architectures
3. Build ResNet-18, ResNet-34, ResNet-50, and deeper variants
4. Train ResNet on CIFAR-10 achieving >93% accuracy
5. Analyze gradient flow through residual connections

## The Degradation Problem

### Motivation

Before ResNet, a puzzling observation plagued deep learning: adding more layers to a network initially improved performance, but beyond a certain depth, accuracy degraded—not from overfitting (training error also increased), but from optimization difficulty.

```
Network Depth vs Training Error (Pre-ResNet):

Depth     Training Error
------    --------------
  20      ████████░░░░ 8.2%
  32      ██████░░░░░░ 6.1%
  44      █████░░░░░░░ 5.3%
  56      ██████░░░░░░ 6.0%  ← Degradation!
  110     ████████░░░░ 8.7%  ← Getting worse
```

### The Identity Mapping Hypothesis

If deeper networks can represent everything shallower networks can (plus more), why do they perform worse? The answer lies in optimization: it's difficult for stacked nonlinear layers to learn identity mappings.

**Key insight**: If added layers could easily learn identity functions, deeper networks would never be worse than shallower ones.

## Residual Learning

### Core Concept

Instead of learning the desired mapping $H(x)$ directly, ResNet learns the **residual** $F(x) = H(x) - x$:

$$H(x) = F(x) + x$$

The output is the sum of:
- **Residual function** $F(x)$: Learned transformations
- **Identity shortcut** $x$: Input passed through unchanged

### Why This Works

1. **Easy identity**: If optimal $H(x) = x$, push $F(x) \to 0$ (easier than learning identity through nonlinear layers)
2. **Gradient highway**: Gradients flow directly through skip connection, bypassing potentially vanishing paths
3. **Ensemble view**: ResNet can be viewed as ensemble of paths of different lengths

### Mathematical Formulation

For a residual block with two layers:

$$y = F(x, \{W_i\}) + x$$

where $F(x, \{W_i\}) = W_2 \cdot \sigma(W_1 \cdot x)$ and $\sigma$ is ReLU.

**Gradient flow**:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$

The "+1" term ensures gradients never completely vanish, regardless of $\frac{\partial F}{\partial x}$.

## Architecture Components

### Basic Block (ResNet-18/34)

Used in shallower ResNets. Two 3×3 convolutions with a skip connection:

```
Input (x)
    │
    ├──────────────────────┐
    │                      │
    ▼                      │
┌─────────────┐            │
│ Conv 3×3    │            │
│ BatchNorm   │            │
│ ReLU        │            │
└─────────────┘            │
    │                      │
    ▼                      │
┌─────────────┐            │
│ Conv 3×3    │            │
│ BatchNorm   │            │
└─────────────┘            │
    │                      │
    ▼                      │
   (+) ◄───────────────────┘
    │
    ▼
┌─────────────┐
│ ReLU        │
└─────────────┘
    │
    ▼
Output
```

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18 and ResNet-34.
    
    Architecture:
        x → [Conv 3×3] → [BN] → [ReLU] → [Conv 3×3] → [BN] → (+) → [ReLU]
        |__________________________________________________|
                           (identity or projection)
    """
    expansion = 1  # Output channels = input channels × expansion
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first conv (1 or 2 for downsampling)
            downsample: Projection shortcut when dimensions change
        """
        super(BasicBlock, self).__init__()
        
        # First convolution (may downsample if stride=2)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution (always stride=1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # Main path: conv → bn → relu → conv → bn
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Project identity if dimensions changed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Skip connection: add identity
        out += identity
        out = self.relu(out)
        
        return out
```

### Bottleneck Block (ResNet-50/101/152)

More parameter-efficient for deeper networks. Uses 1×1 convolutions to reduce then expand dimensions:

```
Input (x)
    │
    ├──────────────────────┐
    │                      │
    ▼                      │
┌─────────────┐            │
│ Conv 1×1    │ ← Reduce   │
│ BatchNorm   │   channels │
│ ReLU        │            │
└─────────────┘            │
    │                      │
    ▼                      │
┌─────────────┐            │
│ Conv 3×3    │ ← Main     │
│ BatchNorm   │   computation
│ ReLU        │            │
└─────────────┘            │
    │                      │
    ▼                      │
┌─────────────┐            │
│ Conv 1×1    │ ← Expand   │
│ BatchNorm   │   channels │
└─────────────┘            │
    │                      │
    ▼                      │
   (+) ◄───────────────────┘
    │
    ▼
┌─────────────┐
│ ReLU        │
└─────────────┘
    │
    ▼
Output
```

```python
class BottleneckBlock(nn.Module):
    """
    Bottleneck Block for ResNet-50, ResNet-101, ResNet-152.
    
    Uses 1×1 convolutions for dimensionality reduction and expansion:
    - 1×1 conv: Reduce channels (e.g., 256 → 64)
    - 3×3 conv: Main computation on reduced dimensions
    - 1×1 conv: Expand channels (e.g., 64 → 256)
    
    More efficient than BasicBlock for deep networks.
    """
    expansion = 4  # Output channels = bottleneck_channels × 4
    
    def __init__(self, in_channels, bottleneck_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        out_channels = bottleneck_channels * self.expansion
        
        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3×3 conv (may downsample)
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 1×1 expand
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))  # Reduce
        out = self.relu(self.bn2(self.conv2(out)))  # 3×3 conv
        out = self.bn3(self.conv3(out))  # Expand
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
```

### Projection Shortcuts

When input and output dimensions differ (stride > 1 or channel mismatch), use a **projection shortcut**:

```python
# 1×1 convolution to match dimensions
downsample = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
    nn.BatchNorm2d(out_channels)
)
```

Two shortcut options from the paper:
- **Option A**: Zero-padding for extra channels (no additional parameters)
- **Option B**: Projection with 1×1 conv (used in practice, slightly better)

## Full ResNet Architecture

### Architecture Overview

```
Input Image (3×224×224)
         │
         ▼
┌──────────────────┐
│ Conv 7×7, stride 2│  → 64×112×112
│ BatchNorm, ReLU   │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ MaxPool 3×3, s=2 │  → 64×56×56
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Layer 1: N blocks│  → 64×56×56 (or 256 for bottleneck)
│ (no downsampling)│
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Layer 2: N blocks│  → 128×28×28 (or 512 for bottleneck)
│ (stride=2 first) │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Layer 3: N blocks│  → 256×14×14 (or 1024 for bottleneck)
│ (stride=2 first) │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Layer 4: N blocks│  → 512×7×7 (or 2048 for bottleneck)
│ (stride=2 first) │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Global AvgPool   │  → 512×1×1 (or 2048)
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Fully Connected  │  → num_classes
└──────────────────┘
```

### ResNet Configurations

| Model | Block Type | Layers per Stage | Total Layers | Parameters |
|-------|------------|------------------|--------------|------------|
| ResNet-18 | BasicBlock | [2, 2, 2, 2] | 18 | 11.7M |
| ResNet-34 | BasicBlock | [3, 4, 6, 3] | 34 | 21.8M |
| ResNet-50 | Bottleneck | [3, 4, 6, 3] | 50 | 25.6M |
| ResNet-101 | Bottleneck | [3, 4, 23, 3] | 101 | 44.5M |
| ResNet-152 | Bottleneck | [3, 8, 36, 3] | 152 | 60.2M |

### Complete Implementation

```python
class ResNet(nn.Module):
    """
    ResNet architecture for image classification.
    
    Args:
        block: BasicBlock or BottleneckBlock
        layers: Number of blocks in each stage [stage1, stage2, stage3, stage4]
        num_classes: Number of output classes
        zero_init_residual: Zero-initialize last BN in each block
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution (stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(self, block, channels, num_blocks, stride):
        """Create a stage with multiple residual blocks."""
        downsample = None
        out_channels = channels * block.expansion
        
        # Projection shortcut for first block if dimensions change
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        # First block may downsample
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize last BN in each residual block
        # This improves performance by ~0.2-0.3%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Factory functions
def resnet18(num_classes=1000, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)

def resnet34(num_classes=1000, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)

def resnet50(num_classes=1000, **kwargs):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes, **kwargs)

def resnet101(num_classes=1000, **kwargs):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes, **kwargs)

def resnet152(num_classes=1000, **kwargs):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes, **kwargs)
```

## Training on CIFAR-10

### Adapted Architecture for Small Images

CIFAR-10 images are 32×32, too small for the standard ImageNet architecture. Modifications:

```python
class ResNetCIFAR(nn.Module):
    """ResNet adapted for CIFAR-10 (32×32 images)."""
    
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 16
        
        # Smaller stem: 3×3 conv instead of 7×7, no max pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Stages with fewer channels
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
    
    def _make_layer(self, block, channels, num_blocks, stride):
        # Same as before
        ...
```

### Training Configuration

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_resnet_cifar10():
    # Hyperparameters
    config = {
        'batch_size': 128,
        'epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    }
    
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
    ])
    
    # Model
    model = resnet18(num_classes=10).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
```

### Expected Results

| Model | CIFAR-10 Accuracy | Training Time (GPU) |
|-------|-------------------|---------------------|
| ResNet-18 | 93-95% | ~2 hours |
| ResNet-34 | 94-95% | ~3 hours |
| ResNet-50 | 94-96% | ~4 hours |

## Gradient Flow Analysis

### Visualizing Skip Connection Benefits

```python
def analyze_gradient_flow(model, input_batch, target):
    """Analyze gradient magnitudes through the network."""
    model.train()
    
    # Forward pass with gradient tracking
    output = model(input_batch)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    
    # Collect gradient statistics per layer
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats[name] = {
                'mean': param.grad.abs().mean().item(),
                'std': param.grad.std().item(),
                'max': param.grad.abs().max().item()
            }
    
    return grad_stats
```

### Comparison: With vs Without Skip Connections

```
Gradient Magnitude by Layer Depth:

With Skip Connections:
Layer 1:  ████████████████ 0.032
Layer 10: ███████████████░ 0.028
Layer 20: ██████████████░░ 0.025
Layer 30: █████████████░░░ 0.022
Layer 40: ████████████░░░░ 0.019  ← Gradients preserved!

Without Skip Connections:
Layer 1:  ████████████████ 0.032
Layer 10: ██████████░░░░░░ 0.015
Layer 20: ████░░░░░░░░░░░░ 0.006
Layer 30: █░░░░░░░░░░░░░░░ 0.001
Layer 40: ░░░░░░░░░░░░░░░░ 0.0001 ← Vanishing!
```

## Variants and Improvements

### Pre-Activation ResNet (ResNet-v2)

He et al. proposed moving BN and ReLU before convolution:

```python
# Original (post-activation)
out = relu(bn(conv(x)))

# Pre-activation
out = conv(relu(bn(x)))
```

**Benefits**: Improved gradient flow, slightly better accuracy

### Wide ResNet

Zagoruyko & Komodakis showed wider networks can outperform deeper ones:

```python
# ResNet-28-10: 28 layers, 10× wider than standard
# Better accuracy than ResNet-1001 with much faster training
```

### ResNeXt

Aggregated residual transformations (grouped convolutions):

```python
# Cardinality: number of parallel paths
self.conv2 = nn.Conv2d(
    channels, channels, kernel_size=3, 
    stride=stride, padding=1, groups=32,  # 32 parallel groups
    bias=False
)
```

## Exercises

### Beginner

1. **Basic Training**: Train ResNet-18 on CIFAR-10 to achieve >90% accuracy
2. **Architecture Exploration**: Compare ResNet-18 vs ResNet-34 training curves
3. **Skip Connection Analysis**: Remove skip connections and observe training behavior

### Intermediate

4. **Pre-activation ResNet**: Implement and compare with post-activation
5. **Gradient Visualization**: Plot gradient magnitudes across layers during training
6. **Learning Rate Study**: Compare step decay vs cosine annealing

### Advanced

7. **Custom Depth**: Design ResNet with 20, 44, 56 layers for CIFAR-10
8. **Wide ResNet**: Implement and evaluate width vs depth trade-off
9. **ResNeXt**: Implement grouped convolutions and compare with standard ResNet

## Key Takeaways

1. **Skip connections** enable training of very deep networks by providing gradient highways
2. **Residual learning** is easier than direct mapping for identity functions
3. **Bottleneck design** improves parameter efficiency for deep networks
4. **Zero-initialization** of final batch norm improves performance slightly
5. **Pre-activation** ordering can further improve gradient flow

## References

1. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. He, K., et al. (2016). Identity Mappings in Deep Residual Networks. ECCV.
3. Zagoruyko, S., & Komodakis, N. (2016). Wide Residual Networks.
4. Xie, S., et al. (2017). Aggregated Residual Transformations (ResNeXt). CVPR.

---

**Next Section**: [VGG Networks](vgg.md)
