# VGG: Very Deep Convolutional Networks

## Overview

VGG Networks demonstrated that network depth is critical for achieving strong performance, using a simple and uniform architecture consisting of stacked 3×3 convolutional filters. Despite being computationally expensive (138M parameters for VGG-16), VGG's straightforward design made it highly influential and easy to understand.

!!! info "Key Paper"
    Simonyan & Zisserman, 2014 - "Very Deep Convolutional Networks for Large-Scale Image Recognition" ([arXiv:1409.1556](https://arxiv.org/abs/1409.1556))

## Learning Objectives

After completing this section, you will be able to:

1. Explain why small 3×3 filters are preferable to larger filters
2. Implement VGG-11, VGG-13, VGG-16, and VGG-19 architectures
3. Understand the importance of batch normalization for training VGG
4. Visualize learned filters and feature maps
5. Analyze VGG's computational cost vs accuracy trade-offs

## Design Philosophy

### Why 3×3 Convolutions?

VGG's key insight: **Stacking small 3×3 filters achieves the same receptive field as larger filters, but with fewer parameters and more nonlinearity.**

#### Receptive Field Equivalence

Two stacked 3×3 convolutions have the same receptive field as one 5×5 convolution:

```
Single 5×5 filter:          Two 3×3 filters:
                            
  X X X X X                   X X X ─┐
  X X X X X                   X X X  │→ 3×3
  X X X X X  ═══════         X X X ─┘
  X X X X X                       │
  X X X X X                       ▼
                              X X X ─┐
  Receptive                   X X X  │→ 3×3
  Field: 5×5                  X X X ─┘
                              
                              Effective RF: 5×5
```

Three 3×3 convolutions equal one 7×7:
- **7×7**: 49 parameters per input channel
- **3 × 3×3**: 27 parameters per input channel → **45% fewer parameters**

#### Additional Nonlinearity

Each 3×3 conv is followed by ReLU, so stacking provides more nonlinear transformations:

- **One 7×7**: 1 nonlinearity
- **Three 3×3**: 3 nonlinearities → **More representational power**

### Uniform Architecture

VGG uses a remarkably consistent design:

- **All convolutions**: 3×3 with stride 1, padding 1 (preserves spatial size)
- **All pooling**: 2×2 max pooling with stride 2 (halves spatial size)
- **Channel progression**: 64 → 128 → 256 → 512 → 512
- **Same-padding**: Maintains spatial dimensions within blocks

## Architecture Details

### VGG Configurations

The paper explored five configurations (A through E):

```
Configuration Table:

Layer       VGG-11(A)  VGG-13(B)  VGG-16(D)  VGG-19(E)
─────────────────────────────────────────────────────
Conv Block 1   64×1      64×2       64×2       64×2
MaxPool 2×2      ↓          ↓          ↓          ↓
─────────────────────────────────────────────────────
Conv Block 2  128×1     128×2      128×2      128×2
MaxPool 2×2      ↓          ↓          ↓          ↓
─────────────────────────────────────────────────────
Conv Block 3  256×2     256×2      256×3      256×4
MaxPool 2×2      ↓          ↓          ↓          ↓
─────────────────────────────────────────────────────
Conv Block 4  512×2     512×2      512×3      512×4
MaxPool 2×2      ↓          ↓          ↓          ↓
─────────────────────────────────────────────────────
Conv Block 5  512×2     512×2      512×3      512×4
MaxPool 2×2      ↓          ↓          ↓          ↓
─────────────────────────────────────────────────────
FC Layers   4096→4096→1000 (same for all)
─────────────────────────────────────────────────────
Weight Layers   11        13         16         19
Parameters     133M      133M       138M       144M
```

### Layer-by-Layer Breakdown (VGG-16)

```
Layer          Output Size    Parameters
──────────────────────────────────────────────────────
Input          224×224×3          -
──────────────────────────────────────────────────────
Conv3-64       224×224×64     3×3×3×64 = 1,728
Conv3-64       224×224×64     3×3×64×64 = 36,864
MaxPool        112×112×64         -
──────────────────────────────────────────────────────
Conv3-128      112×112×128    3×3×64×128 = 73,728
Conv3-128      112×112×128    3×3×128×128 = 147,456
MaxPool        56×56×128          -
──────────────────────────────────────────────────────
Conv3-256      56×56×256      3×3×128×256 = 294,912
Conv3-256      56×56×256      3×3×256×256 = 589,824
Conv3-256      56×56×256      3×3×256×256 = 589,824
MaxPool        28×28×256          -
──────────────────────────────────────────────────────
Conv3-512      28×28×512      3×3×256×512 = 1,179,648
Conv3-512      28×28×512      3×3×512×512 = 2,359,296
Conv3-512      28×28×512      3×3×512×512 = 2,359,296
MaxPool        14×14×512          -
──────────────────────────────────────────────────────
Conv3-512      14×14×512      3×3×512×512 = 2,359,296
Conv3-512      14×14×512      3×3×512×512 = 2,359,296
Conv3-512      14×14×512      3×3×512×512 = 2,359,296
MaxPool        7×7×512            -
──────────────────────────────────────────────────────
FC-4096        4096           7×7×512×4096 = 102,760,448
FC-4096        4096           4096×4096 = 16,777,216
FC-1000        1000           4096×1000 = 4,096,000
──────────────────────────────────────────────────────
Total Parameters:              ~138 Million
```

**Note**: Most parameters (>90%) are in the fully connected layers!

## Implementation

### VGG Architecture

```python
import torch
import torch.nn as nn

class VGG(nn.Module):
    """
    VGG network architecture.
    
    Design Principles:
    - All convolutions: 3×3 with stride 1, padding 1
    - All max pooling: 2×2 with stride 2
    - Channels double after pooling: 64 → 128 → 256 → 512 → 512
    - Three fully connected layers: 4096 → 4096 → num_classes
    """
    
    def __init__(self, features, num_classes=1000, init_weights=True, dropout=0.5):
        """
        Args:
            features: Sequential container of conv and pooling layers
            num_classes: Number of output classes
            init_weights: Whether to initialize weights
            dropout: Dropout probability for FC layers
        """
        super(VGG, self).__init__()
        
        self.features = features
        
        # Adaptive pooling handles variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
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
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x, return_all=False):
        """Extract features at various depths for visualization."""
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x.detach())
        
        if return_all:
            return features
        return x


def make_layers(cfg, batch_norm=False):
    """
    Create VGG feature layers from configuration.
    
    Args:
        cfg: List where integers are conv channels and 'M' is max pooling
        batch_norm: Whether to add batch normalization
    
    Returns:
        nn.Sequential of layers
    """
    layers = []
    in_channels = 3
    
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(*layers)


# Configuration dictionary
# Numbers = conv output channels, 'M' = max pooling
VGG_CONFIGS = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=1000, batch_norm=False):
    """VGG-11: 8 conv + 3 FC = 11 weight layers"""
    return VGG(make_layers(VGG_CONFIGS['A'], batch_norm), num_classes)

def vgg13(num_classes=1000, batch_norm=False):
    """VGG-13: 10 conv + 3 FC = 13 weight layers"""
    return VGG(make_layers(VGG_CONFIGS['B'], batch_norm), num_classes)

def vgg16(num_classes=1000, batch_norm=False):
    """VGG-16: 13 conv + 3 FC = 16 weight layers"""
    return VGG(make_layers(VGG_CONFIGS['D'], batch_norm), num_classes)

def vgg19(num_classes=1000, batch_norm=False):
    """VGG-19: 16 conv + 3 FC = 19 weight layers"""
    return VGG(make_layers(VGG_CONFIGS['E'], batch_norm), num_classes)
```

### VGG with Batch Normalization

The original VGG was published before batch normalization. Adding BN dramatically improves training:

```python
# Without BN: Requires careful learning rate, slow convergence
# With BN: Much faster training, higher accuracy

# Always use BN variant in practice
model = vgg16(num_classes=10, batch_norm=True)
```

## Training

### Challenges with VGG

1. **Large memory footprint**: 138M parameters require significant GPU memory
2. **Slow training**: Many parameters to update
3. **Vanishing gradients**: Deep networks without skip connections

### Training Configuration

```python
def train_vgg_cifar10():
    """Train VGG-16 with BN on CIFAR-10."""
    
    # Use smaller batch size due to model size
    config = {
        'batch_size': 64,  # Smaller than ResNet
        'epochs': 100,
        'lr': 0.001,       # Lower LR for stability
        'weight_decay': 5e-4,
    }
    
    # Data augmentation (same as ResNet)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2470, 0.2435, 0.2616)),
    ])
    
    # Model with batch normalization
    model = vgg16(num_classes=10, batch_norm=True).to(device)
    
    # Adam often works better than SGD for VGG
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                          weight_decay=config['weight_decay'])
    
    # Step decay schedule
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
```

### Expected Results on CIFAR-10

| Model | With BN | Without BN | Training Time |
|-------|---------|------------|---------------|
| VGG-11 | 89-91% | 85-88% | ~2 hours |
| VGG-16 | 91-93% | 88-90% | ~4 hours |
| VGG-19 | 91-93% | 87-90% | ~5 hours |

## Feature Visualization

### Visualizing Learned Filters

```python
def visualize_filters(model, layer_idx=0, num_filters=64):
    """Visualize convolutional filters from a specific layer."""
    
    # Get the specified conv layer
    conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    if layer_idx >= len(conv_layers):
        raise ValueError(f"Layer index {layer_idx} out of range")
    
    filters = conv_layers[layer_idx].weight.data.cpu()
    n_filters = min(num_filters, filters.shape[0])
    
    # Create grid
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # For first layer, visualize RGB
            if filters.shape[1] == 3:
                f = filters[i].permute(1, 2, 0)  # CHW -> HWC
                f = (f - f.min()) / (f.max() - f.min())  # Normalize
            else:
                # For deeper layers, take first channel
                f = filters[i, 0]
                f = (f - f.min()) / (f.max() - f.min())
            
            ax.imshow(f)
        ax.axis('off')
    
    plt.suptitle(f'Layer {layer_idx} Filters')
    plt.tight_layout()
    plt.savefig(f'vgg_filters_layer{layer_idx}.png')
```

### Visualizing Feature Maps

```python
def visualize_feature_maps(model, image, layer_idx=5, num_maps=64):
    """Visualize activation maps for a given input image."""
    
    model.eval()
    with torch.no_grad():
        # Extract features up to specified layer
        features = []
        x = image
        for i, layer in enumerate(model.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
                if len(features) > layer_idx:
                    break
        
        feature_map = features[layer_idx][0]  # Remove batch dimension
    
    # Plot feature maps
    n_maps = min(num_maps, feature_map.shape[0])
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < n_maps:
            ax.imshow(feature_map[i].cpu(), cmap='viridis')
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps at Layer {layer_idx}')
    plt.tight_layout()
    plt.savefig(f'vgg_features_layer{layer_idx}.png')
```

### Feature Hierarchy

What VGG learns at different depths:

```
Layer 1-2 (64 channels):
  - Edge detectors (horizontal, vertical, diagonal)
  - Color blobs
  - Simple gradients

Layer 3-4 (128 channels):
  - Textures
  - Corner detectors
  - Color gradients

Layer 5-7 (256 channels):
  - Texture patterns
  - Object parts
  - More complex shapes

Layer 8-13 (512 channels):
  - Object parts
  - Semantic features
  - Class-specific patterns
```

## Computational Analysis

### Parameter Distribution

```python
def analyze_vgg_parameters(model):
    """Analyze where parameters are located in VGG."""
    
    conv_params = 0
    fc_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Linear):
            fc_params += sum(p.numel() for p in module.parameters())
    
    total = conv_params + fc_params
    
    print(f"Convolutional layers: {conv_params:,} ({100*conv_params/total:.1f}%)")
    print(f"Fully connected layers: {fc_params:,} ({100*fc_params/total:.1f}%)")
    print(f"Total: {total:,}")
    
    # VGG-16 output:
    # Convolutional layers: 14,714,688 (10.7%)
    # Fully connected layers: 123,633,664 (89.3%)  ← Most parameters!
    # Total: 138,348,352
```

### Memory Requirements

```
VGG-16 Memory Usage (batch_size=1, 224×224 input):

Layer Activations:
  Conv1_1: 224×224×64 = 3.1 MB
  Conv1_2: 224×224×64 = 3.1 MB
  Pool1:   112×112×64 = 0.8 MB
  Conv2_1: 112×112×128 = 1.6 MB
  ...
  Total activations: ~90 MB

Parameters: ~138M × 4 bytes = 552 MB

Gradients (training): ~138M × 4 bytes = 552 MB

Total (training): ~1.2 GB per image!
```

## VGG vs Modern Architectures

### Comparison

| Aspect | VGG-16 | ResNet-50 | EfficientNet-B0 |
|--------|--------|-----------|-----------------|
| Parameters | 138M | 25.6M | 5.3M |
| FLOPs | 15.5B | 4.1B | 390M |
| ImageNet Top-1 | 71.3% | 76.1% | 77.3% |
| Year | 2014 | 2015 | 2019 |

### Why Study VGG?

Despite being outdated for deployment, VGG remains important:

1. **Educational value**: Simple, uniform architecture is easy to understand
2. **Feature extraction**: Pretrained VGG features work well for style transfer, perceptual loss
3. **Baseline**: Common benchmark for comparing architectures
4. **Transfer learning**: Still effective for some tasks

## Exercises

### Beginner

1. **Architecture Exploration**: Implement and compare VGG-11, VGG-13, VGG-16, VGG-19
2. **Batch Normalization Impact**: Train with and without BN, compare convergence
3. **Filter Visualization**: Visualize learned filters from layer 1 and interpret patterns

### Intermediate

4. **Receptive Field Calculation**: Calculate effective receptive field at each depth
5. **Feature Map Analysis**: Visualize what different layers respond to
6. **Memory Optimization**: Implement gradient checkpointing to reduce memory

### Advanced

7. **Slim VGG**: Design a more efficient VGG variant with fewer FC parameters
8. **VGG for Transfer Learning**: Use pretrained VGG for style transfer
9. **Comparison Study**: Systematically compare VGG vs ResNet at similar depths

## Key Takeaways

1. **Small filters are better**: 3×3 convolutions achieve equivalent receptive field with fewer parameters
2. **Uniform design**: Consistent architecture is easier to understand and implement
3. **Batch normalization is essential**: Dramatically improves VGG training
4. **FC layers dominate parameters**: >89% of VGG parameters are in fully connected layers
5. **Historical importance**: VGG demonstrated depth matters, influencing all subsequent architectures

## References

1. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
2. Ioffe, S., & Szegedy, C. (2015). Batch Normalization. ICML.
3. Gatys, L. A., et al. (2016). Image Style Transfer Using CNNs. CVPR.

---

**Previous Section**: [ResNet](resnet.md) | **Next Section**: [Inception/GoogLeNet](inception.md)
