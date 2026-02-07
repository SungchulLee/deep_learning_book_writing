# Feature Maps

## Introduction

A **feature map** is the output of applying a convolutional filter to an input—it is a spatial grid of activation values that represents where and how strongly a particular pattern (feature) is detected. Feature maps are the fundamental intermediate representations in CNNs, encoding progressively more abstract information as data flows through the network.

Understanding feature maps is essential for:

1. **Architecture design**: Choosing appropriate channel dimensions and spatial resolutions at each layer
2. **Parameter budgeting**: Computing memory and parameter requirements for a given architecture
3. **Debugging and interpretation**: Visualizing what the network has learned at each stage
4. **Performance optimization**: Identifying computational bottlenecks across layers

---

## Feature Map Geometry

### Tensor Shape Convention

In PyTorch, a feature map tensor has shape $(N, C, H, W)$:

| Dimension | Symbol | Meaning |
|-----------|--------|---------|
| Batch | $N$ | Number of samples processed in parallel |
| Channels | $C$ | Number of feature maps (depth) |
| Height | $H$ | Spatial height |
| Width | $W$ | Spatial width |

```python
import torch
import torch.nn as nn

# A batch of 8 RGB images of size 224×224
x = torch.randn(8, 3, 224, 224)

# After first conv layer: 64 feature maps of size 224×224
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
feat1 = conv1(x)
print(f"Input:  N={x.shape[0]}, C={x.shape[1]}, H={x.shape[2]}, W={x.shape[3]}")
print(f"After conv1: N={feat1.shape[0]}, C={feat1.shape[1]}, H={feat1.shape[2]}, W={feat1.shape[3]}")
# Input:  N=8, C=3, H=224, W=224
# After conv1: N=8, C=64, H=224, W=224
```

### Spatial vs. Channel Dimensions

Feature maps encode two distinct types of information:

- **Spatial dimensions** ($H \times W$): *Where* features are located in the image
- **Channel dimension** ($C$): *What* features are detected at each location

Each channel corresponds to a different learned filter, detecting a different pattern. The spatial grid preserves the relative positions of detected features.

```
Feature Map Stack (C×H×W):

Channel 0 (e.g., horizontal edges):    Channel 1 (e.g., vertical edges):
┌─────────────────┐                    ┌─────────────────┐
│ 0.0  0.0  0.0   │                    │ 0.0  0.8  0.0   │
│ 0.9  0.8  0.7   │                    │ 0.0  0.9  0.0   │
│ 0.0  0.0  0.0   │                    │ 0.0  0.7  0.0   │
└─────────────────┘                    └─────────────────┘

At each spatial position (i,j), the vector across channels
forms a local feature descriptor.
```

---

## How Feature Maps Evolve Through a Network

### The Feature Hierarchy

As data flows through a CNN, feature maps undergo a characteristic transformation:

```
Input Image (3×224×224)
        │
        ▼
    Conv Block 1 ──→ 64 × 224 × 224   (edges, colors, simple textures)
        │
        ▼ (downsample)
    Conv Block 2 ──→ 128 × 112 × 112  (corners, texture combinations)
        │
        ▼ (downsample)
    Conv Block 3 ──→ 256 × 56 × 56    (parts: eyes, wheels, windows)
        │
        ▼ (downsample)
    Conv Block 4 ──→ 512 × 28 × 28    (objects, semantic concepts)
        │
        ▼ (downsample)
    Conv Block 5 ──→ 512 × 14 × 14    (scene-level, abstract)
        │
        ▼ (global average pooling)
    Feature Vector ──→ 512 × 1 × 1
```

The pattern is consistent: **spatial resolution decreases** while **channel count increases**. This reflects the transition from fine-grained spatial detail to rich semantic abstraction.

### The Resolution-Semantics Trade-off

| Property | Early Layers | Deep Layers |
|----------|-------------|-------------|
| Spatial resolution | High ($224 \times 224$) | Low ($7 \times 7$) |
| Channel count | Low (64) | High (512+) |
| Feature type | Low-level (edges, textures) | High-level (objects, scenes) |
| Translation sensitivity | High (precise localization) | Low (position-invariant) |
| Receptive field | Small (local context) | Large (global context) |

---

## Parameter and Memory Analysis

### Parameter Count per Layer

For a convolutional layer with kernel size $K$:

$$\text{Parameters} = C_{out} \times C_{in} \times K^2 + C_{out}$$

### Activation Memory per Layer

The memory required to store a feature map is:

$$\text{Memory (elements)} = N \times C \times H \times W$$
$$\text{Memory (bytes)} = N \times C \times H \times W \times \text{bytes per element}$$

For float32 (4 bytes per element), a single $64 \times 224 \times 224$ feature map uses approximately $64 \times 224 \times 224 \times 4 \approx 12.3$ MB.

### Layer-by-Layer Analysis

```python
import torch
import torch.nn as nn

def analyze_feature_maps(model, input_shape=(1, 3, 224, 224)):
    """
    Analyze feature map shapes, parameters, and memory through a model.
    """
    x = torch.randn(*input_shape)
    
    print(f"{'Layer':<30} {'Output Shape':<25} {'Params':>12} {'Memory (MB)':>12}")
    print("-" * 82)
    print(f"{'Input':<30} {str(list(x.shape)):<25} {'—':>12} {x.numel()*4/1e6:>12.2f}")
    
    total_params = 0
    total_memory = x.numel() * 4  # Input memory
    
    for name, layer in model.named_children():
        x = layer(x)
        params = sum(p.numel() for p in layer.parameters())
        mem_mb = x.numel() * 4 / 1e6
        total_params += params
        total_memory += x.numel() * 4
        
        print(f"{name:<30} {str(list(x.shape)):<25} {params:>12,} {mem_mb:>12.2f}")
    
    print("-" * 82)
    print(f"{'Total':<30} {'':25} {total_params:>12,} {total_memory/1e6:>12.2f}")
    
    return x


# Example: VGG-style feature extractor
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),    # Block 1
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(64, 128, 3, padding=1),  # Block 2
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(128, 256, 3, padding=1), # Block 3
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
)

# Use named Sequential for cleaner output
named_model = nn.Sequential()
names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3']
for name, layer in zip(names, model):
    named_model.add_module(name, layer)

analyze_feature_maps(named_model)
```

---

## The "Double Channels, Halve Resolution" Pattern

Modern CNN architectures follow a consistent design pattern: when spatial resolution is halved (via stride-2 convolution or pooling), the number of channels is doubled. This maintains roughly constant computational cost per layer:

$$\text{FLOPs} \propto C \times C \times K^2 \times H \times W$$

Doubling $C$ while halving both $H$ and $W$:
$$2C \times 2C \times K^2 \times \frac{H}{2} \times \frac{W}{2} = C^2 K^2 HW$$

The FLOPs remain approximately constant, creating a balanced computational profile across the network.

```python
import torch.nn as nn

# ResNet-style channel progression
stages = [
    ("Stage 1", 64,  56, 56),   # After stem
    ("Stage 2", 128, 28, 28),   # 2× channels, 0.5× resolution
    ("Stage 3", 256, 14, 14),   # 4× channels, 0.25× resolution
    ("Stage 4", 512, 7,  7),    # 8× channels, 0.125× resolution
]

print(f"{'Stage':<12} {'Channels':>8} {'Resolution':>12} {'Elements':>12} {'Relative':>10}")
print("-" * 60)
base_elements = None
for name, c, h, w in stages:
    elements = c * h * w
    if base_elements is None:
        base_elements = elements
    print(f"{name:<12} {c:>8} {f'{h}×{w}':>12} {elements:>12,} {elements/base_elements:>10.2f}×")
```

---

## Feature Map Visualization

### Visualizing Learned Features

Examining feature maps helps understand what each layer detects:

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_maps(model, image, layer_name, max_channels=16):
    """
    Visualize feature maps at a specific layer.
    
    Args:
        model: CNN model
        image: Input tensor (1, C, H, W)
        layer_name: Name of the layer to visualize
        max_channels: Maximum number of channels to display
    """
    # Register hook to capture feature maps
    features = {}
    def hook(module, input, output):
        features['output'] = output.detach()
    
    # Find and hook the target layer
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image)
    
    handle.remove()
    
    # Visualize
    feat = features['output'].squeeze(0)  # Remove batch dim
    num_channels = min(feat.shape[0], max_channels)
    
    rows = int(np.ceil(num_channels / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
    
    for i in range(num_channels):
        axes[i].imshow(feat[i].cpu().numpy(), cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps at {layer_name} ({feat.shape[0]} channels, '
                 f'{feat.shape[1]}×{feat.shape[2]})')
    plt.tight_layout()
    plt.savefig(f'feature_maps_{layer_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Channel Statistics

```python
def feature_map_statistics(model, loader, layer_name, num_batches=10):
    """
    Compute activation statistics for a layer across the dataset.
    Useful for diagnosing dead neurons, saturation, and distribution issues.
    """
    features_list = []
    
    def hook(module, input, output):
        features_list.append(output.detach())
    
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            _ = model(images)
    
    handle.remove()
    
    # Concatenate and analyze
    all_features = torch.cat(features_list, dim=0)
    
    # Per-channel statistics
    channel_means = all_features.mean(dim=(0, 2, 3))
    channel_stds = all_features.std(dim=(0, 2, 3))
    channel_sparsity = (all_features == 0).float().mean(dim=(0, 2, 3))
    
    print(f"Layer: {layer_name}")
    print(f"  Feature map shape: {list(all_features.shape[1:])}")
    print(f"  Mean activation: {channel_means.mean():.4f} ± {channel_means.std():.4f}")
    print(f"  Std activation: {channel_stds.mean():.4f}")
    print(f"  Sparsity (% zeros): {channel_sparsity.mean():.1%}")
    print(f"  Dead channels (>99% zero): {(channel_sparsity > 0.99).sum().item()}")
    
    return channel_means, channel_stds, channel_sparsity
```

---

## Feature Maps in Multi-Channel Convolution

### How a Single Convolution Produces a Feature Map

Each output channel is produced by a different 3D kernel that spans all input channels. The kernel slides over the spatial dimensions, computing a dot product at each position:

$$\text{Feature Map}[o, i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m,n} X[c, i+m, j+n] \cdot W[o, c, m, n] + b[o]$$

Each of the $C_{out}$ kernels learns to detect a different spatial pattern across all input channels simultaneously.

### The $1 \times 1$ Convolution as Channel Mixer

A $1 \times 1$ convolution performs no spatial filtering—it operates purely on the channel dimension at each spatial position:

$$Y[o, i, j] = \sum_{c=0}^{C_{in}-1} X[c, i, j] \cdot W[o, c] + b[o]$$

This is equivalent to applying a shared fully connected layer independently at each spatial position. Uses include:

- **Channel reduction**: Reducing $C_{in}$ to a smaller $C_{out}$ (bottleneck)
- **Channel expansion**: Increasing channel dimension
- **Cross-channel learning**: Combining information across feature maps
- **Adding non-linearity**: When followed by activation functions

```python
# 1×1 convolution: pure channel mixing
channel_mixer = nn.Conv2d(256, 64, kernel_size=1)  # 256 → 64 channels

x = torch.randn(1, 256, 14, 14)
out = channel_mixer(x)
print(f"Channel reduction: {x.shape} → {out.shape}")
# Channel reduction: [1, 256, 14, 14] → [1, 64, 14, 14]

params = sum(p.numel() for p in channel_mixer.parameters())
print(f"Parameters: {params:,}")  # 256×64 + 64 = 16,448
```

---

## Feature Map Reuse: Skip Connections

Skip (residual) connections allow feature maps from earlier layers to bypass intermediate layers and be added to or concatenated with later feature maps:

### Additive Skip (ResNet)

$$\mathbf{Y} = F(\mathbf{X}) + \mathbf{X}$$

Requires matching spatial dimensions and channel counts.

### Concatenation Skip (DenseNet, U-Net)

$$\mathbf{Y} = [\mathbf{X}, F(\mathbf{X})]$$

Concatenates along the channel dimension, creating increasingly rich feature representations.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Additive skip: preserves feature map dimensions."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)  # Feature map reuse


class DenseBlock(nn.Module):
    """Concatenation skip: feature maps accumulate."""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False),
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)  # All feature maps concatenated
```

---

## Summary

| Concept | Description |
|---------|-------------|
| **Shape** | $(N, C, H, W)$: batch, channels, height, width |
| **Channels** | Each channel = one learned feature detector's spatial response |
| **Hierarchy** | Early: edges/textures → Deep: objects/semantics |
| **Design pattern** | Double channels when halving resolution |
| **$1 \times 1$ conv** | Pure channel mixing, no spatial filtering |
| **Skip connections** | Reuse earlier feature maps (add or concatenate) |
| **Visualization** | Hook-based extraction reveals learned representations |

## Key Takeaways

1. **Feature maps are 3D tensors** ($C \times H \times W$) where channels encode *what* and spatial dims encode *where*
2. **Progressive abstraction**: Networks transform high-resolution, low-channel inputs into low-resolution, high-channel representations
3. **The double-channels-halve-resolution pattern** maintains balanced computation across stages
4. **$1 \times 1$ convolutions** enable efficient channel dimension manipulation without spatial computation
5. **Activation memory often exceeds parameter memory**, especially for high-resolution inputs—this is a key concern for GPU memory budgeting
6. **Feature map visualization** through hooks is an essential debugging and interpretability tool

## References

1. Zeiler, M. D., & Fergus, R. (2014). "Visualizing and Understanding Convolutional Networks." *ECCV*.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.

3. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). "Densely Connected Convolutional Networks." *CVPR*.

4. Lin, M., Chen, Q., & Yan, S. (2014). "Network In Network." *ICLR*. (Introduced 1×1 convolutions)

5. Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.
