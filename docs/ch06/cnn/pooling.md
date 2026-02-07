# Pooling Layers

## Introduction

**Pooling layers** are fundamental components of CNNs that perform spatial downsampling by summarizing local regions of feature maps. They serve several critical purposes in neural network architectures:

1. **Dimensionality reduction**: Decrease spatial dimensions, reducing computation and memory in subsequent layers
2. **Translation invariance**: Make features more robust to small spatial shifts—a key inductive bias for vision
3. **Increased receptive field**: Allow subsequent layers to "see" more of the input with fewer parameters
4. **Regularization**: Reduce overfitting by providing spatial regularization through information compression
5. **Feature summarization**: Extract dominant features from local regions, discarding less important details

Understanding pooling deeply requires grasping both its computational mechanics and its role in building hierarchical representations.

---

## Types of Pooling

### Max Pooling

**Max pooling** selects the maximum value within each pooling window:

$$y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}$$

where $R_{i,j}$ is the pooling region for output position $(i, j)$.

**Visual Example (2×2 max pooling, stride 2):**

```
Input (4×4)               Output (2×2)
┌───┬───┬───┬───┐        ┌───┬───┐
│ 1 │ 3 │ 2 │ 4 │        │ 5 │ 6 │
├───┼───┼───┼───┤   →    ├───┼───┤
│ 5 │ 2 │ 6 │ 1 │        │ 7 │ 8 │
├───┼───┼───┼───┤        └───┴───┘
│ 4 │ 7 │ 3 │ 8 │
├───┼───┼───┼───┤
│ 2 │ 1 │ 5 │ 2 │
└───┴───┴───┴───┘

Region [0:2, 0:2]: max(1,3,5,2) = 5
Region [0:2, 2:4]: max(2,4,6,1) = 6
Region [2:4, 0:2]: max(4,7,2,1) = 7
Region [2:4, 2:4]: max(3,8,5,2) = 8
```

**Properties:**

| Property | Description |
|----------|-------------|
| Activation selection | Preserves strongest (most confident) activations |
| Translation invariance | Small shifts don't change the max within a window |
| Sparsity | Only one position per window contributes to output |
| Edge preservation | Good at preserving texture and edge information |
| Gradient flow | Sparse—only the max position receives gradient |

**Why max pooling works:** In feature detection, we often care about *whether* a feature is present, not its exact location. Max pooling answers "is this feature anywhere in this region?" by keeping the strongest response.

### Average Pooling

**Average pooling** computes the mean within each pooling window:

$$y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} x_{m,n}$$

**Visual Example (2×2 average pooling, stride 2):**

```
Input (4×4)               Output (2×2)
┌───┬───┬───┬───┐        ┌─────┬─────┐
│ 4 │ 2 │ 8 │ 0 │        │ 2.0 │ 3.0 │
├───┼───┼───┼───┤   →    ├─────┼─────┤
│ 0 │ 2 │ 4 │ 0 │        │ 4.0 │ 4.0 │
├───┼───┼───┼───┤        └─────┴─────┘
│ 8 │ 4 │ 4 │ 4 │
├───┼───┼───┼───┤
│ 4 │ 0 │ 4 │ 0 │
└───┴───┴───┴───┘

Region [0:2, 0:2]: avg(4,2,0,2) = 2.0
Region [0:2, 2:4]: avg(8,0,4,0) = 3.0
Region [2:4, 0:2]: avg(8,4,4,0) = 4.0
Region [2:4, 2:4]: avg(4,4,4,0) = 4.0
```

**Properties:**

| Property | Description |
|----------|-------------|
| Activation preservation | Preserves overall activation level (energy) |
| Smoothing | Acts as a low-pass filter, reducing noise |
| Dense gradient | All positions contribute equally to gradient |
| Global context | Better for tasks requiring holistic understanding |

**When to prefer average pooling:** When all activations carry meaningful information and you want to preserve the overall "energy" rather than just the peak response.

### Global Average Pooling (GAP)

**Global average pooling** reduces each entire feature map (channel) to a single scalar:

$$y_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{c,i,j}$$

```
Input (C×H×W):                    Output (C×1×1):
┌─────────────────┐               ┌───┐
│ Channel 1 (H×W) │    GAP        │ v₁│
│ Channel 2 (H×W) │    →          │ v₂│
│    ...          │               │...│
│ Channel C (H×W) │               │ vC│
└─────────────────┘               └───┘
```

**Why GAP revolutionized classification:**

| Aspect | Fully Connected | Global Average Pooling |
|--------|-----------------|------------------------|
| Parameters | $C \times H \times W \times \text{classes}$ | 0 (+ small 1×1 conv) |
| Example (512×7×7→1000) | 25,088,000 params | 512,000 params (98% reduction) |
| Spatial flexibility | Fixed input size | Any input size |
| Interpretability | Black box | Feature map = class activation |
| Overfitting risk | High | Low |

GAP enforces a direct correspondence between feature maps and categories, making the network more interpretable and less prone to overfitting.

### Global Max Pooling

Similar to GAP but takes the maximum across spatial dimensions:

$$y_c = \max_{i,j} x_{c,i,j}$$

Useful when you want to detect if a feature appears *anywhere* in the image, regardless of its extent.

### L2 Pooling (Root Mean Square)

$$y_{i,j} = \sqrt{\frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} x_{m,n}^2}$$

Emphasizes larger activations more than average pooling but less aggressively than max pooling. Provides a middle ground.

### LP Pooling (Generalized)

LP pooling generalizes average (p=1) and approaches max (p→∞) pooling:

$$y_{i,j} = \left( \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} |x_{m,n}|^p \right)^{1/p}$$

The parameter $p$ controls the "hardness" of the pooling:
- $p=1$: Average pooling
- $p=2$: L2 (RMS) pooling
- $p→∞$: Approaches max pooling

### Stochastic Pooling

Samples from the multinomial distribution based on activation magnitudes:

$$P(k) = \frac{x_k}{\sum_i x_i}, \quad y = x_k \text{ where } k \sim P$$

**Key insight:** During training, provides regularization by introducing randomness. At test time, uses expectation (weighted average). Combines benefits of max pooling (selecting strong activations) with average pooling (considering all values).

## Mathematical Analysis

### Output Dimensions

For input size $H_{in} \times W_{in}$, kernel size $K$, stride $s$, padding $p$, and dilation $d$:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K - 1) - 1}{s} \right\rfloor + 1$$

**Common configurations:**

| Configuration | Effect | Example (224 input) |
|--------------|--------|---------------------|
| K=2, s=2, p=0 | Halve dimensions | 224 → 112 |
| K=3, s=2, p=1 | Halve with overlap | 224 → 112 |
| K=7, s=1, p=0 | Reduce by 6 | 7 → 1 (global) |

### Gradient Computation

Understanding gradients is crucial for training dynamics.

#### Max Pooling Gradient

The gradient flows **only** through the maximum element (sparse routing):

$$\frac{\partial y_{i,j}}{\partial x_{m,n}} = \begin{cases}
1 & \text{if } (m,n) = \arg\max_{(m',n') \in R_{i,j}} x_{m',n'} \\
0 & \text{otherwise}
\end{cases}$$

**Implementation requirement:** Must track indices of maximum elements during forward pass.

**Implication:** Only the "winning" neuron learns. This creates competition—neurons must produce the strongest response to receive gradient signal.

#### Average Pooling Gradient

The gradient is distributed **equally** to all contributing elements:

$$\frac{\partial y_{i,j}}{\partial x_{m,n}} = \frac{1}{|R_{i,j}|}$$

for all $(m,n) \in R_{i,j}$.

**Implication:** All neurons in the pooling region learn equally. More stable gradients but potentially slower learning of discriminative features.

## PyTorch Implementation

### Basic Pooling Operations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create sample input: (batch, channels, height, width)
x = torch.tensor([[[[1., 3., 2., 4.],
                    [2., 1., 1., 2.],
                    [5., 4., 6., 2.],
                    [3., 1., 1., 3.]]]])

print(f"Input shape: {x.shape}")  # [1, 1, 4, 4]
print(f"Input:\n{x.squeeze()}\n")

# Max Pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
out_max = max_pool(x)
print(f"Max pool output:\n{out_max.squeeze()}")
# tensor([[3., 4.],
#         [5., 6.]])

# Average Pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
out_avg = avg_pool(x)
print(f"\nAvg pool output:\n{out_avg.squeeze()}")
# tensor([[1.75, 2.25],
#         [3.25, 3.00]])

# Global Average Pooling
gap = nn.AdaptiveAvgPool2d(1)
out_gap = gap(x)
print(f"\nGAP output: {out_gap.item():.4f}")  # 2.5625

# Global Max Pooling
gmp = nn.AdaptiveMaxPool2d(1)
out_gmp = gmp(x)
print(f"Global Max output: {out_gmp.item():.4f}")  # 6.0
```

### Pooling Parameters Deep Dive

```python
import torch.nn as nn

# Max pooling with various configurations
# kernel_size: Size of pooling window
# stride: Step size (default: kernel_size)
# padding: Zero padding added to input
# dilation: Spacing between kernel elements (atrous/dilated pooling)
# return_indices: Return indices of max values (for unpooling)
# ceil_mode: Use ceiling instead of floor for output size

# Standard 2×2 max pool with stride 2 (halves dimensions)
pool_standard = nn.MaxPool2d(kernel_size=2, stride=2)

# Overlapping pooling (stride < kernel_size)
# Used in AlexNet - slightly better than non-overlapping
pool_overlap = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

# Dilated pooling (sparse sampling)
pool_dilated = nn.MaxPool2d(kernel_size=2, stride=2, dilation=2)

# Return indices for unpooling (encoder-decoder architectures)
pool_indices = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

x = torch.randn(1, 64, 32, 32)

# Using return_indices
out, indices = pool_indices(x)
print(f"Output shape: {out.shape}")      # [1, 64, 16, 16]
print(f"Indices shape: {indices.shape}") # [1, 64, 16, 16]
print(f"Indices contain positions of max values (0 to {2*2-1} per window)")
```

### Adaptive Pooling

Adaptive pooling produces a **fixed output size** regardless of input size—essential for handling variable input dimensions:

```python
import torch
import torch.nn as nn

# Adaptive Average Pooling - specify OUTPUT size, not kernel
adaptive_avg = nn.AdaptiveAvgPool2d((7, 7))  # Always outputs 7×7
adaptive_max = nn.AdaptiveMaxPool2d((4, 4))  # Always outputs 4×4

# Global pooling (output size 1×1)
global_avg = nn.AdaptiveAvgPool2d(1)
global_max = nn.AdaptiveMaxPool2d(1)

# Demonstrate flexibility with different input sizes
print("Adaptive pooling handles any input size:")
for size in [14, 28, 56, 224]:
    x = torch.randn(1, 64, size, size)
    out_avg = adaptive_avg(x)
    out_max = adaptive_max(x)
    print(f"  Input: {size}×{size} → Avg: {out_avg.shape[-2]}×{out_avg.shape[-1]}, "
          f"Max: {out_max.shape[-2]}×{out_max.shape[-1]}")

# Output:
# Input: 14×14 → Avg: 7×7, Max: 4×4
# Input: 28×28 → Avg: 7×7, Max: 4×4
# Input: 56×56 → Avg: 7×7, Max: 4×4
# Input: 224×224 → Avg: 7×7, Max: 4×4
```

**How adaptive pooling works:** It automatically computes the kernel size and stride needed to produce the desired output size from the given input size.

### Output Size Calculation Helper

```python
import numpy as np

def pool_output_size(input_size, kernel_size, stride, padding=0, dilation=1, ceil_mode=False):
    """Calculate pooling output size."""
    numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    if ceil_mode:
        return int(np.ceil(numerator / stride)) + 1
    else:
        return numerator // stride + 1

# Examples
print("Output size calculations:")
print(f"  224 with k=2, s=2: {pool_output_size(224, 2, 2)}")  # 112
print(f"  112 with k=2, s=2: {pool_output_size(112, 2, 2)}")  # 56
print(f"  56 with k=3, s=2, p=1: {pool_output_size(56, 3, 2, padding=1)}")  # 28
print(f"  7 with k=7, s=1: {pool_output_size(7, 7, 1)}")      # 1 (global)
```

## Max Pooling Gradient Flow

Understanding how gradients flow through max pooling is crucial for debugging and architecture design.

```python
import torch
import torch.nn as nn

# Demonstrate gradient flow through max pooling
x = torch.tensor([[[[1., 3., 2., 4.],
                    [2., 1., 1., 2.],
                    [5., 4., 6., 2.],
                    [3., 1., 1., 3.]]]], requires_grad=True)

pool = nn.MaxPool2d(2, 2)
out = pool(x)

print("Input:")
print(x.data.squeeze())
print("\nOutput (max values from each 2×2 region):")
print(out.data.squeeze())

# Backward pass
loss = out.sum()
loss.backward()

print("\nGradient (1 only at max positions, 0 elsewhere):")
print(x.grad.squeeze())

# Expected gradient shows 1 at positions:
# (0,1)=3, (0,3)=4, (2,0)=5, (2,2)=6 - where maxes occurred
```

**Output:**
```
Input:
tensor([[1., 3., 2., 4.],
        [2., 1., 1., 2.],
        [5., 4., 6., 2.],
        [3., 1., 1., 3.]])

Output (max values from each 2×2 region):
tensor([[3., 4.],
        [5., 6.]])

Gradient (1 only at max positions, 0 elsewhere):
tensor([[0., 1., 0., 1.],
        [0., 0., 0., 0.],
        [1., 0., 1., 0.],
        [0., 0., 0., 0.]])
```

**Key insight:** Only 4 of 16 input positions receive gradient signal. This sparsity can be both a strength (focus learning on important features) and a limitation (some neurons rarely learn).

## Max Unpooling

Max unpooling reverses max pooling using stored indices—essential for encoder-decoder architectures like SegNet:

```python
import torch
import torch.nn as nn

# Create encoder-decoder pair
max_pool = nn.MaxPool2d(2, 2, return_indices=True)
max_unpool = nn.MaxUnpool2d(2, 2)

# Input
x = torch.randn(1, 1, 4, 4)
print("Original:")
print(x.squeeze().round(decimals=2))

# Encode: pool and store indices
pooled, indices = max_pool(x)
print("\nPooled:")
print(pooled.squeeze().round(decimals=2))

# Decode: unpool using stored indices
unpooled = max_unpool(pooled, indices)
print("\nUnpooled (sparse reconstruction):")
print(unpooled.squeeze().round(decimals=2))

# Note: Unpooled has zeros except at original max positions
# Information from non-max positions is lost
```

## Advanced Pooling Methods

### Fractional Max Pooling

Uses randomized pooling regions for data augmentation:

```python
# Fractional pooling: non-integer reduction ratios
frac_pool = nn.FractionalMaxPool2d(
    kernel_size=2,
    output_ratio=(0.7, 0.7),  # Output is ~70% of input size
    return_indices=True
)

x = torch.randn(1, 64, 32, 32)
y, indices = frac_pool(x)
print(f"Fractional pooling: {x.shape} → {y.shape}")  # Approximately 22×22
```

**Why it helps:** The randomized boundaries provide regularization during training, similar to dropout but in the spatial domain.

### Spatial Pyramid Pooling (SPP)

Multi-scale pooling that produces fixed-size output regardless of input—enables handling variable image sizes:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling layer.
    
    Pools at multiple scales and concatenates, producing
    a fixed-size output regardless of input spatial dimensions.
    
    Key insight: Different scales capture different context levels:
    - 1×1: Global context (what's in the whole image)
    - 2×2: Quadrant context (what's in each quarter)
    - 4×4: Local context (finer spatial detail)
    """
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels
        # Output size = channels × sum(level²) for all levels
    
    def forward(self, x):
        batch, channels, H, W = x.shape
        outputs = []
        
        for level in self.levels:
            # Adaptive pooling to level×level grid
            pooled = F.adaptive_max_pool2d(x, (level, level))
            # Flatten spatial dimensions
            pooled = pooled.view(batch, channels, -1)
            outputs.append(pooled)
        
        # Concatenate along spatial dimension
        # Output: (batch, channels, 1+4+16) = (batch, channels, 21)
        return torch.cat(outputs, dim=2)


spp = SpatialPyramidPooling([1, 2, 4])

# Works with ANY input size - produces same output size
print("SPP produces fixed output regardless of input:")
for size in [7, 14, 28, 56]:
    x = torch.randn(1, 256, size, size)
    out = spp(x)
    print(f"  Input: {size}×{size} → Output: {out.shape}")
# All produce: (1, 256, 21)
```

### Region of Interest (RoI) Pooling

Extracts fixed-size features from arbitrary regions—fundamental to object detection:

```python
from torchvision.ops import roi_pool, roi_align

# Feature map from backbone
features = torch.randn(1, 256, 14, 14)

# Regions of interest: (batch_index, x1, y1, x2, y2)
rois = torch.tensor([
    [0, 0, 0, 7, 7],      # Top-left quadrant
    [0, 7, 0, 14, 7],     # Top-right quadrant
    [0, 3, 3, 10, 10],    # Center region
], dtype=torch.float32)

# RoI Pool: quantizes coordinates (introduces misalignment)
pooled_roi = roi_pool(features, rois, output_size=(7, 7), spatial_scale=1.0)
print(f"RoI Pool output: {pooled_roi.shape}")  # (3, 256, 7, 7)

# RoI Align: uses bilinear interpolation (pixel-perfect alignment)
aligned_roi = roi_align(features, rois, output_size=(7, 7), spatial_scale=1.0)
print(f"RoI Align output: {aligned_roi.shape}")  # (3, 256, 7, 7)
```

**RoI Pool vs RoI Align:** RoI Align (from Mask R-CNN) eliminates quantization artifacts by using bilinear interpolation, critical for pixel-precise tasks like instance segmentation.

## Pooling in Classic Architectures

### Historical Evolution

```python
import torch.nn as nn

# LeNet-5 (1998): Average pooling (called "subsampling")
# Motivation: Biological plausibility, smoothing
lenet_pool = nn.AvgPool2d(2, 2)

# AlexNet (2012): Overlapping max pooling
# Innovation: k=3, s=2 slightly better than k=2, s=2
alexnet_pool = nn.MaxPool2d(kernel_size=3, stride=2)

# VGG (2014): Standard max pooling
# Philosophy: Simplicity, uniform 2×2 pooling
vgg_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# GoogLeNet/Inception (2014): Multiple parallel pooling paths
# Innovation: Different receptive fields processed in parallel

# ResNet (2015): Strided convolution replaces some pooling
# Shift: Learnable downsampling gaining preference
```

### ResNet-style Stem

```python
class ResNetStem(nn.Module):
    """
    ResNet initial downsampling: aggressive early reduction.
    
    224×224 → 112×112 (conv stride 2) → 56×56 (max pool)
    Total 4× reduction before main blocks.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)    # 224 → 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112 → 56
        return x

stem = ResNetStem()
x = torch.randn(1, 3, 224, 224)
out = stem(x)
print(f"ResNet stem: 224×224 → {out.shape[-1]}×{out.shape[-1]}")  # 56×56
```

### Global Average Pooling for Classification

```python
class ModernClassifier(nn.Module):
    """
    Modern classification head using Global Average Pooling.
    
    Replaces the massive FC layers of AlexNet/VGG with:
    1. Optional 1×1 conv to adjust channels
    2. Global Average Pooling
    3. Single linear layer (or none if using 1×1 conv)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.gap(x)        # (N, C, H, W) → (N, C, 1, 1)
        x = x.flatten(1)       # (N, C, 1, 1) → (N, C)
        x = self.fc(x)         # (N, C) → (N, num_classes)
        return x


# Parameter comparison
in_channels = 512
spatial_size = 7
num_classes = 1000

# VGG-style: FC after flatten (512×7×7 → 4096 → 4096 → 1000)
vgg_fc_params = (512 * 7 * 7 * 4096) + (4096 * 4096) + (4096 * 1000)
print(f"VGG-style FC params: {vgg_fc_params:,}")  # ~119 million

# Modern: GAP + single FC (512 → 1000)
modern_params = 512 * 1000 + 1000  # weights + bias
print(f"Modern GAP params: {modern_params:,}")  # 513,000

print(f"Parameter reduction: {vgg_fc_params / modern_params:.0f}×")
```

## Pooling vs Strided Convolution

Modern architectures increasingly replace pooling with strided convolutions. Understanding the trade-offs is essential.

### Detailed Comparison

| Aspect | Pooling (Max/Avg) | Strided Convolution |
|--------|-------------------|---------------------|
| **Parameters** | 0 | $K^2 \times C_{in} \times C_{out}$ |
| **Learnable** | No (fixed operation) | Yes (learned downsampling) |
| **Information** | Max or average | Learned weighted combination |
| **Gradient flow** | Sparse (max) or uniform (avg) | Dense (all positions contribute) |
| **Computation** | Cheaper (no multiply-add) | More expensive |
| **Flexibility** | Fixed behavior | Adapts to data |
| **When to use** | Early layers, regularization | Modern architectures, GANs |

### Practical Comparison

```python
import torch
import torch.nn as nn

x = torch.randn(1, 64, 32, 32)

# Max pooling (no parameters)
pool = nn.MaxPool2d(2, 2)
out_pool = pool(x)

# Strided convolution (learnable)
strided_conv = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
out_conv = strided_conv(x)

pool_params = 0
conv_params = sum(p.numel() for p in strided_conv.parameters())

print(f"Max pool:     {out_pool.shape}, params: {pool_params}")
print(f"Strided conv: {out_conv.shape}, params: {conv_params:,}")
# Strided conv: 36,928 parameters (64×64×3×3 + 64)
```

### "Striding for Simplicity" Insight

The "All Convolutional Net" paper showed that replacing pooling with strided convolutions can match or exceed pooling performance:

```python
# Classic CNN pattern (VGG-style)
classic_block = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Non-learnable downsampling
)

# Modern pattern (All-Conv / ResNet-style)
modern_block = nn.Sequential(
    nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Learnable downsampling
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
)
```

**Key insight:** Strided convolutions provide denser gradient flow, which can improve training dynamics, especially in generative models (GANs) where checkerboard artifacts from upsampling need strong gradient signal to correct.

## Complete Example: Pooling Comparison Network

```python
import torch
import torch.nn as nn

class PoolingComparisonNetwork(nn.Module):
    """
    Network demonstrating different pooling strategies.
    Useful for empirical comparison of pooling methods.
    """
    def __init__(self, num_classes=10, pooling_type='max'):
        super().__init__()
        
        # Feature extraction block 1
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Pooling layer selection
        if pooling_type == 'max':
            self.pool1 = nn.MaxPool2d(2, 2)
            self.pool2 = nn.MaxPool2d(2, 2)
        elif pooling_type == 'avg':
            self.pool1 = nn.AvgPool2d(2, 2)
            self.pool2 = nn.AvgPool2d(2, 2)
        elif pooling_type == 'strided':
            self.pool1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
            self.pool2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        
        # Feature extraction block 2
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling and classifier (always use GAP here)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)
        
        self.pooling_type = pooling_type
    
    def forward(self, x):
        x = self.features1(x)
        x = self.pool1(x)
        x = self.features2(x)
        x = self.pool2(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# Compare different pooling types
print("Pooling comparison (CIFAR-10 style input 32×32):")
for pool_type in ['max', 'avg', 'strided']:
    model = PoolingComparisonNetwork(num_classes=10, pooling_type=pool_type)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  {pool_type:8s} pooling: output {out.shape}, params: {params:,}")
```

## Best Practices Summary

### When to Use Max Pooling

1. **Image classification**: Preserves discriminative features (edges, textures)
2. **Early layers**: Aggressive feature selection, translation invariance
3. **When you care about presence, not magnitude**: "Is this feature here?"

### When to Use Average Pooling

1. **Global pooling before classifier**: GAP is the modern standard
2. **Semantic segmentation**: Preserves spatial structure better
3. **When all activations are meaningful**: Dense prediction tasks

### When to Use Strided Convolution

1. **Modern architectures**: ResNet, EfficientNet, ConvNeXt
2. **Generative models**: Better gradient flow for GANs, VAEs
3. **When you want learnable downsampling**: Let the network decide

### Architecture Patterns

```python
# Modern classification network pattern
class ModernCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Stem: aggressive downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),  # Pooling OK in stem
        )
        
        # Body: strided convolutions for downsampling
        self.body = nn.Sequential(
            # Stage 1-4 with strided conv between stages
            # ...
        )
        
        # Head: GAP + linear (no FC layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )
```

## Key Takeaways

1. **Max pooling** selects strongest activations, providing translation invariance and sparse gradient flow
2. **Average pooling** preserves overall activation levels with dense gradient distribution
3. **Global Average Pooling** revolutionized classification heads, reducing parameters by 100×
4. **Adaptive pooling** enables variable input sizes with fixed output
5. **Strided convolutions** offer learnable alternatives with denser gradients
6. **Output size formula**: $H_{out} = \lfloor(H_{in} + 2p - d(k-1) - 1) / s\rfloor + 1$

**The trend:** Modern architectures use less pooling and more strided convolutions, but GAP remains standard for classification heads.

## References

1. Boureau, Y.-L., et al. (2010). "A Theoretical Analysis of Feature Pooling in Visual Recognition." *ICML*.
2. Lin, M., Chen, Q., & Yan, S. (2014). "Network In Network." *ICLR*. (Introduced GAP)
3. He, K., et al. (2015). "Spatial Pyramid Pooling in Deep Convolutional Networks." *IEEE TPAMI*.
4. Springenberg, J.T., et al. (2015). "Striving for Simplicity: The All Convolutional Net." *ICLR Workshop*.
5. Scherer, D., Müller, A., & Behnke, S. (2010). "Evaluation of Pooling Operations in Convolutional Architectures." *ICANN*.
