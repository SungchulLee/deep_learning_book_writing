# Pooling Layers

## Introduction

**Pooling layers** are fundamental components of CNNs that perform spatial downsampling. They serve several critical purposes:

1. **Dimensionality reduction**: Decrease spatial dimensions, reducing computation
2. **Translation invariance**: Make features more robust to small spatial shifts
3. **Increased receptive field**: Allow subsequent layers to see more of the input
4. **Regularization**: Reduce overfitting by providing a form of spatial regularization

This section provides a comprehensive treatment of pooling operations, their mathematical properties, and practical implementations.

## Types of Pooling

### Max Pooling

**Max pooling** selects the maximum value within each pooling window:

$$y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}$$

where $R_{i,j}$ is the pooling region for output position $(i, j)$.

**Properties:**

- Preserves strongest activations
- Provides local translation invariance
- Non-differentiable at points where maximum switches

**Visual Example (2×2 max pooling):**

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

### Average Pooling

**Average pooling** computes the mean within each pooling window:

$$y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} x_{m,n}$$

**Properties:**

- Preserves overall activation level
- Smoother gradients than max pooling
- Better for dense prediction tasks

### Global Average Pooling (GAP)

**Global average pooling** reduces each channel to a single value:

$$y_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{c,i,j}$$

**Properties:**

- Eliminates need for fully connected layers
- Reduces parameters significantly
- Enforces correspondence between feature maps and categories
- Popularized by Network in Network (NIN) and GoogLeNet

### Global Max Pooling

Similar to GAP but takes the maximum:

$$y_c = \max_{i,j} x_{c,i,j}$$

## Mathematical Analysis

### Output Dimensions

For input size $H \times W$, pooling kernel $K \times K$, stride $s$, and padding $p$:

$$H_{out} = \left\lfloor \frac{H + 2p - K}{s} \right\rfloor + 1$$

Most common configuration: $K = 2$, $s = 2$, $p = 0$, which halves spatial dimensions.

### Gradient Computation

#### Max Pooling Gradient

The gradient flows only through the maximum element:

$$\frac{\partial y_{i,j}}{\partial x_{m,n}} = \begin{cases}
1 & \text{if } (m,n) = \arg\max_{(m',n') \in R_{i,j}} x_{m',n'} \\
0 & \text{otherwise}
\end{cases}$$

**Implementation note:** Must track indices of maximum elements during forward pass.

#### Average Pooling Gradient

The gradient is distributed equally:

$$\frac{\partial y_{i,j}}{\partial x_{m,n}} = \frac{1}{|R_{i,j}|}$$

for all $(m,n) \in R_{i,j}$.

## PyTorch Implementation

### Basic Pooling Operations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create sample input: (batch, channels, height, width)
x = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
print("Input:")
print(x.squeeze())

# Max Pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
y_max = max_pool(x)
print("\nMax Pooling (2x2, stride=2):")
print(y_max.squeeze())

# Average Pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
y_avg = avg_pool(x)
print("\nAverage Pooling (2x2, stride=2):")
print(y_avg.squeeze())

# Global Average Pooling
gap = nn.AdaptiveAvgPool2d(1)
y_gap = gap(x)
print("\nGlobal Average Pooling:")
print(y_gap.squeeze())

# Global Max Pooling
gmp = nn.AdaptiveMaxPool2d(1)
y_gmp = gmp(x)
print("\nGlobal Max Pooling:")
print(y_gmp.squeeze())
```

### Max Pooling with Index Tracking

```python
# MaxPool2d can return indices for unpooling
max_pool_idx = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

x = torch.randn(1, 1, 4, 4)
y, indices = max_pool_idx(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Indices shape:", indices.shape)
print("\nIndices (flat indices of max elements):")
print(indices)

# Unpooling using indices
unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
x_reconstructed = unpool(y, indices)
print("\nReconstructed shape:", x_reconstructed.shape)
```

### Overlapping Pooling

```python
# Overlapping pooling: kernel > stride
# Used in AlexNet (3x3 kernel, stride 2)
overlap_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

x = torch.randn(1, 64, 56, 56)
y = overlap_pool(x)
print(f"Overlapping pooling: {x.shape} → {y.shape}")
# Output: [1, 64, 28, 28]
```

### Adaptive Pooling

```python
# Adaptive pooling: specify output size, not kernel size
# Useful when input sizes vary

adaptive_avg = nn.AdaptiveAvgPool2d(output_size=(7, 7))
adaptive_max = nn.AdaptiveMaxPool2d(output_size=(7, 7))

# Works with any input size
for size in [224, 299, 384, 512]:
    x = torch.randn(1, 512, size // 32, size // 32)
    y = adaptive_avg(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
```

## Manual Implementation

### Max Pooling from Scratch

```python
def max_pool2d_manual(x, kernel_size, stride=None, padding=0):
    """
    Manual implementation of 2D max pooling.
    
    Args:
        x: Input tensor (batch, channels, H, W)
        kernel_size: Pooling window size
        stride: Stride (default: kernel_size)
        padding: Zero padding
    
    Returns:
        Pooled output and indices
    """
    if stride is None:
        stride = kernel_size
    
    batch, channels, H, W = x.shape
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    
    # Apply padding
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding), value=float('-inf'))
    
    _, _, H_padded, W_padded = x.shape
    
    # Output dimensions
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1
    
    # Initialize output and indices
    output = torch.zeros(batch, channels, H_out, W_out, device=x.device)
    indices = torch.zeros(batch, channels, H_out, W_out, dtype=torch.long, device=x.device)
    
    for b in range(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    # Extract pooling region
                    h_start = i * sH
                    w_start = j * sW
                    region = x[b, c, h_start:h_start+kH, w_start:w_start+kW]
                    
                    # Find maximum and its index
                    max_val = region.max()
                    max_idx = region.argmax()
                    
                    output[b, c, i, j] = max_val
                    # Convert to flat index in original tensor
                    local_h, local_w = max_idx // kW, max_idx % kW
                    indices[b, c, i, j] = (h_start + local_h) * W_padded + (w_start + local_w)
    
    return output, indices

# Test manual implementation
x = torch.randn(2, 3, 8, 8)

# Manual
y_manual, idx_manual = max_pool2d_manual(x, kernel_size=2, stride=2)

# PyTorch
pool = nn.MaxPool2d(2, 2, return_indices=True)
y_pytorch, idx_pytorch = pool(x)

print(f"Output match: {torch.allclose(y_manual, y_pytorch)}")
```

### Average Pooling from Scratch

```python
def avg_pool2d_manual(x, kernel_size, stride=None, padding=0, count_include_pad=True):
    """
    Manual implementation of 2D average pooling.
    
    Args:
        x: Input tensor (batch, channels, H, W)
        kernel_size: Pooling window size
        stride: Stride (default: kernel_size)
        padding: Zero padding
        count_include_pad: Include padding in averaging denominator
    
    Returns:
        Pooled output
    """
    if stride is None:
        stride = kernel_size
    
    batch, channels, H, W = x.shape
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    
    # Apply padding
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding), value=0)
    
    _, _, H_padded, W_padded = x.shape
    
    # Output dimensions
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1
    
    # Initialize output
    output = torch.zeros(batch, channels, H_out, W_out, device=x.device)
    
    for b in range(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    # Extract pooling region
                    h_start = i * sH
                    w_start = j * sW
                    region = x[b, c, h_start:h_start+kH, w_start:w_start+kW]
                    
                    # Compute average
                    if count_include_pad:
                        output[b, c, i, j] = region.mean()
                    else:
                        # Count only non-padding elements
                        # (simplified: would need to track actual padding)
                        output[b, c, i, j] = region.sum() / (kH * kW)
    
    return output

# Test manual implementation
x = torch.randn(2, 3, 8, 8)

y_manual = avg_pool2d_manual(x, kernel_size=2, stride=2)
y_pytorch = F.avg_pool2d(x, kernel_size=2, stride=2)

print(f"Output match: {torch.allclose(y_manual, y_pytorch)}")
```

## Advanced Pooling Methods

### Fractional Max Pooling

**Fractional max pooling** uses randomized pooling regions for data augmentation:

```python
# Fractional pooling: non-integer reduction ratios
frac_pool = nn.FractionalMaxPool2d(
    kernel_size=2,
    output_ratio=(0.7, 0.7),  # Output is ~70% of input size
    return_indices=True
)

x = torch.randn(1, 64, 32, 32)
y, indices = frac_pool(x)
print(f"Fractional pooling: {x.shape} → {y.shape}")  # Approximately 22x22
```

### LP Pooling

**LP pooling** generalizes average (L1) and max (L∞) pooling:

$$y_{i,j} = \left( \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} |x_{m,n}|^p \right)^{1/p}$$

```python
# LP Pooling (p=2 gives L2 norm pooling)
lp_pool = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2)

x = torch.randn(1, 64, 8, 8)
y = lp_pool(x)
print(f"LP pooling: {x.shape} → {y.shape}")
```

### Spatial Pyramid Pooling (SPP)

**SPP** pools at multiple scales and concatenates:

```python
class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling module.
    Enables CNNs to handle variable input sizes.
    """
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels
    
    def forward(self, x):
        batch, channels, H, W = x.shape
        pooled = []
        
        for level in self.levels:
            # Adaptive pooling to level x level grid
            pool = nn.AdaptiveMaxPool2d(output_size=(level, level))
            pooled_level = pool(x)  # (batch, channels, level, level)
            pooled_level = pooled_level.view(batch, -1)  # Flatten
            pooled.append(pooled_level)
        
        # Concatenate all levels
        return torch.cat(pooled, dim=1)

spp = SpatialPyramidPooling(levels=[1, 2, 4])

# Works with any input size!
for size in [7, 14, 28]:
    x = torch.randn(2, 256, size, size)
    y = spp(x)
    print(f"SPP input: {x.shape} → output: {y.shape}")
# All produce same output size: (2, 256*(1+4+16)) = (2, 5376)
```

### Region of Interest (RoI) Pooling

**RoI pooling** extracts fixed-size features from arbitrary regions:

```python
from torchvision.ops import roi_pool, roi_align

# Feature map
features = torch.randn(1, 256, 14, 14)

# Regions of interest: (batch_index, x1, y1, x2, y2)
# Coordinates in feature map scale
rois = torch.tensor([
    [0, 0, 0, 7, 7],      # Top-left quadrant
    [0, 7, 0, 14, 7],     # Top-right quadrant
    [0, 3, 3, 10, 10],    # Center region
], dtype=torch.float32)

# RoI Pool: quantizes coordinates (may lose precision)
pooled_roi = roi_pool(features, rois, output_size=(7, 7), spatial_scale=1.0)
print(f"RoI Pool output: {pooled_roi.shape}")  # (3, 256, 7, 7)

# RoI Align: uses bilinear interpolation (more precise)
aligned_roi = roi_align(features, rois, output_size=(7, 7), spatial_scale=1.0)
print(f"RoI Align output: {aligned_roi.shape}")  # (3, 256, 7, 7)
```

## Pooling vs. Strided Convolution

Modern architectures often replace pooling with strided convolutions:

### Comparison

| Aspect | Pooling | Strided Convolution |
|--------|---------|---------------------|
| Parameters | 0 | $K^2 \times C_{in} \times C_{out}$ |
| Learnable | No | Yes |
| Information preservation | Fixed selection | Learned aggregation |
| Gradient flow | Sparse (max) or uniform (avg) | Dense |
| Common use | Older architectures | Modern architectures |

### Example: ResNet-style Downsampling

```python
class ResNetDownsample(nn.Module):
    """
    ResNet-style downsampling: strided conv instead of pooling.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Strided 1x1 convolution for downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.downsample(x)

# Compare with max pooling
x = torch.randn(1, 64, 56, 56)

# Pooling approach
pool_down = nn.MaxPool2d(2, 2)
y_pool = pool_down(x)
print(f"Pooling: {x.shape} → {y_pool.shape}, params: 0")

# Strided conv approach
conv_down = ResNetDownsample(64, 128)
y_conv = conv_down(x)
params = sum(p.numel() for p in conv_down.parameters())
print(f"Strided conv: {x.shape} → {y_conv.shape}, params: {params}")
```

## Best Practices

### When to Use Max Pooling

1. **Image classification**: Preserves discriminative features
2. **Early layers**: More aggressive feature selection
3. **When translation invariance is desired**

### When to Use Average Pooling

1. **Global pooling before classifier**: GAP is standard
2. **Semantic segmentation**: Preserves spatial structure
3. **When all activations contribute equally**

### When to Use Strided Convolution

1. **Modern architectures**: ResNet, DenseNet, EfficientNet
2. **When learnable downsampling is beneficial**
3. **Generative models**: Better gradient flow

### Common Patterns

```python
# Classic CNN pattern (VGG-style)
classic = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Downsample with pooling
)

# Modern pattern (ResNet-style)
modern = nn.Sequential(
    nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Downsample with strided conv
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
)

# Classification head pattern
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
    nn.Flatten(),
    nn.Linear(512, num_classes),
)
```

## Summary

Key points about pooling layers:

1. **Max pooling** selects strongest activations, providing translation invariance
2. **Average pooling** preserves overall activation levels
3. **Global pooling** reduces spatial dimensions to 1×1, eliminating FC layers
4. **Adaptive pooling** handles variable input sizes
5. **Strided convolutions** offer learnable alternatives to fixed pooling

Pooling design considerations:

- Use GAP before classification heads (modern standard)
- Consider strided convolutions for downsampling
- Match pooling to task requirements (classification vs. segmentation)
- Overlapping pooling can improve performance marginally

## Exercises

1. **Gradient Flow Analysis**: Implement max pooling with gradient computation from scratch. Visualize which input positions receive gradients.

2. **Pooling Comparison**: Train the same architecture with (a) max pooling, (b) average pooling, (c) strided convolution for downsampling. Compare accuracy and training dynamics on CIFAR-10.

3. **SPP Implementation**: Implement SPP and demonstrate that it allows a single model to handle multiple input sizes.

4. **RoI Align**: Implement RoI Align from scratch using bilinear interpolation.

5. **Fractional Pooling**: Study how fractional max pooling affects model performance as a data augmentation technique.

## References

1. Scherer, D., Müller, A., & Behnke, S. (2010). Evaluation of pooling operations in convolutional architectures for object recognition. *ICANN 2010*.

2. Lin, M., Chen, Q., & Yan, S. (2014). Network in network. *ICLR 2014*.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial pyramid pooling in deep convolutional networks for visual recognition. *IEEE TPAMI*.

4. Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2015). Striving for simplicity: The all convolutional net. *ICLR Workshop 2015*.
