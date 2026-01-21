# Padding, Stride, and Dilation

## Introduction

The basic convolution operation, while powerful, has limitations: it shrinks the spatial dimensions of the output, treats edge pixels differently from center pixels, and always moves one pixel at a time. **Padding**, **stride**, and **dilation** are three modifications to the convolution operation that provide greater control over the output dimensions and the receptive field of each output unit.

This section provides a comprehensive treatment of these concepts with mathematical derivations and practical PyTorch implementations.

## Padding

### Motivation

Without padding, a convolution with a $K \times K$ kernel reduces the spatial dimensions:

$$H_{out} = H_{in} - K + 1, \quad W_{out} = W_{in} - K + 1$$

This creates two problems:

1. **Spatial shrinkage**: After several layers, the feature map becomes very small
2. **Border information loss**: Pixels near the edges contribute to fewer output values than central pixels

### Types of Padding

#### Zero Padding

The most common approach is to add zeros around the input:

```
Original (3×3)        Zero Padded (5×5, p=1)
┌───┬───┬───┐        ┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │        │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┤        ├───┼───┼───┼───┼───┤
│ 4 │ 5 │ 6 │   →    │ 0 │ 1 │ 2 │ 3 │ 0 │
├───┼───┼───┤        ├───┼───┼───┼───┼───┤
│ 7 │ 8 │ 9 │        │ 0 │ 4 │ 5 │ 6 │ 0 │
└───┴───┴───┘        ├───┼───┼───┼───┼───┤
                     │ 0 │ 7 │ 8 │ 9 │ 0 │
                     ├───┼───┼───┼───┼───┤
                     │ 0 │ 0 │ 0 │ 0 │ 0 │
                     └───┴───┴───┴───┴───┘
```

#### Same Padding

Padding chosen to make output size equal to input size (when stride=1):

$$p = \frac{K - 1}{2}$$

For a $3 \times 3$ kernel: $p = 1$. For a $5 \times 5$ kernel: $p = 2$.

#### Valid Padding

No padding at all ($p = 0$). Output is smaller than input.

#### Other Padding Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Zeros | Pad with zeros | Default, most common |
| Reflect | Mirror values at boundary | Image processing |
| Replicate | Repeat edge values | Natural image boundaries |
| Circular | Wrap around | Periodic signals |

### Output Size with Padding

With padding $p$ on each side:

$$H_{out} = H_{in} + 2p - K + 1$$
$$W_{out} = W_{in} + 2p - K + 1$$

For **same padding** with odd kernel size $K$:

$$p = \frac{K - 1}{2} \implies H_{out} = H_{in}$$

### PyTorch Implementation of Padding

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input tensor: (batch, channels, height, width)
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.float32).view(1, 1, 3, 3)

print("Original shape:", x.shape)  # [1, 1, 3, 3]

# Zero padding
x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
print("Zero padded shape:", x_padded.shape)  # [1, 1, 5, 5]
print("Zero padded:\n", x_padded.squeeze())

# Reflect padding
x_reflect = F.pad(x, (1, 1, 1, 1), mode='reflect')
print("\nReflect padded:\n", x_reflect.squeeze())

# Replicate padding
x_replicate = F.pad(x, (1, 1, 1, 1), mode='replicate')
print("\nReplicate padded:\n", x_replicate.squeeze())

# Circular padding
x_circular = F.pad(x, (1, 1, 1, 1), mode='circular')
print("\nCircular padded:\n", x_circular.squeeze())
```

```python
# Convolution with padding in nn.Conv2d
conv_no_pad = nn.Conv2d(1, 1, kernel_size=3, padding=0)  # Valid
conv_same = nn.Conv2d(1, 1, kernel_size=3, padding=1)     # Same
conv_same_auto = nn.Conv2d(1, 1, kernel_size=3, padding='same')  # Automatic same

# Different padding modes
conv_reflect = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='reflect')
conv_replicate = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate')
conv_circular = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='circular')

x = torch.randn(1, 1, 8, 8)
print(f"Input: {x.shape}")
print(f"No padding output: {conv_no_pad(x).shape}")  # [1, 1, 6, 6]
print(f"Same padding output: {conv_same(x).shape}")  # [1, 1, 8, 8]
```

## Stride

### Motivation

In standard convolution, the kernel moves one pixel at a time. **Stride** controls how far the kernel moves between applications:

- **Stride 1**: Standard convolution, kernel moves 1 pixel
- **Stride 2**: Kernel moves 2 pixels, reducing output size by half
- **Stride > 1**: Downsampling without explicit pooling

### Mathematical Definition

With stride $s$, the output dimensions become:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - K}{s} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} + 2p - K}{s} \right\rfloor + 1$$

The floor function $\lfloor \cdot \rfloor$ handles cases where the kernel doesn't fit perfectly.

### Visual Example

```
Input (6×6), Kernel (3×3), Stride=2

Position 1:                Position 2:
┌───┬───┬───┬───┬───┬───┐  ┌───┬───┬───┬───┬───┬───┐
│[●]│[●]│[●]│   │   │   │  │   │   │[●]│[●]│[●]│   │
├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┤
│[●]│[●]│[●]│   │   │   │  │   │   │[●]│[●]│[●]│   │
├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┤
│[●]│[●]│[●]│   │   │   │  │   │   │[●]│[●]│[●]│   │
├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │  │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │  │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │  │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┘  └───┴───┴───┴───┴───┴───┘

Output: (6-3)/2 + 1 = 2, so output is 2×2
```

### Stride for Downsampling

Stride > 1 is commonly used as an alternative to pooling for spatial reduction:

| Method | Pros | Cons |
|--------|------|------|
| Pooling | No parameters, keeps more info | Fixed operation |
| Strided Conv | Learnable downsampling | More parameters |

Modern architectures (ResNet, etc.) often prefer strided convolutions over pooling.

### PyTorch Implementation of Stride

```python
import torch
import torch.nn as nn

# Different stride configurations
conv_s1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # No downsampling
conv_s2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 2x downsampling
conv_s4 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3)  # 4x downsampling

x = torch.randn(1, 3, 224, 224)

print(f"Input: {x.shape}")                    # [1, 3, 224, 224]
print(f"Stride 1: {conv_s1(x).shape}")        # [1, 64, 224, 224]
print(f"Stride 2: {conv_s2(x).shape}")        # [1, 64, 112, 112]
print(f"Stride 4: {conv_s4(x).shape}")        # [1, 64, 56, 56]
```

```python
# Asymmetric stride (different for height and width)
conv_asymmetric = nn.Conv2d(3, 64, kernel_size=3, stride=(2, 1), padding=1)
print(f"Asymmetric stride: {conv_asymmetric(x).shape}")  # [1, 64, 112, 224]
```

## Dilation (Atrous Convolution)

### Motivation

Standard convolution has a limited receptive field determined by the kernel size. To capture larger context, we can either:

1. Use larger kernels (more parameters)
2. Stack more layers (more computation)
3. Use **dilated convolution** (same parameters, larger receptive field)

### Mathematical Definition

Dilation inserts "holes" (zeros) between kernel elements. With dilation rate $d$, the **effective kernel size** becomes:

$$K_{eff} = K + (K - 1)(d - 1) = d(K - 1) + 1$$

The output size formula generalizes to:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K-1) - 1}{s} \right\rfloor + 1$$

### Visual Example

```
Standard (d=1)      Dilated (d=2)       Dilated (d=3)
K=3, K_eff=3        K=3, K_eff=5        K=3, K_eff=7

[●][●][●]           [●][ ][●][ ][●]     [●][ ][ ][●][ ][ ][●]
[●][●][●]           [ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ][ ][ ]
[●][●][●]           [●][ ][●][ ][●]     [ ][ ][ ][ ][ ][ ][ ]
                    [ ][ ][ ][ ][ ]     [●][ ][ ][●][ ][ ][●]
                    [●][ ][●][ ][●]     [ ][ ][ ][ ][ ][ ][ ]
                                        [ ][ ][ ][ ][ ][ ][ ]
                                        [●][ ][ ][●][ ][ ][●]

[●] = kernel weight position
[ ] = skipped (implicit zero)
```

### Properties of Dilated Convolution

1. **Exponentially growing receptive field**: With stacked dilated convolutions using rates 1, 2, 4, 8, the receptive field grows exponentially while parameters grow linearly.

2. **Dense prediction**: Unlike strided convolutions, dilated convolutions can maintain spatial resolution while capturing global context.

3. **No information loss**: Unlike pooling, all input information is preserved.

### PyTorch Implementation of Dilation

```python
import torch
import torch.nn as nn

# Dilated convolutions
conv_d1 = nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1)   # Standard
conv_d2 = nn.Conv2d(3, 64, kernel_size=3, dilation=2, padding=2)   # Dilation 2
conv_d4 = nn.Conv2d(3, 64, kernel_size=3, dilation=4, padding=4)   # Dilation 4
conv_d8 = nn.Conv2d(3, 64, kernel_size=3, dilation=8, padding=8)   # Dilation 8

x = torch.randn(1, 3, 64, 64)

# All produce same output size with appropriate padding
print(f"Input: {x.shape}")
print(f"Dilation 1: {conv_d1(x).shape}")  # [1, 64, 64, 64]
print(f"Dilation 2: {conv_d2(x).shape}")  # [1, 64, 64, 64]
print(f"Dilation 4: {conv_d4(x).shape}")  # [1, 64, 64, 64]
print(f"Dilation 8: {conv_d8(x).shape}")  # [1, 64, 64, 64]

# Effective kernel sizes
print(f"\nEffective kernel sizes:")
print(f"d=1: {1*(3-1)+1} = 3×3")
print(f"d=2: {2*(3-1)+1} = 5×5")
print(f"d=4: {4*(3-1)+1} = 9×9")
print(f"d=8: {8*(3-1)+1} = 17×17")

# All have same number of parameters!
for name, conv in [("d=1", conv_d1), ("d=2", conv_d2), 
                    ("d=4", conv_d4), ("d=8", conv_d8)]:
    params = sum(p.numel() for p in conv.parameters())
    print(f"{name}: {params} parameters")
```

### Multi-Scale Feature Extraction with Dilation

```python
class AtrousSpatialPyramidPooling(nn.Module):
    """
    ASPP module from DeepLab for multi-scale feature extraction.
    Uses parallel dilated convolutions with different rates.
    """
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        
        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolutions at different rates
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 
                         padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for rate in rates
        ])
        
        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 convolution to combine features
        num_features = 1 + len(rates) + 1  # 1x1 + dilated convs + global pool
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_features, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Apply all branches
        features = [self.conv1x1(x)]
        features += [conv(x) for conv in self.dilated_convs]
        
        # Global pooling branch (upsampled to match size)
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', 
                                   align_corners=False)
        features.append(global_feat)
        
        # Concatenate and project
        x = torch.cat(features, dim=1)
        x = self.project(x)
        
        return x

# Example usage
aspp = AtrousSpatialPyramidPooling(in_channels=256, out_channels=256)
x = torch.randn(2, 256, 32, 32)
out = aspp(x)
print(f"ASPP input: {x.shape}, output: {out.shape}")
```

## Complete Output Size Formula

Combining all three parameters, the general formula for output dimensions is:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K_H - 1) - 1}{s} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W_{in} + 2p - d(K_W - 1) - 1}{s} \right\rfloor + 1$$

where:

- $H_{in}, W_{in}$: Input dimensions
- $K_H, K_W$: Kernel dimensions
- $p$: Padding
- $s$: Stride
- $d$: Dilation

### Output Size Calculator

```python
def conv_output_size(input_size, kernel_size, padding=0, stride=1, dilation=1):
    """
    Calculate output size of a convolution operation.
    
    Args:
        input_size: Input spatial dimension (H or W)
        kernel_size: Kernel size
        padding: Padding amount
        stride: Stride
        dilation: Dilation rate
    
    Returns:
        Output spatial dimension
    """
    effective_kernel = dilation * (kernel_size - 1) + 1
    output = (input_size + 2 * padding - effective_kernel) // stride + 1
    return output

# Examples
print("Conv2d output sizes:")
print(f"Input=224, K=3, p=1, s=1, d=1: {conv_output_size(224, 3, 1, 1, 1)}")  # 224
print(f"Input=224, K=3, p=1, s=2, d=1: {conv_output_size(224, 3, 1, 2, 1)}")  # 112
print(f"Input=224, K=7, p=3, s=2, d=1: {conv_output_size(224, 7, 3, 2, 1)}")  # 112
print(f"Input=56, K=3, p=2, s=1, d=2: {conv_output_size(56, 3, 2, 1, 2)}")    # 56
```

## Common Configurations

### Preserve Spatial Size (stride=1)

| Kernel | Dilation | Padding | Output |
|--------|----------|---------|--------|
| 3×3 | 1 | 1 | Same |
| 5×5 | 1 | 2 | Same |
| 7×7 | 1 | 3 | Same |
| 3×3 | 2 | 2 | Same |
| 3×3 | 4 | 4 | Same |

### Halve Spatial Size (stride=2)

| Kernel | Dilation | Padding | Input | Output |
|--------|----------|---------|-------|--------|
| 3×3 | 1 | 1 | 224 | 112 |
| 4×4 | 1 | 1 | 224 | 112 |
| 7×7 | 1 | 3 | 224 | 112 |

## Practical Guidelines

### Choosing Padding

1. **Same padding** ($p = (K-1)/2$ for odd $K$) is most common in modern architectures
2. **Valid padding** ($p = 0$) is used when exact output size doesn't matter
3. Use `padding='same'` in PyTorch for automatic calculation

### Choosing Stride

1. **Stride 1** for feature extraction layers
2. **Stride 2** for downsampling (alternative to pooling)
3. Modern architectures prefer strided convolutions over max pooling

### Choosing Dilation

1. **Dilation 1** (standard) for most layers
2. **Increasing dilation** (1, 2, 4, 8) for dense prediction tasks
3. **ASPP-style** parallel dilations for multi-scale features

!!! tip "Architecture Design Tip"
    In ResNet-style architectures, strided convolutions are used for downsampling. The typical pattern is:
    
    - First layer: 7×7, stride 2 (reduces by 2×)
    - Max pool: 3×3, stride 2 (reduces by 2×)
    - Later stages: 3×3, stride 2 at stage transitions

## Summary

| Concept | Purpose | Effect on Output Size |
|---------|---------|----------------------|
| Padding | Preserve spatial dimensions, use edge information | Increases |
| Stride | Downsample, reduce computation | Decreases |
| Dilation | Increase receptive field without more parameters | Depends on padding |

The interplay of these three parameters gives CNNs remarkable flexibility in controlling:

- **Output resolution**: Critical for tasks requiring different output sizes
- **Receptive field**: How much context each output pixel "sees"
- **Computational cost**: Larger strides and dilations can reduce computation

## Exercises

1. **Output Size Calculation**: Given an input of 512×512 with 3 channels, calculate the output size after these sequential operations:
   - Conv(3→64, K=7, s=2, p=3)
   - MaxPool(K=3, s=2, p=1)
   - Conv(64→128, K=3, s=2, p=1)
   - Conv(128→256, K=3, s=1, p=1, d=2)

2. **Same Padding**: Implement a function that calculates the required padding to maintain spatial dimensions for any kernel size and dilation.

3. **Dilated Convolution Visualization**: Create a visualization showing how different dilation rates affect the receptive field of a 3×3 kernel.

4. **Architecture Analysis**: Analyze the stride and padding choices in ResNet-50. How does the spatial resolution change through the network?

## References

1. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR 2016*.

2. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE TPAMI*.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
