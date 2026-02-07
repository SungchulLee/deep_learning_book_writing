# Padding and Stride

## Introduction

The basic convolution operation, while powerful, has limitations: it shrinks the spatial dimensions of the output, treats edge pixels differently from center pixels, and always moves one pixel at a time. **Padding** and **stride** are two modifications to the convolution operation that provide control over the output dimensions and downsampling behavior.

Understanding these parameters is essential for designing CNN architectures that can:
- Preserve spatial information through deep networks
- Control downsampling precisely
- Balance computational cost with feature resolution

---

## Padding

### Motivation

Without padding, a convolution with a $K \times K$ kernel reduces the spatial dimensions:

$$H_{out} = H_{in} - K + 1, \quad W_{out} = W_{in} - K + 1$$

This creates two fundamental problems:

1. **Spatial shrinkage**: After several layers, the feature map becomes very small
2. **Border information loss**: Pixels near the edges contribute to fewer output values than central pixels

### Types of Padding

#### No Padding (Valid Convolution)

Output size shrinks with each convolution:

```
Input (5×5):                    Kernel (3×3):         Output (3×3):
┌─────────────────────┐         ┌─────────────┐       ┌─────────────┐
│  ×   ×   ×   ×   ×  │         │  *   *   *  │       │  o   o   o  │
│  ×   ×   ×   ×   ×  │    *    │  *   *   *  │   =   │  o   o   o  │
│  ×   ×   ×   ×   ×  │         │  *   *   *  │       │  o   o   o  │
│  ×   ×   ×   ×   ×  │         └─────────────┘       └─────────────┘
│  ×   ×   ×   ×   ×  │
└─────────────────────┘
```

#### Same Padding (Zero Padding)

Pad to keep output size equal to input size (when stride=1):

$$p = \frac{K - 1}{2}$$

For a $3 \times 3$ kernel: $p = 1$. For a $5 \times 5$ kernel: $p = 2$.

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

#### Full Padding

Pad enough so every input element is visited by every kernel position:

$$p = K - 1$$

Output size: $H_{out} = H_{in} + K - 1$

### Padding Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Zeros | Pad with zeros | Default, most common |
| Reflect | Mirror values at boundary | Image processing, avoids artifacts |
| Replicate | Repeat edge values | Natural image boundaries |
| Circular | Wrap around | Periodic signals, tiling |

---

## Deep Insight: The Border Effect Problem

### Unequal Contribution of Pixels

In a valid convolution, different input pixels contribute to different numbers of output pixels:

```
5×5 input, 3×3 kernel (valid):

Contribution count for each input pixel:
┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 2 │ 1 │   Corner pixels: contribute to 1 output
├───┼───┼───┼───┼───┤   Edge pixels: contribute to 2-3 outputs
│ 2 │ 4 │ 6 │ 4 │ 2 │   Center pixels: contribute to 9 outputs
├───┼───┼───┼───┼───┤
│ 3 │ 6 │ 9 │ 6 │ 3 │   
├───┼───┼───┼───┼───┤   
│ 2 │ 4 │ 6 │ 4 │ 2 │
├───┼───┼───┼───┼───┤
│ 1 │ 2 │ 3 │ 2 │ 1 │
└───┴───┴───┴───┴───┘
```

This creates a **systematic bias** where:
- Edge information is underrepresented in the learned features
- Gradients during backpropagation are weaker at boundaries
- The network "learns" that edges are less important

### How Padding Solves This

With same padding, corner pixels participate in more convolutions by being "surrounded" by padding values:

```
With p=1 padding:
┌───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 0 │
├───┼───┼───┼───┼───┼───┼───┤
│ 0 │...│   │   │   │...│ 0 │
...
```

Now the corner pixel at position (1,1) can be the center of a kernel placement, not just the corner.

### The Zero Artifact Problem

Zero padding has a subtle issue: **the network can detect padding**. 

Consider a 3×3 kernel learning to detect edges. At the boundary:
- Real edge: high contrast between actual values
- Padded boundary: contrast between actual values and zeros

This can cause:
1. Spurious edge detections at image boundaries
2. Different learned features for boundary vs. interior
3. Checkerboard artifacts in generative models

**Solution**: Reflection or replication padding creates more natural boundaries:

```python
import torch.nn.functional as F

# Zero padding: creates artificial edges
F.pad(x, (1,1,1,1), mode='constant', value=0)

# Reflect padding: mirrors actual content
F.pad(x, (1,1,1,1), mode='reflect')  

# Replicate padding: extends edge values
F.pad(x, (1,1,1,1), mode='replicate')
```

---

## Stride

### Concept

Stride determines how many positions the kernel moves between applications:

- **Stride 1**: Kernel moves one position at a time (dense output)
- **Stride 2**: Kernel moves two positions (output size halved)
- **Stride > 1**: Downsamples the feature map

### Mathematical Definition

With stride $s$ and padding $p$, the output dimensions become:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - K}{s} \right\rfloor + 1$$

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

---

## Deep Insight: Strided Convolution vs Pooling

### The Downsampling Decision

Both strided convolutions and pooling reduce spatial dimensions, but they work fundamentally differently:

| Aspect | Strided Convolution | Max Pooling | Average Pooling |
|--------|---------------------|-------------|-----------------|
| Parameters | Has learnable weights | None | None |
| Operation | Weighted sum | Take maximum | Take average |
| Information | Learned combination | Keeps strongest signal | Blurs/smooths |
| Gradient flow | Through all inputs | Only through max | Through all inputs |
| Aliasing | Can learn anti-aliasing | Susceptible | Acts as low-pass |

### Why Modern Architectures Prefer Strided Convolutions

1. **Learnable downsampling**: The network can learn *what* to preserve during downsampling, not just blindly take the max or average.

2. **Combined operation**: Strided conv = feature extraction + downsampling in one step, vs. pooling requiring a separate conv layer.

3. **Better gradient flow**: Max pooling creates "winner-take-all" gradients where only the max-contributing input receives gradient.

```python
import torch.nn as nn

# Traditional approach: conv + pool
traditional = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),  # [B, 128, H, W]
    nn.MaxPool2d(2, 2)                  # [B, 128, H/2, W/2]
)

# Modern approach: strided conv
modern = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # [B, 128, H/2, W/2]
```

### The Aliasing Problem

When downsampling, high-frequency content above the Nyquist frequency causes aliasing (the "wagon wheel effect"). 

- **Max pooling**: Directly samples, prone to aliasing
- **Average pooling**: Natural low-pass filter, reduces aliasing
- **Strided conv**: Can learn to be a low-pass filter if beneficial

Recent work (like "Making Convolutional Networks Shift-Invariant Again") shows that adding explicit anti-aliasing before any strided operation improves shift invariance, classification accuracy, and robustness to input perturbations.

---

## Complete Output Size Formula

Combining padding and stride (and dilation $d$ for completeness):

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K - 1) - 1}{s} \right\rfloor + 1$$

where:
- $H_{in}$: Input height
- $p$: Padding
- $K$: Kernel size
- $d$: Dilation (default 1; see [Dilated Convolutions](dilated_convolutions.md))
- $s$: Stride

### Python Implementation

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
print(f"Input=224, K=3, p=1, s=1: {conv_output_size(224, 3, 1, 1)}")  # 224
print(f"Input=224, K=3, p=1, s=2: {conv_output_size(224, 3, 1, 2)}")  # 112
print(f"Input=224, K=7, p=3, s=2: {conv_output_size(224, 7, 3, 2)}")  # 112
```

---

## PyTorch Implementation

### Padding Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input tensor
x = torch.tensor([[[[1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                    [9., 10., 11., 12.],
                    [4., 3., 2., 1.]]]])

# Zero padding (default)
x_zeros = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)

# Reflection padding (mirrors at boundary)
x_reflect = F.pad(x, (1, 1, 1, 1), mode='reflect')

# Replication padding (repeats edge values)
x_replicate = F.pad(x, (1, 1, 1, 1), mode='replicate')

# Circular padding (wraps around)
x_circular = F.pad(x, (1, 1, 1, 1), mode='circular')
```

### Conv2d with Different Parameters

```python
# Padding options
conv_valid = nn.Conv2d(3, 64, kernel_size=3, padding=0)       # Valid
conv_same = nn.Conv2d(3, 64, kernel_size=3, padding=1)        # Same
conv_auto = nn.Conv2d(3, 64, kernel_size=3, padding='same')   # Automatic same
conv_reflect = nn.Conv2d(3, 64, kernel_size=3, padding=1, padding_mode='reflect')

# Stride options
conv_s1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # No downsampling
conv_s2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 2× downsampling
conv_s4 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3)  # 4× downsampling

# Test
x = torch.randn(1, 3, 224, 224)
print(f"Input: {x.shape}")
print(f"Stride 1: {conv_s1(x).shape}")  # [1, 64, 224, 224]
print(f"Stride 2: {conv_s2(x).shape}")  # [1, 64, 112, 112]
print(f"Stride 4: {conv_s4(x).shape}")  # [1, 64, 56, 56]
```

### Complete Comparison Example

```python
def analyze_conv(name, conv, input_shape):
    """Analyze convolution layer properties."""
    x = torch.randn(*input_shape)
    y = conv(x)
    
    # Count parameters
    params = sum(p.numel() for p in conv.parameters())
    
    print(f"{name}:")
    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(y.shape)}")
    print(f"  Parameters:   {params:,}")
    print()

# Input: 1 batch, 64 channels, 56×56
input_shape = (1, 64, 56, 56)

# Standard 3×3 conv
conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
analyze_conv("Standard 3×3", conv1, input_shape)

# Strided 3×3 conv (downsampling)
conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
analyze_conv("Strided 3×3 (s=2)", conv2, input_shape)
```

---

## Practical Guidelines

### Padding Selection

| Goal | Padding | Notes |
|------|---------|-------|
| Reduce dimensions | 0 (valid) | Used in some classification heads |
| Preserve dimensions | $(K-1)/2$ (same) | Most common in modern architectures |
| Handle edges better | Reflection or replication | For image-to-image tasks |

### Stride Selection

| Goal | Stride | Notes |
|------|--------|-------|
| Dense feature maps | 1 | Standard feature extraction |
| Downsample 2× | 2 | Preferred over pooling in modern nets |
| Aggressive downsampling | 4+ | Only in first layer (e.g., ResNet stem) |

### Architecture Design Tips

In ResNet-style architectures, the typical downsampling pattern is:
- **First layer**: 7×7, stride 2 (reduces by 2×)
- **Max pool**: 3×3, stride 2 (reduces by 2×)  
- **Later stages**: 3×3, stride 2 at stage transitions

---

## Summary

| Concept | Purpose | Effect on Output Size | Key Insight |
|---------|---------|----------------------|-------------|
| **Padding** | Preserve spatial dimensions, use edge information | Increases | Solves border effect, but zero padding creates detectable artifacts |
| **Stride** | Downsample, reduce computation | Decreases | Learnable downsampling (strided conv) superior to fixed pooling |

The interplay of these parameters gives CNNs flexibility in controlling output resolution, computational cost, and information preservation across different tasks.

---

## References

1. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.

3. Springenberg, J.T., et al. (2015). Striving for simplicity: The All Convolutional Net. *ICLR Workshop*.

4. Zhang, R. (2019). Making convolutional networks shift-invariant again. *ICML 2019*.
