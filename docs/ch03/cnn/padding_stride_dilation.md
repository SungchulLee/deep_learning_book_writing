# Padding, Stride, and Dilation

## Introduction

The basic convolution operation, while powerful, has limitations: it shrinks the spatial dimensions of the output, treats edge pixels differently from center pixels, and always moves one pixel at a time. **Padding**, **stride**, and **dilation** are three modifications to the convolution operation that provide greater control over the output dimensions and the receptive field of each output unit.

Understanding these parameters is essential for designing CNN architectures that can:
- Preserve spatial information through deep networks
- Control downsampling precisely
- Capture multi-scale context efficiently

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

With stride $s$, the output dimensions become:

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

Recent work (like "Making Convolutional Networks Shift-Invariant Again") shows that adding explicit anti-aliasing before any strided operation improves:
- Shift invariance
- Classification accuracy
- Robustness to input perturbations

---

## Dilation (Atrous Convolution)

### Motivation

Standard convolution has a limited receptive field determined by the kernel size. To capture larger context, we can either:

1. Use larger kernels (more parameters: $O(K^2)$)
2. Stack more layers (more computation, deeper gradients)
3. Use **dilated convolution** (same parameters, larger receptive field)

### Mathematical Definition

Dilation inserts "holes" (zeros) between kernel elements. With dilation rate $d$, the **effective kernel size** becomes:

$$K_{eff} = K + (K - 1)(d - 1) = d(K - 1) + 1$$

For $K=3$:
- $d=1$: $K_{eff} = 3$
- $d=2$: $K_{eff} = 5$
- $d=4$: $K_{eff} = 9$
- $d=8$: $K_{eff} = 17$

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

### Output Size Formula

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K - 1) - 1}{s} \right\rfloor + 1$$

---

## Deep Insight: The Receptive Field Trade-off

### Receptive Field Growth

The receptive field is how much of the input each output pixel "sees." Different approaches grow it at different rates:

| Method | Receptive Field Growth | Parameter Growth |
|--------|------------------------|------------------|
| Larger kernels | Linear with K | Quadratic: $O(K^2)$ |
| Deeper networks | Linear with depth | Linear with depth |
| Dilated convolutions | Exponential with dilation stack | Constant |

### Exponential Growth with Stacked Dilations

Stack dilated convolutions with rates 1, 2, 4, 8:

```
Layer 1 (d=1): RF = 3
Layer 2 (d=2): RF = 3 + 2×2 = 7  
Layer 3 (d=4): RF = 7 + 2×4 = 15
Layer 4 (d=8): RF = 15 + 2×8 = 31
```

With just 4 layers and 9 parameters each (36 total), we achieve a 31×31 receptive field. A single standard conv would need a 31×31 kernel with 961 parameters!

### The Gridding Problem

Stacking dilated convolutions with the same rate causes a "gridding" artifact:

```
Dilation=2, two layers:

Layer 1 samples:    Layer 2 samples:    Combined coverage:
[●][ ][●][ ][●]     [●][ ][●][ ][●]     [●][ ][●][ ][●]
[ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]
[●][ ][●][ ][●]  →  [●][ ][●][ ][●]  =  [●][ ][●][ ][●]
[ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]
[●][ ][●][ ][●]     [●][ ][●][ ][●]     [●][ ][●][ ][●]

Problem: The [ ] positions are NEVER sampled!
```

**Solution**: Use dilations that are not multiples of each other, or use a "sawtooth" pattern:
- Good: 1, 2, 5, 1, 2, 5 (HDC pattern from "Understanding Convolution for Semantic Segmentation")
- Good: 1, 2, 4, 8 then repeat
- Bad: 2, 2, 2, 2 (gridding)

---

## Deep Insight: Dilation for Dense Prediction

### The Segmentation Challenge

For semantic segmentation, we need:
1. **Large receptive field**: To understand context (is this pixel part of a cat or a dog?)
2. **High resolution output**: To preserve fine boundaries

Standard CNNs face a dilemma:
- Downsampling (stride/pool) increases receptive field but loses resolution
- Upsampling recovers resolution but information is already lost

### The Dilated Convolution Solution

Dilated convolutions provide **large receptive field without downsampling**:

```
Standard CNN path:
Input (224×224) → Conv/Pool → (112×112) → Conv/Pool → (56×56) → ... → (7×7)
                                                                        ↓
                                                                   Upsample
                                                                        ↓
                                                              Output (224×224)
                                                           [Information lost!]

Dilated CNN path:
Input (224×224) → Conv d=1 → (224×224) → Conv d=2 → (224×224) → Conv d=4 → ...
                                                                        ↓
                                                              Output (224×224)
                                                        [Full resolution preserved!]
```

### Multi-Scale Feature Extraction: ASPP

The Atrous Spatial Pyramid Pooling (ASPP) module captures multi-scale context by applying parallel dilated convolutions:

```python
class AtrousSpatialPyramidPooling(nn.Module):
    """
    ASPP module from DeepLab for multi-scale feature extraction.
    Uses parallel dilated convolutions with different rates.
    """
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        
        # 1x1 convolution (global features)
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
        
        # Global average pooling (image-level features)
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
```

---

## Deep Insight: WaveNet and Temporal Dilation

### The Audio Challenge

Audio signals require extremely long receptive fields (seconds of audio at 16kHz = tens of thousands of samples). Standard convolutions would need either:
- Impractically large kernels
- Hundreds of stacked layers

### Exponential Dilation for Temporal Data

WaveNet uses **exponentially increasing dilation** for causal convolutions:

```python
class DilatedCausalConv1d(nn.Module):
    """Dilated causal convolution for sequence modeling."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
    
    def forward(self, x):
        out = self.conv(x)
        # Remove right padding to maintain causality
        return out[:, :, :-self.padding] if self.padding > 0 else out

# Stack with exponentially increasing dilation
def build_wavenet_stack(channels, kernel_size=2, num_layers=10):
    layers = []
    for i in range(num_layers):
        dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        layers.append(DilatedCausalConv1d(channels, channels, kernel_size, dilation))
    return nn.Sequential(*layers)

# Receptive field calculation:
# With K=2 and dilations [1, 2, 4, ..., 512]:
# RF = 1 + sum(d * (K-1)) = 1 + (1+2+4+...+512) = 1024 samples
```

This achieves a receptive field of 1024 samples with only 10 layers!

---

## Complete Output Size Formula

Combining padding, stride, and dilation:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K - 1) - 1}{s} \right\rfloor + 1$$

where:
- $H_{in}$: Input height
- $p$: Padding
- $K$: Kernel size
- $d$: Dilation
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
print(f"Input=224, K=3, p=1, s=1, d=1: {conv_output_size(224, 3, 1, 1, 1)}")  # 224
print(f"Input=224, K=3, p=1, s=2, d=1: {conv_output_size(224, 3, 1, 2, 1)}")  # 112
print(f"Input=224, K=7, p=3, s=2, d=1: {conv_output_size(224, 7, 3, 2, 1)}")  # 112
print(f"Input=56, K=3, p=2, s=1, d=2: {conv_output_size(56, 3, 2, 1, 2)}")    # 56
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

# Dilation options
conv_d1 = nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1)   # Standard
conv_d2 = nn.Conv2d(3, 64, kernel_size=3, dilation=2, padding=2)   # Dilation 2
conv_d4 = nn.Conv2d(3, 64, kernel_size=3, dilation=4, padding=4)   # Dilation 4

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
    
    # Calculate receptive field (simplified for single layer)
    k = conv.kernel_size[0]
    d = conv.dilation[0]
    receptive_field = d * (k - 1) + 1
    
    print(f"{name}:")
    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(y.shape)}")
    print(f"  Parameters:   {params:,}")
    print(f"  Receptive field: {receptive_field}×{receptive_field}")
    print()

# Input: 1 batch, 64 channels, 56×56
input_shape = (1, 64, 56, 56)

# Standard 3×3 conv
conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, dilation=1)
analyze_conv("Standard 3×3", conv1, input_shape)

# Strided 3×3 conv (downsampling)
conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, dilation=1)
analyze_conv("Strided 3×3 (s=2)", conv2, input_shape)

# Dilated 3×3 conv
conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, stride=1, dilation=2)
analyze_conv("Dilated 3×3 (d=2)", conv3, input_shape)

# 5×5 conv (same receptive field as dilated 3×3 d=2)
conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1, dilation=1)
analyze_conv("Standard 5×5", conv4, input_shape)
```

**Output:**
```
Standard 3×3:
  Input shape:  [1, 64, 56, 56]
  Output shape: [1, 128, 56, 56]
  Parameters:   73,856
  Receptive field: 3×3

Strided 3×3 (s=2):
  Input shape:  [1, 64, 56, 56]
  Output shape: [1, 128, 28, 28]
  Parameters:   73,856
  Receptive field: 3×3

Dilated 3×3 (d=2):
  Input shape:  [1, 64, 56, 56]
  Output shape: [1, 128, 56, 56]
  Parameters:   73,856
  Receptive field: 5×5

Standard 5×5:
  Input shape:  [1, 64, 56, 56]
  Output shape: [1, 128, 56, 56]
  Parameters:   204,928
  Receptive field: 5×5
```

**Key observation**: Dilated 3×3 achieves the same 5×5 receptive field as standard 5×5 with **64% fewer parameters**!

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

### Dilation Selection

| Goal | Dilation | Notes |
|------|----------|-------|
| Standard convolution | 1 | Default for most layers |
| Larger receptive field | 2, 4, 8, ... | Use exponentially increasing |
| Multi-scale features | Multiple rates (ASPP) | For segmentation |

### Architecture Design Tips

In ResNet-style architectures, the typical downsampling pattern is:
- **First layer**: 7×7, stride 2 (reduces by 2×)
- **Max pool**: 3×3, stride 2 (reduces by 2×)  
- **Later stages**: 3×3, stride 2 at stage transitions

For dense prediction (segmentation):
- Remove later downsampling layers
- Replace with dilated convolutions
- Use ASPP for multi-scale context

---

## Summary

| Concept | Purpose | Effect on Output Size | Key Insight |
|---------|---------|----------------------|-------------|
| **Padding** | Preserve spatial dimensions, use edge information | Increases | Solves border effect, but zero padding creates detectable artifacts |
| **Stride** | Downsample, reduce computation | Decreases | Learnable downsampling superior to fixed pooling |
| **Dilation** | Increase receptive field without more parameters | Depends on padding | Exponential RF growth enables efficient long-range context |

The interplay of these three parameters gives CNNs remarkable flexibility in controlling:
- **Output resolution**: Critical for tasks requiring different output sizes
- **Receptive field**: How much context each output pixel "sees"
- **Computational cost**: Larger strides reduce computation; dilations add none
- **Information preservation**: Different trade-offs for different tasks

---

## References

1. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.

2. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR 2016*.

3. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE TPAMI*.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.

5. van den Oord, A., et al. (2016). WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*.

6. Zhang, R. (2019). Making convolutional networks shift-invariant again. *ICML 2019*.

7. Wang, P., et al. (2018). Understanding convolution for semantic segmentation. *WACV 2018*.
