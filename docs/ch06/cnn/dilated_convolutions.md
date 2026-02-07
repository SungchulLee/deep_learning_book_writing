# Dilated Convolutions

## Introduction

Standard convolution has a limited receptive field determined by the kernel size. To capture larger spatial context, we can either use larger kernels (more parameters: $O(K^2)$), stack more layers (more computation, deeper gradients), or use **dilated convolution** (same parameters, larger receptive field).

**Dilated convolution** (also called **atrous convolution**, from the French *à trous* meaning "with holes") inserts gaps between kernel elements, effectively expanding the kernel's spatial coverage without increasing the number of parameters. This technique is foundational to modern architectures for semantic segmentation, audio generation, and any task requiring large receptive fields at high resolution.

---

## Mathematical Definition

### Effective Kernel Size

Dilation inserts "holes" (zeros) between kernel elements. With dilation rate $d$, the **effective kernel size** becomes:

$$K_{eff} = K + (K - 1)(d - 1) = d(K - 1) + 1$$

For $K=3$:
- $d=1$: $K_{eff} = 3$ (standard convolution)
- $d=2$: $K_{eff} = 5$
- $d=4$: $K_{eff} = 9$
- $d=8$: $K_{eff} = 17$

### Formal Definition

For a 2D input $X$ and kernel $W$ with dilation $d$:

$$Y[i, j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X[i + d \cdot m, j + d \cdot n] \cdot W[m, n]$$

The kernel weights are the same as standard convolution—dilation only changes *which input positions* are sampled.

### Output Size Formula

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K - 1) - 1}{s} \right\rfloor + 1$$

To maintain the same output size as input (with stride 1), set padding to:

$$p = \frac{d(K-1)}{2}$$

For $K=3$, $d=2$: $p = 2$. For $K=3$, $d=4$: $p = 4$.

---

## Visual Representation

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

---

## Receptive Field Growth

### Comparison of Approaches

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

### Efficiency Comparison

```python
import torch.nn as nn

def compute_receptive_field(layers):
    """Compute receptive field for a sequence of conv/pool layers."""
    rf, jump = 1, 1
    for layer in layers:
        k = layer.get('kernel', 1)
        s = layer.get('stride', 1)
        d = layer.get('dilation', 1)
        k_eff = d * (k - 1) + 1
        rf = rf + (k_eff - 1) * jump
        jump = jump * s
    return rf

# Standard 3×3 convolutions (5 layers)
standard = [{'kernel': 3, 'stride': 1} for _ in range(5)]

# Dilated convolutions with increasing dilation
dilated = [
    {'kernel': 3, 'stride': 1, 'dilation': 1},
    {'kernel': 3, 'stride': 1, 'dilation': 2},
    {'kernel': 3, 'stride': 1, 'dilation': 4},
    {'kernel': 3, 'stride': 1, 'dilation': 8},
    {'kernel': 3, 'stride': 1, 'dilation': 16},
]

rf_standard = compute_receptive_field(standard)
rf_dilated = compute_receptive_field(dilated)

print(f"Standard 5 layers (3×3): RF = {rf_standard}")    # 11
print(f"Dilated 5 layers (d=1,2,4,8,16): RF = {rf_dilated}")  # 63
print(f"Ratio: {rf_dilated / rf_standard:.1f}× larger with same parameters!")
```

---

## PyTorch Implementation

### Basic Dilated Convolution

```python
import torch
import torch.nn as nn

# Standard 3×3 convolution
conv_standard = nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1)

# Dilated 3×3 convolution (d=2): 5×5 effective receptive field
conv_d2 = nn.Conv2d(64, 128, kernel_size=3, dilation=2, padding=2)

# Dilated 3×3 convolution (d=4): 9×9 effective receptive field
conv_d4 = nn.Conv2d(64, 128, kernel_size=3, dilation=4, padding=4)

x = torch.randn(1, 64, 56, 56)

# All produce the same output size (with appropriate padding)
print(f"Standard: {conv_standard(x).shape}")  # [1, 128, 56, 56]
print(f"Dilation 2: {conv_d2(x).shape}")      # [1, 128, 56, 56]
print(f"Dilation 4: {conv_d4(x).shape}")      # [1, 128, 56, 56]

# Same number of parameters!
for name, conv in [("Standard", conv_standard), ("d=2", conv_d2), ("d=4", conv_d4)]:
    params = sum(p.numel() for p in conv.parameters())
    k_eff = conv.dilation[0] * (conv.kernel_size[0] - 1) + 1
    print(f"{name}: params={params:,}, effective RF={k_eff}×{k_eff}")
```

**Key observation**: Dilated 3×3 achieves the same 5×5 receptive field as standard 5×5 with **64% fewer parameters**!

### Complete Comparison

```python
def analyze_conv(name, conv, input_shape):
    """Analyze convolution layer properties."""
    x = torch.randn(*input_shape)
    y = conv(x)
    params = sum(p.numel() for p in conv.parameters())
    k = conv.kernel_size[0]
    d = conv.dilation[0]
    receptive_field = d * (k - 1) + 1
    
    print(f"{name}:")
    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(y.shape)}")
    print(f"  Parameters:   {params:,}")
    print(f"  Receptive field: {receptive_field}×{receptive_field}")
    print()

input_shape = (1, 64, 56, 56)

# Dilated 3×3 conv
conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, stride=1, dilation=2)
analyze_conv("Dilated 3×3 (d=2)", conv3, input_shape)

# Standard 5×5 conv (same receptive field as dilated 3×3 d=2)
conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1, dilation=1)
analyze_conv("Standard 5×5", conv4, input_shape)
```

**Output:**
```
Dilated 3×3 (d=2):
  Parameters:   73,856
  Receptive field: 5×5

Standard 5×5:
  Parameters:   204,928
  Receptive field: 5×5
```

---

## The Gridding Problem

### The Issue

Stacking dilated convolutions with the **same rate** causes a "gridding" artifact where some input positions are never sampled:

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

### Solutions

**Use dilations that are not multiples of each other**, or use a "sawtooth" pattern:

- ✅ Good: $1, 2, 5, 1, 2, 5$ (HDC pattern from "Understanding Convolution for Semantic Segmentation")
- ✅ Good: $1, 2, 4, 8$ then repeat
- ❌ Bad: $2, 2, 2, 2$ (gridding)

The key insight is that consecutive dilation rates should not share a common factor greater than 1. The HDC (Hybrid Dilated Convolution) pattern ensures all input positions are covered.

---

## Dilation for Dense Prediction

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

The Atrous Spatial Pyramid Pooling (ASPP) module captures multi-scale context by applying parallel dilated convolutions at different rates:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Test ASPP module
aspp = AtrousSpatialPyramidPooling(256, 256, rates=[6, 12, 18])
x = torch.randn(2, 256, 28, 28)
out = aspp(x)
print(f"ASPP: {x.shape} → {out.shape}")  # Same spatial dimensions
print(f"Parameters: {sum(p.numel() for p in aspp.parameters()):,}")
```

---

## WaveNet and Temporal Dilation

### The Audio Challenge

Audio signals require extremely long receptive fields (seconds of audio at 16kHz = tens of thousands of samples). Standard convolutions would need either impractically large kernels or hundreds of stacked layers.

### Exponential Dilation for Temporal Data

WaveNet uses **exponentially increasing dilation** for causal 1D convolutions:

```python
import torch.nn as nn

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

This achieves a receptive field of 1024 samples with only 10 layers and constant parameter count per layer. See [1D Convolutions](conv1d.md) for more on temporal convolution architectures.

---

## Practical Guidelines

### Dilation Selection

| Goal | Dilation | Notes |
|------|----------|-------|
| Standard convolution | 1 | Default for most layers |
| Larger receptive field | 2, 4, 8, ... | Use exponentially increasing |
| Multi-scale features | Multiple rates (ASPP) | For segmentation |

### Common Dilation Patterns

```python
# Pattern 1: Exponentially increasing (WaveNet-style)
# Dilation rates: 1, 2, 4, 8, 16
# Best for: temporal data, large receptive fields

# Pattern 2: ASPP parallel rates
# Dilation rates: 6, 12, 18 (applied in parallel)
# Best for: multi-scale segmentation context

# Pattern 3: HDC (no gridding)
# Dilation rates: 1, 2, 5, 1, 2, 5, ...
# Best for: dense prediction without artifacts

# Pattern 4: Sawtooth reset
# Dilation rates: 1, 2, 4, 8, 1, 2, 4, 8
# Best for: repeated blocks with fresh coverage
```

### Architecture Design for Dense Prediction

For segmentation, the standard approach is to take a classification backbone (e.g., ResNet) and modify it:
- Remove the last two downsampling stages
- Replace with dilated convolutions to maintain resolution
- Add ASPP for multi-scale context

```python
# ResNet backbone modification for segmentation
# Original:  stride=2 at stage 4, stride=2 at stage 5
# Modified:  dilation=2 at stage 4, dilation=4 at stage 5

# This preserves 1/8 resolution instead of 1/32:
# Original:  224 → 112 → 56 → 28 → 14 → 7    (stride 32)
# Modified:  224 → 112 → 56 → 28 → 28 → 28    (stride 8)
```

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Operation** | Standard convolution with gaps between kernel elements |
| **Effective kernel** | $K_{eff} = d(K-1) + 1$ |
| **Parameters** | Same as standard (dilation is free!) |
| **Key benefit** | Exponential receptive field growth with constant parameters |
| **Main pitfall** | Gridding artifacts from repeated same-rate dilation |
| **Primary use** | Semantic segmentation, audio, any task needing large RF at full resolution |

## Key Takeaways

1. **Dilation expands the receptive field** by inserting gaps between kernel elements—no additional parameters
2. **Exponential stacking** ($d = 1, 2, 4, 8, ...$) achieves massive receptive fields with very few layers
3. **The gridding problem** occurs when stacking same-rate dilations—use varying rates to ensure full coverage
4. **ASPP** captures multi-scale context by applying parallel dilated convolutions at different rates
5. **For dense prediction**, replacing downsampling with dilation preserves spatial resolution while maintaining large receptive fields
6. **WaveNet-style architectures** use dilated causal convolutions for efficient temporal modeling

## References

1. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR 2016*.

2. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE TPAMI*.

3. van den Oord, A., et al. (2016). WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*.

4. Wang, P., et al. (2018). Understanding convolution for semantic segmentation. *WACV 2018*.

5. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.
