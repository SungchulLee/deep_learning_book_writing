# Grouped and Depthwise Separable Convolution

## Introduction

Standard convolution operations have high computational costs due to the dense connections between input and output channels. **Grouped convolution** and **depthwise separable convolution** are architectural innovations that factorize the convolution operation, dramatically reducing parameters and computation while maintaining or even improving performance.

These techniques are foundational to efficient CNN architectures like MobileNet, EfficientNet, ShuffleNet, and ResNeXt.

## Standard Convolution Review

For standard convolution with input channels $C_{in}$, output channels $C_{out}$, and kernel size $K$:

**Parameters:** $C_{out} \times C_{in} \times K \times K$

**FLOPs:** $C_{out} \times C_{in} \times K^2 \times H_{out} \times W_{out}$

## Grouped Convolution

### Concept

**Grouped convolution** divides input and output channels into $G$ groups:

- Input split into $G$ groups of $C_{in}/G$ channels each
- Each group has its own filters producing $C_{out}/G$ channels
- Outputs concatenated

### Computational Savings

**Parameters:** $\frac{C_{out} \times C_{in} \times K^2}{G}$ (reduction by factor $G$)

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Grouped convolution with 4 groups
conv_grouped = nn.Conv2d(
    in_channels=256,
    out_channels=512,
    kernel_size=3,
    padding=1,
    groups=4  # 4 independent groups
)

x = torch.randn(1, 256, 14, 14)
y = conv_grouped(x)
print(f"Output: {y.shape}")  # [1, 512, 14, 14]
```

## Depthwise Convolution

**Depthwise convolution** is grouped convolution where $G = C_{in}$:

```python
# Depthwise: each channel processed independently
depthwise = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
# Parameters: only 64 * 9 = 576 (vs 64 * 64 * 9 = 36,864 standard)
```

## Depthwise Separable Convolution

Combines depthwise and pointwise (1×1) convolutions:

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, 
                                   stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# ~8-9x fewer parameters than standard 3x3 conv
```

**Reduction ratio:** $\frac{1}{C_{out}} + \frac{1}{K^2} \approx \frac{1}{9}$ for 3×3 kernels.

## MobileNetV2 Inverted Residual

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch
        
        self.conv = nn.Sequential(
            # Expansion
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Projection (linear)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)
```

## Channel Shuffle

For grouped convolutions, shuffle channels to enable cross-group communication:

```python
def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)
```

## Summary

| Type | Parameters | Use Case |
|------|------------|----------|
| Grouped | ÷G reduction | ResNeXt |
| Depthwise Separable | ~9× reduction | MobileNet, EfficientNet |
| Inverted Residual | Expansion + depthwise | MobileNetV2+ |

## References

1. Howard, A. G., et al. (2017). MobileNets: Efficient CNNs for Mobile Vision Applications.
2. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
3. Zhang, X., et al. (2018). ShuffleNet: An Extremely Computation-Efficient CNN.
