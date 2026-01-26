# MobileNet: Efficient Mobile Vision

## Overview

MobileNet uses **depthwise separable convolutions** to build lightweight neural networks for mobile and embedded vision applications. By factoring standard convolutions, MobileNet achieves 8-9× fewer parameters and FLOPs while maintaining competitive accuracy.

!!! info "Key Papers"
    - Howard et al., 2017 - "MobileNets: Efficient CNNs for Mobile Vision" ([arXiv:1704.04861](https://arxiv.org/abs/1704.04861))
    - Sandler et al., 2018 - "MobileNetV2: Inverted Residuals" ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381))

## Learning Objectives

1. Understand depthwise separable convolutions
2. Implement MobileNetV1 and MobileNetV2
3. Apply width and resolution multipliers
4. Deploy efficient models on resource-constrained devices

## Depthwise Separable Convolutions

### Standard vs Depthwise Separable

**Standard Convolution**:
- Input: H × W × C_in
- Kernel: K × K × C_in × C_out
- Parameters: K² × C_in × C_out
- FLOPs: H × W × K² × C_in × C_out

**Depthwise Separable** = Depthwise + Pointwise:

```
Standard Conv:              Depthwise Separable:

Input (H×W×C_in)           Input (H×W×C_in)
      │                          │
      ▼                          ▼
┌───────────────┐          ┌───────────────┐
│ K×K×C_in×C_out│          │ K×K×1×C_in    │ Depthwise
└───────────────┘          │ (groups=C_in) │ (one filter per channel)
      │                    └───────────────┘
      ▼                          │
Output (H×W×C_out)               ▼
                           ┌───────────────┐
                           │ 1×1×C_in×C_out│ Pointwise
                           │ (combine)     │
                           └───────────────┘
                                 │
                                 ▼
                           Output (H×W×C_out)
```

### Computational Savings

**Reduction ratio**:
$$\frac{\text{Depthwise Separable}}{\text{Standard}} = \frac{K^2 \cdot C_{in} + C_{in} \cdot C_{out}}{K^2 \cdot C_{in} \cdot C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

For K=3, C_out=256: **~8-9× fewer operations**

## Implementation

### Depthwise Separable Block

```python
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution = Depthwise + Pointwise"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Depthwise: one filter per input channel
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise: 1×1 conv to combine channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

### MobileNetV1

```python
class MobileNetV1(nn.Module):
    """MobileNet V1 with depthwise separable convolutions."""
    
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        
        # Configuration: [out_channels, stride]
        config = [
            [64, 1], [128, 2], [128, 1], [256, 2], [256, 1],
            [512, 2], [512, 1], [512, 1], [512, 1], [512, 1], [512, 1],
            [1024, 2], [1024, 1]
        ]
        
        # Stem
        in_ch = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True)
        )
        
        # Depthwise separable blocks
        layers = []
        for out_ch, stride in config:
            out_ch = int(out_ch * width_mult)
            layers.append(DepthwiseSeparableConv(in_ch, out_ch, stride))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        
        # Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## MobileNetV2: Inverted Residuals

### Key Innovation: Linear Bottlenecks

V2 introduces **inverted residuals**: expand → depthwise → compress

```
MobileNetV1 (no skip):    MobileNetV2 (inverted residual):

Input                     Input ──────────────────┐
   │                         │                    │
   ▼                         ▼                    │
Depthwise 3×3             Expand 1×1 (6×)         │
   │                         │                    │
   ▼                         ▼                    │
Pointwise 1×1             Depthwise 3×3           │
   │                         │                    │
   ▼                         ▼                    │
Output                    Compress 1×1 (linear)   │
                             │                    │
                             ▼                    │
                            (+) ◄─────────────────┘
                             │
                             ▼
                          Output
```

### Why Linear Bottleneck?

ReLU destroys information in low-dimensional spaces. The projection layer uses **no activation** (linear) to preserve information.

```python
class InvertedResidual(nn.Module):
    """Inverted Residual block from MobileNetV2."""
    
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expand_ratio
        
        layers = []
        
        # Expansion (if ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection (LINEAR - no ReLU!)
        layers.extend([
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
```

### MobileNetV2 Architecture

```python
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        
        # Config: [expand, channels, layers, stride]
        config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Stem
        in_ch = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks
        blocks = []
        for expand, ch, layers, stride in config:
            out_ch = int(ch * width_mult)
            for i in range(layers):
                blocks.append(InvertedResidual(
                    in_ch, out_ch, stride if i == 0 else 1, expand
                ))
                in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        
        # Head
        final_ch = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, final_ch, 1, bias=False),
            nn.BatchNorm2d(final_ch),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(final_ch, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.features(self.stem(x)))
```

## Width and Resolution Multipliers

### Width Multiplier (α)

Reduces channels uniformly:

```python
# α = 1.0: Full MobileNet (3.5M params)
# α = 0.75: ~2.6M params
# α = 0.5: ~1.9M params
# α = 0.35: ~1.66M params

model = MobileNetV2(width_mult=0.5)  # Half the channels
```

### Resolution Multiplier (ρ)

Reduces input resolution:

```python
# ρ = 1.0: 224×224
# ρ = 0.857: 192×192
# ρ = 0.714: 160×160
# ρ = 0.571: 128×128

transform = transforms.Resize(160)  # Use smaller input
```

## MobileNetV3

V3 adds hardware-aware NAS, h-swish activation, and SE blocks:

```python
class HSwish(nn.Module):
    """Hard-swish: x * ReLU6(x + 3) / 6"""
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class MobileNetV3Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, expand, use_se, use_hs):
        super().__init__()
        hidden = in_ch * expand
        act = HSwish if use_hs else nn.ReLU
        
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            act()
        ) if expand != 1 else nn.Identity()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel, stride, kernel//2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            act()
        )
        
        self.se = SqueezeExcitation(hidden, hidden // 4) if use_se else nn.Identity()
        
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        self.use_residual = (stride == 1 and in_ch == out_ch)
    
    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        return out + x if self.use_residual else out
```

## Performance Comparison

| Model | Top-1 | Params | FLOPs | Latency (CPU) |
|-------|-------|--------|-------|---------------|
| MobileNetV1 | 70.6% | 4.2M | 569M | 113ms |
| MobileNetV2 | 72.0% | 3.5M | 300M | 75ms |
| MobileNetV3-Small | 67.4% | 2.5M | 56M | 15ms |
| MobileNetV3-Large | 75.2% | 5.4M | 219M | 51ms |
| ResNet-50 | 76.1% | 25.6M | 4.1B | ~400ms |

## Exercises

1. Compare standard conv vs depthwise separable FLOPs on paper
2. Implement MobileNetV2 and train on CIFAR-10
3. Measure actual CPU inference time with different width multipliers
4. Implement MobileNetV3 with h-swish and SE

## Key Takeaways

1. **Depthwise separable** convolutions reduce computation ~8-9×
2. **Inverted residuals** expand then compress with linear projection
3. **Width/resolution multipliers** provide easy accuracy-efficiency trade-offs
4. MobileNet family is essential for edge/mobile deployment

## References

1. Howard et al. (2017). MobileNets. arXiv.
2. Sandler et al. (2018). MobileNetV2. CVPR.
3. Howard et al. (2019). MobileNetV3. ICCV.

---

**Previous**: [EfficientNet](efficientnet.md) | **Next**: [DenseNet](densenet.md)
