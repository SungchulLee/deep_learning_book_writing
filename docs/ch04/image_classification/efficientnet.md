# EfficientNet: Principled Model Scaling

## Overview

EfficientNet introduced **compound scaling**, a principled method that uniformly scales network depth, width, and resolution using fixed scaling coefficients. Combined with a Neural Architecture Search (NAS) baseline, this approach achieves state-of-the-art accuracy with significantly fewer parameters and FLOPs.

!!! info "Key Paper"
    Tan & Le, 2019 - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" ([arXiv:1905.11946](https://arxiv.org/abs/1905.11946))

## Learning Objectives

After completing this section, you will be able to:

1. Understand compound scaling for balanced network growth
2. Implement MBConv blocks with squeeze-and-excitation
3. Build EfficientNet-B0 through B7 architectures
4. Compare compound scaling with single-dimension scaling

## The Scaling Problem

### Traditional Approaches

Before EfficientNet, models were scaled along single dimensions:

- **Width**: More channels (Wide ResNet)
- **Depth**: More layers (ResNet-152)
- **Resolution**: Larger images (299×299, 331×331)

**Problem**: Single-dimension scaling yields diminishing returns.

### Key Insight: Coupled Dimensions

The paper observed that optimal design requires balancing all three dimensions:

- Higher resolution → needs more depth for fine details
- More depth → needs more width for capacity
- All dimensions should scale together

## Compound Scaling

### Mathematical Formulation

$$\text{depth: } d = \alpha^\phi$$
$$\text{width: } w = \beta^\phi$$
$$\text{resolution: } r = \gamma^\phi$$

Constraint: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (FLOPs double per φ increment)

**EfficientNet coefficients** (from grid search):
- α = 1.2, β = 1.1, γ = 1.15

### EfficientNet Family

| Model | φ | Resolution | Parameters | FLOPs | Top-1 |
|-------|---|------------|------------|-------|-------|
| B0 | 0 | 224 | 5.3M | 390M | 77.3% |
| B1 | 0.5 | 240 | 7.8M | 700M | 79.2% |
| B2 | 1 | 260 | 9.2M | 1.0B | 80.3% |
| B3 | 2 | 300 | 12M | 1.8B | 81.7% |
| B4 | 3 | 380 | 19M | 4.2B | 83.0% |
| B5 | 4 | 456 | 30M | 9.9B | 83.7% |
| B6 | 5 | 528 | 43M | 19B | 84.2% |
| B7 | 6 | 600 | 66M | 37B | 84.4% |

## Architecture Components

### Swish Activation

$$\text{Swish}(x) = x \cdot \sigma(x)$$

```python
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
# Or use nn.SiLU() in PyTorch
```

### Squeeze-and-Excitation Block

```python
class SqueezeExcitation(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, 1),
            nn.SiLU(),
            nn.Conv2d(squeeze_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)
```

### MBConv Block

```python
class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck with SE."""
    
    def __init__(self, in_ch, out_ch, kernel, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expand_ratio
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel, stride, kernel//2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )
        
        # SE
        self.se = SqueezeExcitation(hidden, max(1, int(in_ch * se_ratio)))
        
        # Projection (linear)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x
```

### Full EfficientNet

```python
class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000, dropout=0.2):
        super().__init__()
        
        # Base config: [expand, channels, layers, stride, kernel]
        config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        # Stem
        out_ch = self._round(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        
        # Blocks
        in_ch = out_ch
        blocks = []
        for expand, ch, layers, stride, kernel in config:
            out_ch = self._round(ch, width_mult)
            layers = int(math.ceil(depth_mult * layers))
            for i in range(layers):
                blocks.append(MBConvBlock(in_ch, out_ch, kernel, stride if i==0 else 1, expand))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final = self._round(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, final, 1, bias=False),
            nn.BatchNorm2d(final),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final, num_classes)
        )
    
    def _round(self, ch, mult, divisor=8):
        ch = int(ch * mult)
        return max(divisor, (ch + divisor//2) // divisor * divisor)
    
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

# Factory functions
def efficientnet_b0(num_classes=1000):
    return EfficientNet(1.0, 1.0, num_classes, 0.2)

def efficientnet_b3(num_classes=1000):
    return EfficientNet(1.2, 1.4, num_classes, 0.3)
```

## Training Recommendations

- **Optimizer**: RMSprop (α=0.9, ε=0.001, momentum=0.9)
- **Learning rate**: 0.256 with exponential decay
- **Augmentation**: RandAugment/AutoAugment
- **Regularization**: Stochastic depth, dropout

## Comparison

| Model | Top-1 | Params | FLOPs |
|-------|-------|--------|-------|
| ResNet-50 | 76.1% | 25.6M | 4.1B |
| EfficientNet-B0 | 77.3% | 5.3M | 390M |
| EfficientNet-B3 | 81.7% | 12M | 1.8B |

**EfficientNet achieves better accuracy with ~5-10× fewer FLOPs.**

## Exercises

1. Compare EfficientNet-B0 vs ResNet-50 on CIFAR-10
2. Implement SE block and measure accuracy improvement
3. Grid search for custom α, β, γ on a small dataset
4. Implement progressive training (increasing resolution)

## Key Takeaways

1. **Compound scaling** balances depth, width, and resolution
2. **MBConv + SE** provides efficient attention-enhanced blocks
3. **Swish** offers slight improvements over ReLU
4. EfficientNet achieves SOTA accuracy with fewer resources

## References

1. Tan & Le (2019). EfficientNet: Rethinking Model Scaling. ICML.
2. Tan & Le (2021). EfficientNetV2. ICML.
3. Sandler et al. (2018). MobileNetV2. CVPR.

---

**Previous**: [Inception](inception.md) | **Next**: [MobileNet](mobilenet.md)
