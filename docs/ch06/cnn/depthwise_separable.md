# Grouped and Depthwise Separable Convolution

## Introduction

Standard convolution operations have high computational costs due to the dense connections between input and output channels. **Grouped convolution** and **depthwise separable convolution** are architectural innovations that factorize the convolution operation, dramatically reducing parameters and computation while maintaining or even improving performance.

These techniques are foundational to efficient CNN architectures like MobileNet, EfficientNet, ShuffleNet, and ResNeXt, enabling deployment on mobile and edge devices.

---

## Standard Convolution Review

For standard convolution with input channels $C_{in}$, output channels $C_{out}$, and kernel size $K$:

- **Parameters**: $C_{out} \times C_{in} \times K \times K$
- **FLOPs**: $C_{out} \times C_{in} \times K^2 \times H_{out} \times W_{out}$

Each filter has **full connectivity** to all input channels—this is where the computational cost comes from.

---

## Grouped Convolution

### Concept

**Grouped convolution** divides both input and output channels into $G$ groups, with each group processed independently:

```
Standard Convolution:           Grouped Convolution (G=2):
                                
C_in ────────────→ C_out        C_in/2 ──→ C_out/2  (Group 1)
(all channels      (all           
connected)         outputs)      C_in/2 ──→ C_out/2  (Group 2)
```

- Input split into $G$ groups of $C_{in}/G$ channels each
- Each group has its own filters producing $C_{out}/G$ channels
- Outputs concatenated along channel dimension

### Mathematical Formulation

For group $g$ (where $g = 0, 1, ..., G-1$):

$$Y_{g}[o, i, j] = \sum_{c=0}^{C_{in}/G - 1} \sum_{m,n} X_g[c, i+m, j+n] \cdot K_g[o, c, m, n]$$

where:
- $X_g$: Input channels $[g \cdot C_{in}/G, (g+1) \cdot C_{in}/G)$
- $Y_g$: Output channels $[g \cdot C_{out}/G, (g+1) \cdot C_{out}/G)$
- $K_g$: Kernels for group $g$

### Computational Savings

| Metric | Standard | Grouped (G groups) | Reduction |
|--------|----------|-------------------|-----------|
| Parameters | $C_{out} \times C_{in} \times K^2$ | $C_{out} \times \frac{C_{in}}{G} \times K^2$ | $G\times$ |
| FLOPs | $C_{out} \times C_{in} \times K^2 \times H \times W$ | $\frac{1}{G}$ of standard | $G\times$ |

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Standard convolution
conv_standard = nn.Conv2d(64, 128, kernel_size=3, padding=1)
params_standard = sum(p.numel() for p in conv_standard.parameters())
print(f"Standard conv params: {params_standard:,}")  # 73,856

# Grouped convolution (G=2)
conv_grouped_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=2)
params_grouped_2 = sum(p.numel() for p in conv_grouped_2.parameters())
print(f"Grouped conv (G=2) params: {params_grouped_2:,}")  # 36,992 (2× reduction)

# Grouped convolution (G=4)
conv_grouped_4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)
params_grouped_4 = sum(p.numel() for p in conv_grouped_4.parameters())
print(f"Grouped conv (G=4) params: {params_grouped_4:,}")  # 18,560 (4× reduction)

# Verify shapes
x = torch.randn(1, 64, 32, 32)
print(f"\nInput shape: {x.shape}")
print(f"Standard output: {conv_standard(x).shape}")
print(f"Grouped (G=2) output: {conv_grouped_2(x).shape}")
print(f"Grouped (G=4) output: {conv_grouped_4(x).shape}")
```

### Constraints

- $C_{in}$ must be divisible by $G$
- $C_{out}$ must be divisible by $G$
- When $G = C_{in} = C_{out}$, we get depthwise convolution

---

## Depthwise Convolution

### Concept

**Depthwise convolution** is the extreme case of grouped convolution where $G = C_{in}$. Each input channel has its own separate filter:

```
Depthwise Convolution:

Channel 1 ──[Filter 1]──→ Output Channel 1
Channel 2 ──[Filter 2]──→ Output Channel 2
Channel 3 ──[Filter 3]──→ Output Channel 3
    ⋮            ⋮              ⋮
Channel C ──[Filter C]──→ Output Channel C
```

### Mathematical Formulation

$$Y[c, i, j] = \sum_{m,n} X[c, i+m, j+n] \cdot K[c, m, n]$$

Each channel is convolved independently with its own $K \times K$ filter.

### Computational Analysis

| Metric | Standard | Depthwise | Reduction Factor |
|--------|----------|-----------|------------------|
| Parameters | $C \times C \times K^2$ | $C \times K^2$ | $C\times$ |
| FLOPs | $C^2 \times K^2 \times H \times W$ | $C \times K^2 \times H \times W$ | $C\times$ |

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Depthwise convolution: groups = in_channels = out_channels
depthwise_conv = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=1,
    groups=64  # Key: groups equals number of channels
)

params = sum(p.numel() for p in depthwise_conv.parameters())
print(f"Depthwise conv params: {params:,}")  # 640 (64 × 3 × 3 + 64 bias)

# Compare with standard
standard_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
standard_params = sum(p.numel() for p in standard_conv.parameters())
print(f"Standard conv params: {standard_params:,}")  # 36,928

print(f"Reduction: {standard_params / params:.1f}×")  # ~58× reduction
```

---

## Depthwise Separable Convolution

### Concept

Depthwise separable convolution splits standard convolution into two steps:

1. **Depthwise**: Spatial filtering (captures spatial patterns per channel)
2. **Pointwise (1×1)**: Channel mixing (combines information across channels)

```
Standard Convolution:
Input (C_in, H, W) ──[K×K×C_in×C_out]──→ Output (C_out, H, W)

Depthwise Separable Convolution:
Input (C_in, H, W) ──[Depthwise K×K×C_in]──→ (C_in, H, W) ──[Pointwise 1×1×C_in×C_out]──→ Output (C_out, H, W)
```

### Mathematical Formulation

**Step 1 - Depthwise**:
$$M[c, i, j] = \sum_{m,n} X[c, i+m, j+n] \cdot K_{dw}[c, m, n]$$

**Step 2 - Pointwise**:
$$Y[o, i, j] = \sum_{c=0}^{C_{in}-1} M[c, i, j] \cdot K_{pw}[o, c]$$

### Computational Comparison

For input $(C_{in}, H, W)$, output $(C_{out}, H', W')$, kernel $K$:

| Component | Parameters | FLOPs |
|-----------|-----------|-------|
| Standard | $C_{in} \times C_{out} \times K^2$ | $C_{in} \times C_{out} \times K^2 \times H' \times W'$ |
| Depthwise | $C_{in} \times K^2$ | $C_{in} \times K^2 \times H' \times W'$ |
| Pointwise | $C_{in} \times C_{out}$ | $C_{in} \times C_{out} \times H' \times W'$ |
| **DS Total** | $C_{in}(K^2 + C_{out})$ | $C_{in}(K^2 + C_{out}) \times H' \times W'$ |

### Reduction Ratio

$$\frac{\text{Standard}}{\text{Depthwise Separable}} = \frac{C_{in} \times C_{out} \times K^2}{C_{in} \times K^2 + C_{in} \times C_{out}} = \frac{1}{\frac{1}{C_{out}} + \frac{1}{K^2}}$$

For $K=3$, $C_{out}=256$: **Reduction ≈ 8-9×**

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.
    
    Consists of:
    1. Depthwise convolution: spatial filtering per channel
    2. Pointwise convolution: 1×1 conv for channel mixing
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, bias=False):
        super().__init__()
        
        # Depthwise: groups = in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        
        # Pointwise: 1×1 convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Compare with standard convolution
in_ch, out_ch = 64, 128
kernel_size = 3

standard = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1, bias=False)
ds_conv = DepthwiseSeparableConv(in_ch, out_ch, kernel_size, padding=1, bias=False)

standard_params = sum(p.numel() for p in standard.parameters())
ds_params = sum(p.numel() for p in ds_conv.parameters())

print(f"Standard conv params: {standard_params:,}")     # 73,728
print(f"Depthwise separable params: {ds_params:,}")     # 8,768
print(f"Reduction: {standard_params / ds_params:.1f}×")  # ~8.4×

# Verify output shapes match
x = torch.randn(1, in_ch, 32, 32)
print(f"\nStandard output: {standard(x).shape}")
print(f"DS conv output: {ds_conv(x).shape}")
```

---

## MobileNet V1 Block

MobileNet uses depthwise separable convolutions with batch normalization and ReLU:

```python
import torch
import torch.nn as nn

class MobileNetV1Block(nn.Module):
    """
    MobileNet V1 style depthwise separable block.
    
    Structure:
    Depthwise Conv → BN → ReLU → Pointwise Conv → BN → ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

---

## MobileNet V2: Inverted Residuals

MobileNet V2 introduces **inverted residual blocks** with linear bottlenecks:

```
Standard Residual:          Inverted Residual:
wide → narrow → wide        narrow → wide → narrow

Input (C)                   Input (C)
    ↓                           ↓
Conv 1×1 (C→C/4)           Conv 1×1 (C→C×t)  [Expansion]
    ↓                           ↓
Conv 3×3 (C/4)             DWConv 3×3 (C×t)  [Depthwise]
    ↓                           ↓
Conv 1×1 (C/4→C)           Conv 1×1 (C×t→C') [Projection]
    ↓                           ↓
Add residual               Add residual (if stride=1 and C=C')
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    """
    MobileNet V2 Inverted Residual Block.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for depthwise conv
        expand_ratio: Expansion factor for hidden channels
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_channels = in_channels * expand_ratio
        
        layers = []
        
        # Expansion (1×1 conv) - only if expand_ratio > 1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=stride,
                      padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection (1×1 conv) - LINEAR (no activation!)
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


# Example usage
block = InvertedResidual(32, 32, stride=1, expand_ratio=6)
x = torch.randn(1, 32, 56, 56)
out = block(x)
print(f"Input: {x.shape}, Output: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

---

## Channel Shuffle (ShuffleNet)

Grouped convolutions don't allow information flow between groups. **Channel shuffle** addresses this limitation:

```
Before Shuffle:                After Shuffle:
Group 1: [a₁, a₂, a₃]         [a₁, b₁, c₁]
Group 2: [b₁, b₂, b₃]    →    [a₂, b₂, c₂]
Group 3: [c₁, c₂, c₃]         [a₃, b₃, c₃]
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    """
    Channel shuffle operation.
    
    Reshapes tensor from (N, C, H, W) to (N, G, C//G, H, W),
    transposes groups and channels, then flattens back.
    """
    N, C, H, W = x.shape
    
    # Reshape: (N, C, H, W) → (N, G, C//G, H, W)
    x = x.view(N, groups, C // groups, H, W)
    
    # Transpose: (N, G, C//G, H, W) → (N, C//G, G, H, W)
    x = x.transpose(1, 2).contiguous()
    
    # Flatten: (N, C//G, G, H, W) → (N, C, H, W)
    x = x.view(N, C, H, W)
    
    return x


class ShuffleNetBlock(nn.Module):
    """ShuffleNet V1 unit with grouped convolutions and channel shuffle."""
    
    def __init__(self, in_channels, out_channels, groups=3, stride=1):
        super().__init__()
        
        self.stride = stride
        self.groups = groups
        
        mid_channels = out_channels // 4
        
        if stride == 2:
            out_channels = out_channels - in_channels
        
        # Group convolution 1×1
        self.gconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise 3×3
        self.dwconv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        
        # Group convolution 1×1
        self.gconv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.gconv1(x)
        out = channel_shuffle(out, self.groups)
        out = self.dwconv(out)
        out = self.gconv2(out)
        
        shortcut = self.shortcut(x)
        
        if self.stride == 2:
            out = torch.cat([out, shortcut], dim=1)
        else:
            out = out + shortcut
        
        return nn.functional.relu(out)


# Test
block = ShuffleNetBlock(24, 24, groups=3, stride=1)
x = torch.randn(1, 24, 56, 56)
out = block(x)
print(f"ShuffleNet block: {x.shape} → {out.shape}")
```

---

## Efficiency Comparison

```python
import torch
import torch.nn as nn

def count_ops_and_params(model, input_shape):
    """Count parameters and estimate FLOPs."""
    params = sum(p.numel() for p in model.parameters())
    
    flops = 0
    x = torch.randn(*input_shape)
    
    def hook(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2:]
            flops += (2 * module.kernel_size[0] * module.kernel_size[1] * 
                     module.in_channels * module.out_channels * 
                     out_h * out_w // module.groups)
    
    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))
    
    _ = model(x)
    
    for h in hooks:
        h.remove()
    
    return params, flops


# Compare different convolution types
in_ch, out_ch = 64, 128
H, W = 56, 56

standard = nn.Conv2d(in_ch, out_ch, 3, padding=1)
ds_conv = nn.Sequential(
    nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
    nn.Conv2d(in_ch, out_ch, 1)
)
grouped = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=4)

input_shape = (1, in_ch, H, W)

print("Comparison of Convolution Types:")
print("-" * 50)
for name, model in [("Standard", standard), 
                    ("Depthwise Sep", ds_conv), 
                    ("Grouped (G=4)", grouped)]:
    params, flops = count_ops_and_params(model, input_shape)
    print(f"{name:15s}: Params={params:>10,}, FLOPs={flops:>15,}")
```

---

## Architecture Comparison

| Architecture | Key Innovation | Typical Use Case |
|--------------|----------------|------------------|
| **MobileNetV1** | Depthwise separable conv | Mobile deployment |
| **MobileNetV2** | Inverted residual + linear bottleneck | Mobile/edge |
| **ShuffleNet** | Channel shuffle + grouped conv | Extremely efficient |
| **EfficientNet** | Compound scaling + MBConv | State-of-the-art efficiency |
| **ResNeXt** | Grouped convolutions in residual | High accuracy |

---

## Summary

| Type | Parameters | Reduction | Use Case |
|------|------------|-----------|----------|
| Standard | $C_{out} \times C_{in} \times K^2$ | — | Baseline |
| Grouped (G) | $\div G$ | $G\times$ | ResNeXt |
| Depthwise | $C \times K^2$ | $C\times$ | Spatial filtering |
| Depthwise Separable | $C_{in}(K^2 + C_{out})$ | ~8-9× | MobileNet, EfficientNet |
| Inverted Residual | Expansion + depthwise | Efficient | MobileNetV2+ |

## Key Takeaways

1. **Grouped convolution** divides channels into independent groups, reducing parameters by $G\times$
2. **Depthwise convolution** is grouped conv with $G = C_{in}$, applying one filter per channel
3. **Depthwise separable** = depthwise + pointwise, achieving ~8-9× parameter reduction
4. **Channel shuffle** enables information flow between groups in ShuffleNet
5. **Inverted residuals** (MobileNetV2) use narrow → wide → narrow structure with linear bottleneck
6. These techniques enable efficient models for mobile/edge deployment without significant accuracy loss

## References

1. Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."
2. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks."
4. Zhang, X., et al. (2018). "ShuffleNet: An Extremely Computation-Efficient CNN for Mobile Devices."
5. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
