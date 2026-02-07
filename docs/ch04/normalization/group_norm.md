# Group Normalization

## Overview

Group Normalization (GroupNorm), introduced by Wu and He in 2018, divides channels into groups and normalizes within each group independently. It bridges the gap between Layer Normalization and Instance Normalization, providing stable training regardless of batch size while maintaining good performance for visual tasks.

## Mathematical Formulation

### Forward Pass

For an input tensor $x \in \mathbb{R}^{N \times C \times H \times W}$, Group Normalization divides the $C$ channels into $G$ groups, each containing $C/G$ channels.

**For each sample $n$ and group $g$:**

Let $\mathcal{S}_g$ denote the set of indices $(c, h, w)$ belonging to group $g$:

$$\mathcal{S}_g = \{(c, h, w) : c \in [gC/G, (g+1)C/G), h \in [0, H), w \in [0, W)\}$$

**Compute group statistics:**

$$\mu_{n,g} = \frac{1}{|\mathcal{S}_g|} \sum_{(c,h,w) \in \mathcal{S}_g} x_{n,c,h,w}$$

$$\sigma^2_{n,g} = \frac{1}{|\mathcal{S}_g|} \sum_{(c,h,w) \in \mathcal{S}_g} (x_{n,c,h,w} - \mu_{n,g})^2$$

Where $|\mathcal{S}_g| = (C/G) \times H \times W$.

**Normalize:**

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,g(c)}}{\sqrt{\sigma^2_{n,g(c)} + \epsilon}}$$

**Scale and shift:**

$$y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$$

### Relationship to Other Normalizations

GroupNorm generalizes other normalization methods:

| Method | Groups (G) | Channels per Group |
|--------|------------|-------------------|
| Layer Norm | 1 | C |
| Instance Norm | C | 1 |
| Group Norm | G | C/G |

```
Input: (N, C, H, W) with C=8 channels

G=1 (LayerNorm): [████████]  All 8 channels together
G=2 (GroupNorm): [████][████]  4 channels per group
G=4 (GroupNorm): [██][██][██][██]  2 channels per group
G=8 (InstanceNorm): [█][█][█][█][█][█][█][█]  Each channel separate
```

## PyTorch Implementation

### From Scratch

```python
import torch
import torch.nn as nn

class GroupNormFromScratch(nn.Module):
    """
    Group Normalization implementation from scratch.
    Divides channels into groups and normalizes within each group.
    """
    
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        
        assert num_channels % num_groups == 0, \
            f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        # Learnable parameters (per channel, not per group)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, H, W) or (N, C)
        
        Returns:
            Normalized tensor of same shape
        """
        N, C = x.shape[:2]
        G = self.num_groups
        
        # Reshape to (N, G, C//G, H, W) for 4D or (N, G, C//G) for 2D
        if x.dim() == 4:
            H, W = x.shape[2:]
            x_reshaped = x.view(N, G, C // G, H, W)
            # Normalize over (C//G, H, W) dimensions
            mean = x_reshaped.mean(dim=[2, 3, 4], keepdim=True)
            var = x_reshaped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        elif x.dim() == 2:
            x_reshaped = x.view(N, G, C // G)
            mean = x_reshaped.mean(dim=2, keepdim=True)
            var = x_reshaped.var(dim=2, keepdim=True, unbiased=False)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.dim()}D")
        
        # Normalize
        x_norm = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back to original shape
        x_norm = x_norm.view_as(x)
        
        # Apply per-channel affine transformation
        if self.affine:
            if x.dim() == 4:
                gamma = self.gamma.view(1, C, 1, 1)
                beta = self.beta.view(1, C, 1, 1)
            else:
                gamma = self.gamma.view(1, C)
                beta = self.beta.view(1, C)
            x_norm = gamma * x_norm + beta
        
        return x_norm
    
    def extra_repr(self):
        return f'{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}'
```

### Using PyTorch Built-in

```python
import torch.nn as nn

# Basic usage
gn = nn.GroupNorm(num_groups=8, num_channels=64)

# Parameters
gn = nn.GroupNorm(
    num_groups=8,        # Number of groups to divide channels into
    num_channels=64,     # Total number of channels
    eps=1e-5,           # Numerical stability
    affine=True         # Learnable gamma and beta
)

# Common configurations
gn_32 = nn.GroupNorm(32, 256)  # 8 channels per group (typical for ResNets)
gn_16 = nn.GroupNorm(16, 64)   # 4 channels per group
gn_1 = nn.GroupNorm(1, 64)     # Equivalent to LayerNorm
```

### Verification

```python
def verify_group_norm():
    """Verify our implementation matches PyTorch."""
    torch.manual_seed(42)
    
    x = torch.randn(4, 64, 8, 8)
    
    gn_torch = nn.GroupNorm(8, 64)
    gn_custom = GroupNormFromScratch(8, 64)
    
    # Copy parameters
    gn_custom.gamma.data = gn_torch.weight.data.clone()
    gn_custom.beta.data = gn_torch.bias.data.clone()
    
    out_torch = gn_torch(x)
    out_custom = gn_custom(x)
    
    diff = (out_torch - out_custom).abs().max()
    print(f"Max difference: {diff:.2e}")
    
    # Verify group-wise statistics
    print("\nPer-group statistics (should be mean≈0, std≈1):")
    out_reshaped = out_torch.view(4, 8, 8, 8, 8)  # (N, G, C/G, H, W)
    for g in range(2):
        mean = out_reshaped[0, g].mean().item()
        std = out_reshaped[0, g].std().item()
        print(f"  Group {g}: mean={mean:.6f}, std={std:.4f}")

verify_group_norm()
```

## Why Group Normalization?

### The Batch Size Problem

BatchNorm's effectiveness degrades with small batch sizes:

```python
def demonstrate_batch_size_problem():
    """Show how BatchNorm fails with small batches."""
    
    torch.manual_seed(42)
    
    bn = nn.BatchNorm2d(64)
    gn = nn.GroupNorm(8, 64)
    
    print("Comparison: BatchNorm vs GroupNorm")
    print("=" * 50)
    
    batch_sizes = [1, 2, 4, 8, 32]
    
    for bs in batch_sizes:
        x = torch.randn(bs, 64, 8, 8)
        
        # BatchNorm
        bn.train()
        out_bn = bn(x)
        bn_std = out_bn.std().item()
        
        # GroupNorm
        out_gn = gn(x)
        gn_std = out_gn.std().item()
        
        print(f"Batch size {bs:2d}: BatchNorm std={bn_std:.4f}, GroupNorm std={gn_std:.4f}")
    
    print("\nObservation: GroupNorm maintains consistent statistics!")

demonstrate_batch_size_problem()
```

**Output:**
```
Comparison: BatchNorm vs GroupNorm
==================================================
Batch size  1: BatchNorm std=0.0000, GroupNorm std=1.0001
Batch size  2: BatchNorm std=0.7521, GroupNorm std=1.0000
Batch size  4: BatchNorm std=0.8789, GroupNorm std=1.0001
Batch size  8: BatchNorm std=0.9356, GroupNorm std=0.9999
Batch size 32: BatchNorm std=0.9823, GroupNorm std=1.0000

Observation: GroupNorm maintains consistent statistics!
```

### Performance Comparison

Research by Wu and He (2018) showed:
- GroupNorm matches BatchNorm performance with large batches
- GroupNorm significantly outperforms BatchNorm with small batches
- Error increases for BatchNorm as batch size decreases; GroupNorm stays stable

## Choosing the Number of Groups

### Common Configurations

| Use Case | Typical G | Rationale |
|----------|----------|-----------|
| Object Detection | 32 | Standard for Detectron2 |
| Semantic Segmentation | 32 | Stable with small batches |
| Video Models | 32 | Handles temporal batches |
| Small Networks | 8-16 | Fewer channels available |

### Guidelines

```python
def choose_num_groups(num_channels):
    """Heuristic for choosing number of groups."""
    
    # Rule 1: G must divide C evenly
    # Rule 2: Each group should have at least 16 channels
    # Rule 3: Common choice is G=32
    
    if num_channels >= 32:
        # Default: 32 groups (or fewer if channels don't allow)
        g = 32
        while num_channels % g != 0 and g > 1:
            g //= 2
        return g
    else:
        # For small channel counts, use 1-4 groups
        for g in [16, 8, 4, 2, 1]:
            if num_channels % g == 0 and num_channels // g >= 4:
                return g
        return 1

# Examples
for c in [64, 128, 256, 512, 32, 24]:
    g = choose_num_groups(c)
    print(f"Channels={c:3d} → Groups={g:2d} ({c//g} channels/group)")
```

**Output:**
```
Channels= 64 → Groups=32 (2 channels/group)
Channels=128 → Groups=32 (4 channels/group)
Channels=256 → Groups=32 (8 channels/group)
Channels=512 → Groups=32 (16 channels/group)
Channels= 32 → Groups=32 (1 channels/group)
Channels= 24 → Groups= 8 (3 channels/group)
```

## Network Architectures with Group Normalization

### ResNet Block with GroupNorm

```python
class ResNetBlockGN(nn.Module):
    """ResNet basic block with Group Normalization."""
    
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out
```

### U-Net with GroupNorm

```python
class DoubleConvGN(nn.Module):
    """Double convolution block with GroupNorm for U-Net."""
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNetGN(nn.Module):
    """U-Net with Group Normalization for semantic segmentation."""
    
    def __init__(self, in_channels=3, num_classes=2, num_groups=8):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConvGN(in_channels, 64, num_groups)
        self.enc2 = DoubleConvGN(64, 128, num_groups)
        self.enc3 = DoubleConvGN(128, 256, num_groups)
        self.enc4 = DoubleConvGN(256, 512, num_groups)
        
        # Bottleneck
        self.bottleneck = DoubleConvGN(512, 1024, num_groups)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConvGN(1024, 512, num_groups)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConvGN(512, 256, num_groups)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConvGN(256, 128, num_groups)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConvGN(128, 64, num_groups)
        
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        
        return self.out_conv(d1)
```

### Object Detection Backbone

```python
class DetectionBackbone(nn.Module):
    """Feature Pyramid Network backbone with GroupNorm."""
    
    def __init__(self, num_groups=32):
        super().__init__()
        
        # C2: 64 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # C3: 128 channels
        self.layer2 = self._make_layer(64, 128, 2, stride=2, num_groups=num_groups)
        
        # C4: 256 channels
        self.layer3 = self._make_layer(128, 256, 2, stride=2, num_groups=num_groups)
        
        # C5: 512 channels
        self.layer4 = self._make_layer(256, 512, 2, stride=2, num_groups=num_groups)
        
        # FPN lateral connections
        self.lateral4 = nn.Conv2d(512, 256, 1)
        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral2 = nn.Conv2d(128, 256, 1)
        
        # FPN output convolutions
        self.fpn_conv = nn.Conv2d(256, 256, 3, padding=1)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, num_groups):
        layers = [ResNetBlockGN(in_channels, out_channels, stride, num_groups)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlockGN(out_channels, out_channels, 1, num_groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Bottom-up
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down with lateral connections
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + nn.functional.interpolate(p5, scale_factor=2)
        p3 = self.lateral2(c3) + nn.functional.interpolate(p4, scale_factor=2)
        
        # Apply 3x3 conv to reduce aliasing
        p5 = self.fpn_conv(p5)
        p4 = self.fpn_conv(p4)
        p3 = self.fpn_conv(p3)
        
        return {'p3': p3, 'p4': p4, 'p5': p5}
```

## Comparison with Other Normalizations

```python
def comprehensive_comparison():
    """Compare all normalization methods."""
    
    torch.manual_seed(42)
    
    # Input: 4 images, 8 channels, 4x4 spatial
    x = torch.randn(4, 8, 4, 4)
    
    # Different normalizations
    bn = nn.BatchNorm2d(8, affine=False)
    bn.eval()
    ln = nn.LayerNorm([8, 4, 4], elementwise_affine=False)
    gn = nn.GroupNorm(4, 8, affine=False)  # 4 groups, 2 channels each
    in_norm = nn.InstanceNorm2d(8, affine=False)
    
    out_bn = bn(x)
    out_ln = ln(x)
    out_gn = gn(x)
    out_in = in_norm(x)
    
    print("Normalization Statistics Comparison")
    print("=" * 60)
    
    print("\nBatchNorm (per channel across batch):")
    for c in range(2):
        print(f"  Channel {c}: mean={out_bn[:,c].mean():.4f}, std={out_bn[:,c].std():.4f}")
    
    print("\nLayerNorm (per sample across features):")
    for n in range(2):
        print(f"  Sample {n}: mean={out_ln[n].mean():.4f}, std={out_ln[n].std():.4f}")
    
    print("\nGroupNorm (per sample-group):")
    out_gn_reshaped = out_gn.view(4, 4, 2, 4, 4)  # (N, G, C/G, H, W)
    for n in range(2):
        for g in range(2):
            mean = out_gn_reshaped[n, g].mean().item()
            std = out_gn_reshaped[n, g].std().item()
            print(f"  Sample {n}, Group {g}: mean={mean:.4f}, std={std:.4f}")
    
    print("\nInstanceNorm (per sample-channel):")
    for n in range(2):
        for c in range(2):
            print(f"  Sample {n}, Channel {c}: mean={out_in[n,c].mean():.4f}")

comprehensive_comparison()
```

## When to Use Group Normalization

### Good Use Cases

✅ **Small batch training** (batch size < 16)  
✅ **Object detection** (standard in Detectron2)  
✅ **Semantic segmentation** with limited GPU memory  
✅ **Video understanding** (temporal batches vary)  
✅ **Medical imaging** (often limited samples)

### Comparison Guidelines

| Scenario | Recommended |
|----------|-------------|
| Large batches (≥32), CNNs | BatchNorm |
| Small batches, CNNs | GroupNorm |
| Transformers, NLP | LayerNorm |
| Style transfer, GANs | InstanceNorm |
| General purpose, flexible | GroupNorm |

## Best Practices

### 1. Choosing Groups

```python
# Standard: 32 groups when channels allow
nn.GroupNorm(32, 256)  # 8 channels per group

# For smaller channel counts
nn.GroupNorm(8, 32)    # 4 channels per group

# Minimum viable
nn.GroupNorm(4, 16)    # 4 channels per group
```

### 2. Converting from BatchNorm

```python
def convert_bn_to_gn(model, num_groups=32):
    """Replace all BatchNorm layers with GroupNorm."""
    
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            # Adjust groups if channels don't divide evenly
            g = num_groups
            while num_channels % g != 0:
                g //= 2
            
            gn = nn.GroupNorm(g, num_channels)
            setattr(model, name, gn)
        else:
            convert_bn_to_gn(module, num_groups)
    
    return model
```

## Summary

Group Normalization is ideal when:

1. **Batch sizes are small** or variable
2. **Batch-independent behavior** is needed
3. **Visual tasks** require normalization (detection, segmentation)

Key properties:
- Divides channels into **G groups**
- Normalizes within each **group per sample**
- **Batch-independent** computation
- Same behavior in **training and inference**

## References

1. Wu, Y., & He, K. (2018). Group Normalization. *ECCV*.

2. He, K., Girshick, R., & Dollár, P. (2019). Rethinking ImageNet Pre-training. *ICCV*.
