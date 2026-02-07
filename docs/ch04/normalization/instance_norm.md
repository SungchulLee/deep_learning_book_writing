# Instance Normalization

## Overview

Instance Normalization (InstanceNorm), introduced by Ulyanov et al. in 2016, normalizes each sample and each channel independently over the spatial dimensions. Originally developed for fast neural style transfer, it has become essential for generative models, particularly GANs and image-to-image translation tasks.

## Mathematical Formulation

### Forward Pass

For an input tensor $x \in \mathbb{R}^{N \times C \times H \times W}$, Instance Normalization computes statistics **per sample, per channel**:

**For each sample $n$ and channel $c$:**

$$\mu_{n,c} = \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w}$$

$$\sigma^2_{n,c} = \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{n,c,h,w} - \mu_{n,c})^2$$

**Normalize:**

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma^2_{n,c} + \epsilon}}$$

**Scale and shift (optional):**

$$y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$$

Where $\gamma, \beta \in \mathbb{R}^{C}$ are learnable per-channel parameters and $\epsilon$ is a small constant for numerical stability.

### Comparison of Normalization Axes

```
Input shape: (N, C, H, W) where:
N = batch size, C = channels, H = height, W = width

Method          | Normalized over | Statistics per
----------------|-----------------|----------------
Batch Norm      | (N, H, W)       | C               (one per channel)
Layer Norm      | (C, H, W)       | N               (one per sample)
Instance Norm   | (H, W)          | N × C           (one per sample-channel)
Group Norm      | (H, W, C/G)     | N × G           (one per sample-group)
```

## PyTorch Implementation

### From Scratch

```python
import torch
import torch.nn as nn

class InstanceNorm2dFromScratch(nn.Module):
    """
    Instance Normalization implementation from scratch.
    Normalizes each instance and channel independently over spatial dims.
    """
    
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, H, W)
        
        Returns:
            Normalized tensor of same shape
        """
        N, C, H, W = x.shape
        
        # Compute mean and variance over spatial dimensions (H, W)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.affine:
            gamma = self.gamma.view(1, C, 1, 1)
            beta = self.beta.view(1, C, 1, 1)
            x_norm = gamma * x_norm + beta
        
        return x_norm
```

### Using PyTorch Built-in

```python
import torch.nn as nn

# For 2D data (images): (N, C, H, W)
instance_norm_2d = nn.InstanceNorm2d(num_features=64)

# For 1D data (sequences): (N, C, L)
instance_norm_1d = nn.InstanceNorm1d(num_features=64)

# Common parameters
instance_norm = nn.InstanceNorm2d(
    num_features=64,
    eps=1e-5,
    affine=True,                # Learnable gamma/beta
    track_running_stats=False   # Usually False
)
```

## Why Instance Normalization for Style Transfer?

### The Key Insight

Instance Normalization removes **instance-specific contrast information**, which is crucial for style transfer:

```python
def demonstrate_style_transfer_motivation():
    """Show why InstanceNorm is essential for style transfer."""
    
    torch.manual_seed(42)
    
    # Simulate two images with different contrast
    bright_image = torch.randn(1, 3, 64, 64) + 3.0  # High mean
    dark_image = torch.randn(1, 3, 64, 64) - 2.0    # Low mean
    
    print("Original images:")
    print(f"  Bright: mean={bright_image.mean():.2f}")
    print(f"  Dark:   mean={dark_image.mean():.2f}")
    
    # With Batch Normalization (mixes statistics!)
    bn = nn.BatchNorm2d(3, affine=False)
    bn.eval()
    combined = torch.cat([bright_image, dark_image], dim=0)
    bn_out = bn(combined)
    
    print("\nWith BatchNorm (problematic):")
    print(f"  Bright: mean={bn_out[0].mean():.2f}")
    print(f"  Dark:   mean={bn_out[1].mean():.2f}")
    print("  → Statistics mixed!")
    
    # With Instance Normalization
    instance_norm = nn.InstanceNorm2d(3, affine=False)
    in_bright = instance_norm(bright_image)
    in_dark = instance_norm(dark_image)
    
    print("\nWith InstanceNorm (correct):")
    print(f"  Bright: mean={in_bright.mean():.4f}")
    print(f"  Dark:   mean={in_dark.mean():.4f}")
    print("  → Each normalized independently!")
```

## Network Architectures

### Style Transfer Network

```python
class ResidualBlock(nn.Module):
    """Residual block with Instance Normalization."""
    
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )
    
    def forward(self, x):
        return x + self.block(x)


class StyleTransferNetwork(nn.Module):
    """Fast neural style transfer network."""
    
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, padding=4, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.residual = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x
```

### CycleGAN Generator

```python
class CycleGANGenerator(nn.Module):
    """CycleGAN generator with Instance Normalization."""
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 4
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3,
                                  stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2, affine=True),
                nn.ReLU(inplace=True)
            ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
```

## When to Use Instance Normalization

### Good Use Cases

✅ **Style transfer** (fast neural style transfer)  
✅ **Image-to-image translation** (CycleGAN, Pix2Pix)  
✅ **GANs** where samples should be independent  
✅ **Domain adaptation** tasks

### Avoid When

❌ **Image classification** (BatchNorm is better)  
❌ **NLP tasks** (LayerNorm is standard)  
❌ **Semantic segmentation** (GroupNorm often preferred)

## Summary

Instance Normalization is essential for:

1. **Style transfer** - removes instance-specific contrast
2. **GANs** - independent sample processing
3. **Image translation** - domain adaptation

Key properties:
- Normalizes over **spatial dimensions** (H, W)
- Statistics computed **per sample, per channel**
- **Batch-independent** computation
- Same behavior in **training and inference**

## References

1. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). Instance Normalization: The Missing Ingredient for Fast Stylization. *arXiv preprint arXiv:1607.08022*.

2. Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *ICCV*.
