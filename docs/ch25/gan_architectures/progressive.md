# Progressive GAN

Progressive GAN (ProGAN), introduced by Karras et al. (2018), revolutionized high-resolution image generation by training from low to high resolution progressively.

## Core Idea

Instead of training a GAN directly at high resolution (which is unstable), start at low resolution and progressively add layers:

```
Training Phases:
4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128 → 256×256 → 512×512 → 1024×1024
```

## Architecture

### Progressive Growing

```python
class ProgressiveGenerator(nn.Module):
    """Generator that grows progressively."""
    
    def __init__(self, latent_dim=512, max_resolution=1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial 4x4 block
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Progressive blocks (added during training)
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        # Initial to_rgb
        self.to_rgb.append(nn.Conv2d(512, 3, 1))
        
    def add_block(self, in_ch, out_ch):
        """Add new resolution block."""
        block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.blocks.append(block)
        self.to_rgb.append(nn.Conv2d(out_ch, 3, 1))
```

### Smooth Fade-In

New layers are faded in smoothly using α blending:

$$\text{output} = (1 - \alpha) \cdot \text{upsampled\_old} + \alpha \cdot \text{new\_block}$$

```python
def forward_with_fade(self, x, alpha, current_depth):
    """Forward pass with fade-in for new block."""
    # Process through existing blocks
    for i, block in enumerate(self.blocks[:current_depth-1]):
        x = block(x)
    
    if current_depth > 1 and alpha < 1:
        # Blend old and new paths
        old_path = F.interpolate(self.to_rgb[current_depth-2](x), scale_factor=2)
        x = self.blocks[current_depth-1](x)
        new_path = self.to_rgb[current_depth-1](x)
        return (1 - alpha) * old_path + alpha * new_path
    else:
        x = self.blocks[current_depth-1](x) if current_depth > 0 else x
        return self.to_rgb[current_depth-1](x)
```

## Key Innovations

### 1. Minibatch Standard Deviation

Encourages diversity by appending batch statistics:

```python
class MinibatchStdDev(nn.Module):
    def forward(self, x):
        std = x.std(dim=0).mean()
        std_map = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_map], dim=1)
```

### 2. Equalized Learning Rate

Scale weights at runtime instead of careful initialization:

```python
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = (2 / in_features) ** 0.5
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)
```

### 3. Pixelwise Feature Normalization

Normalize features per pixel (instead of batch):

```python
class PixelNorm(nn.Module):
    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
```

## Training Schedule

| Resolution | Training Images |
|------------|-----------------|
| 4×4 | 800K |
| 8×8 | 800K (fade) + 800K (stable) |
| 16×16 | 800K + 800K |
| ... | ... |
| 1024×1024 | 800K + 800K |

## Results

- First GAN to generate 1024×1024 faces
- High-quality, diverse outputs
- FID score of 8.04 on CelebA-HQ

## Summary

| Innovation | Purpose |
|------------|---------|
| Progressive growing | Stable high-res training |
| Smooth fade-in | Avoid sudden changes |
| Minibatch std | Encourage diversity |
| Equalized LR | Stable learning |
| Pixel norm | Replace batch norm |
