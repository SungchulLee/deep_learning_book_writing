# SPADE: Spatially-Adaptive Normalization

SPADE (Spatially-Adaptive Denormalization), introduced by Park et al. (2019) in "Semantic Image Synthesis with Spatially-Adaptive Normalization," enables high-quality image synthesis from semantic segmentation maps by modulating generator activations spatially based on the input layout.

## Motivation

Standard conditional GANs like Pix2Pix concatenate the semantic map with noise and feed it through the generator. However, normalization layers (BatchNorm, InstanceNorm) tend to wash out semantic information in the early layers. SPADE addresses this by injecting the semantic layout directly into the normalization parameters at every layer.

## Mathematical Formulation

### Standard Batch Normalization

$$\text{BN}(h) = \gamma \cdot \frac{h - \mu}{\sigma} + \beta$$

where $\gamma$ and $\beta$ are learned scalars (per channel).

### SPADE Normalization

$$\text{SPADE}(h, m) = \gamma_{c,y,x}(m) \cdot \frac{h_{n,c,y,x} - \mu_c}{\sigma_c} + \beta_{c,y,x}(m)$$

where $\gamma(m)$ and $\beta(m)$ are **spatially-varying** parameters derived from the segmentation map $m$ via learned convolutions.

Key difference: $\gamma$ and $\beta$ are **functions of the input semantic map** and vary at every spatial location.

## Architecture

### SPADE Normalization Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    """
    Spatially-Adaptive Denormalization.
    
    Learns spatially-varying scale and bias from semantic segmentation map.
    
    Args:
        norm_nc: Number of channels in the normalized activation
        label_nc: Number of semantic classes in the segmentation map
        nhidden: Hidden dimension for the shared convolution
    """
    
    def __init__(self, norm_nc, label_nc, nhidden=128):
        super().__init__()
        
        # Parameter-free normalization
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        # Shared convolution to process segmentation map
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Separate branches for scale (gamma) and bias (beta)
        self.conv_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
    
    def forward(self, x, segmap):
        # Step 1: Normalize activation
        normalized = self.param_free_norm(x)
        
        # Step 2: Resize segmentation map to match activation spatial size
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        
        # Step 3: Produce spatially-varying scale and bias
        actv = self.shared(segmap)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)
        
        # Step 4: Apply denormalization
        return normalized * (1 + gamma) + beta
```

### SPADE Residual Block

```python
class SPADEResBlk(nn.Module):
    """Residual block with SPADE normalization."""
    
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)
    
    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s
    
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg), 0.2))
        
        return x_s + dx
```

### SPADE Generator

```python
class SPADEGenerator(nn.Module):
    """
    SPADE Generator for semantic image synthesis.
    
    Takes a segmentation map and noise vector, produces a photorealistic image.
    The segmentation map is injected at every layer via SPADE normalization.
    """
    
    def __init__(self, label_nc, z_dim=256, ngf=64):
        super().__init__()
        
        self.z_dim = z_dim
        nf = ngf
        
        # Initial projection from noise
        self.fc = nn.Linear(z_dim, 16 * nf * 4 * 4)
        
        # SPADE residual blocks (progressive upsampling)
        self.head = SPADEResBlk(16 * nf, 16 * nf, label_nc)
        self.up_1 = SPADEResBlk(16 * nf, 16 * nf, label_nc)
        self.up_2 = SPADEResBlk(16 * nf, 8 * nf, label_nc)
        self.up_3 = SPADEResBlk(8 * nf, 4 * nf, label_nc)
        self.up_4 = SPADEResBlk(4 * nf, 2 * nf, label_nc)
        self.up_5 = SPADEResBlk(2 * nf, 1 * nf, label_nc)
        
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
    
    def forward(self, segmap, z=None):
        if z is None:
            z = torch.randn(segmap.size(0), self.z_dim, device=segmap.device)
        
        x = self.fc(z).view(-1, 16 * 64, 4, 4)  # 4×4
        
        x = self.head(x, segmap)
        x = self.up(x)                            # 8×8
        x = self.up_1(x, segmap)
        x = self.up(x)                            # 16×16
        x = self.up_2(x, segmap)
        x = self.up(x)                            # 32×32
        x = self.up_3(x, segmap)
        x = self.up(x)                            # 64×64
        x = self.up_4(x, segmap)
        x = self.up(x)                            # 128×128
        x = self.up_5(x, segmap)
        x = self.up(x)                            # 256×256
        
        x = self.conv_img(F.leaky_relu(x, 0.2))
        return torch.tanh(x)
```

## Multi-Scale Discriminator

SPADE uses a multi-scale PatchGAN discriminator that operates at different resolutions:

```python
class MultiscaleDiscriminator(nn.Module):
    """Multi-scale discriminator for SPADE."""
    
    def __init__(self, input_nc, n_scales=2):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(input_nc) for _ in range(n_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(self, x):
        results = []
        for disc in self.discriminators:
            results.append(disc(x))
            x = self.downsample(x)
        return results
```

## Loss Function

SPADE combines multiple losses:

$$\mathcal{L} = \mathcal{L}_{GAN} + \lambda_{FM} \mathcal{L}_{FM} + \lambda_{VGG} \mathcal{L}_{VGG}$$

where:
- $\mathcal{L}_{GAN}$: Hinge loss on multi-scale discriminator
- $\mathcal{L}_{FM}$: Feature matching loss (match intermediate discriminator features)
- $\mathcal{L}_{VGG}$: Perceptual loss (match VGG features for realism)

```python
def spade_generator_loss(D, G, segmap, real_img, lambda_fm=10, lambda_vgg=10):
    fake_img = G(segmap)
    
    # Multi-scale GAN loss (hinge)
    fake_preds = D(torch.cat([segmap, fake_img], dim=1))
    gan_loss = sum(-pred.mean() for pred in fake_preds)
    
    # Feature matching loss
    real_preds = D(torch.cat([segmap, real_img], dim=1))
    fm_loss = 0
    for real_feat, fake_feat in zip(real_preds, fake_preds):
        fm_loss += F.l1_loss(fake_feat, real_feat.detach())
    
    # VGG perceptual loss
    vgg_loss = perceptual_loss(fake_img, real_img)
    
    return gan_loss + lambda_fm * fm_loss + lambda_vgg * vgg_loss
```

## Comparison with Other Conditional Methods

| Method | Conditioning | Normalization | Spatial Control |
|--------|-------------|---------------|-----------------|
| **cGAN** | Concatenation | BatchNorm | Limited |
| **Pix2Pix** | U-Net skip connections | BatchNorm | Moderate |
| **SPADE** | Spatially-adaptive norm | SPADE | Full per-pixel |

## Applications in Quantitative Finance

SPADE's spatially-adaptive conditioning has analogues in financial modeling:

- **Regime-conditional generation**: Different market regimes (bull, bear, crisis) can be encoded as "semantic maps" over time, conditioning the generator to produce scenario-appropriate return distributions at each timestep
- **Sector-specific modeling**: Spatial conditioning can encode sector relationships, allowing generation of correlated but sector-specific return paths

## Summary

| Component | Description |
|-----------|-------------|
| **Core idea** | Inject semantic layout via normalization parameters |
| **Key module** | SPADE: spatially-varying γ and β from segmap |
| **Generator** | SPADE ResBlks with progressive upsampling |
| **Discriminator** | Multi-scale PatchGAN |
| **Losses** | Hinge + Feature matching + VGG perceptual |
| **Resolution** | Up to 512×512 (original paper) |
| **Key advantage** | Preserves semantic information through all layers |

SPADE represents a significant advancement in conditional image synthesis, demonstrating that how conditioning information is injected matters as much as what information is provided.
