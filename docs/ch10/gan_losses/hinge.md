# Hinge Loss

The hinge loss for GANs, popularized by Spectral Normalization GAN (SN-GAN) and BigGAN, provides a simple and effective alternative to cross-entropy and Wasserstein losses.

## Mathematical Formulation

### Discriminator Loss

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{data}}}[\max(0, 1 - D(x))] + \mathbb{E}_{z \sim p_z}[\max(0, 1 + D(G(z)))]$$

The discriminator is penalized when:
- $D(x) < 1$ for real samples (should be ≥ 1)
- $D(G(z)) > -1$ for fake samples (should be ≤ -1)

### Generator Loss

$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[D(G(z))]$$

The generator simply maximizes the critic score on fake samples.

## Intuition

### Margin-Based Classification

Unlike cross-entropy which pushes outputs toward 0 or 1, hinge loss creates a **margin**:

- Real samples: D(x) > 1 (margin of safety)
- Fake samples: D(G(z)) < -1 (margin of safety)

Once samples are correctly classified with sufficient margin, no further gradient is applied.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_hinge_loss():
    """Visualize hinge loss behavior."""
    
    d_output = np.linspace(-3, 3, 1000)
    
    # Hinge loss for real samples: max(0, 1 - D(x))
    hinge_real = np.maximum(0, 1 - d_output)
    
    # Hinge loss for fake samples: max(0, 1 + D(G(z)))
    hinge_fake = np.maximum(0, 1 + d_output)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(d_output, hinge_real, linewidth=2)
    axes[0].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Margin')
    axes[0].fill_between(d_output, hinge_real, alpha=0.3)
    axes[0].set_xlabel('D(x) for real samples')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Hinge Loss: Real Samples')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(d_output, hinge_fake, linewidth=2)
    axes[1].axvline(x=-1, color='red', linestyle='--', alpha=0.5, label='Margin')
    axes[1].fill_between(d_output, hinge_fake, alpha=0.3)
    axes[1].set_xlabel('D(G(z)) for fake samples')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Hinge Loss: Fake Samples')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Implementation

### Hinge Loss Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeGANLoss:
    """
    Hinge loss for GANs.
    
    Used in SN-GAN, BigGAN, and other modern architectures.
    """
    
    def discriminator_loss(self, d_real, d_fake):
        """
        Discriminator hinge loss.
        
        L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
        
        Args:
            d_real: D(x) - discriminator output on real data
            d_fake: D(G(z)) - discriminator output on fake data
        
        Returns:
            loss: Scalar loss
            info: Dict with details
        """
        # Hinge loss for real: max(0, 1 - D(x))
        real_loss = F.relu(1.0 - d_real).mean()
        
        # Hinge loss for fake: max(0, 1 + D(G(z)))
        fake_loss = F.relu(1.0 + d_fake).mean()
        
        total_loss = real_loss + fake_loss
        
        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item(),
            'real_margin_violations': (d_real < 1).float().mean().item(),
            'fake_margin_violations': (d_fake > -1).float().mean().item()
        }
    
    def generator_loss(self, d_fake):
        """
        Generator loss: -E[D(G(z))]
        
        Simple: maximize discriminator output on fake samples.
        """
        return -d_fake.mean()
```

### Alternative Implementation with Margins

```python
class HingeGANLossWithMargin:
    """Hinge loss with configurable margins."""
    
    def __init__(self, margin_real=1.0, margin_fake=1.0):
        """
        Args:
            margin_real: Margin for real samples (D(x) should exceed this)
            margin_fake: Margin for fake samples (D(G(z)) should be below -this)
        """
        self.margin_real = margin_real
        self.margin_fake = margin_fake
    
    def discriminator_loss(self, d_real, d_fake):
        """Discriminator loss with configurable margins."""
        real_loss = F.relu(self.margin_real - d_real).mean()
        fake_loss = F.relu(self.margin_fake + d_fake).mean()
        return real_loss + fake_loss
    
    def generator_loss(self, d_fake):
        """Generator loss."""
        return -d_fake.mean()
```

## Discriminator Architecture

With hinge loss, the discriminator outputs an **unbounded score** (no sigmoid):

```python
class HingeDiscriminator(nn.Module):
    """Discriminator for hinge loss - no sigmoid output."""
    
    def __init__(self, image_channels=1, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0),
            # NO SIGMOID - output is unbounded score
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
```

## Training Loop

```python
def train_hinge_gan(G, D, dataloader, config, device):
    """Train GAN with hinge loss."""
    
    loss_fn = HingeGANLoss()
    
    g_optimizer = torch.optim.Adam(G.parameters(), lr=config['g_lr'], betas=(0.0, 0.9))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config['d_lr'], betas=(0.0, 0.9))
    
    for epoch in range(config['n_epochs']):
        for i, (real_data, _) in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # ==================
            # Train Discriminator
            # ==================
            d_optimizer.zero_grad()
            
            # Real data
            d_real = D(real_data)
            
            # Fake data
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_data = G(z)
            d_fake = D(fake_data.detach())
            
            d_loss, d_info = loss_fn.discriminator_loss(d_real, d_fake)
            d_loss.backward()
            d_optimizer.step()
            
            # ===============
            # Train Generator
            # ===============
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_data = G(z)
            d_fake = D(fake_data)
            
            g_loss = loss_fn.generator_loss(d_fake)
            g_loss.backward()
            g_optimizer.step()
```

## Comparison with Other Losses

### Gradient Behavior

```python
def compare_d_gradients():
    """Compare discriminator gradients across loss functions."""
    
    d_real = np.linspace(-2, 3, 1000)
    
    # BCE gradient for real: -1/D(x)
    # (approximation, actual depends on sigmoid)
    bce_grad = -1 / (1e-8 + 1 / (1 + np.exp(-d_real)))
    
    # Hinge gradient for real: -1 if D(x) < 1, else 0
    hinge_grad = np.where(d_real < 1, -1, 0)
    
    # Wasserstein gradient: constant -1
    wgan_grad = np.ones_like(d_real) * -1
    
    plt.figure(figsize=(10, 5))
    plt.plot(d_real, bce_grad, label='BCE', alpha=0.7)
    plt.plot(d_real, hinge_grad, label='Hinge', alpha=0.7, linewidth=2)
    plt.plot(d_real, wgan_grad, label='Wasserstein', alpha=0.7, linestyle='--')
    plt.xlabel('D(x) for real samples')
    plt.ylabel('Gradient')
    plt.title('Discriminator Gradient Comparison (Real Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 1)
```

### Loss Comparison Table

| Loss | D(real) Target | D(fake) Target | Gradient Behavior |
|------|----------------|----------------|-------------------|
| BCE | → 1 (sigmoid) | → 0 (sigmoid) | Continuous, can saturate |
| Wasserstein | Maximize | Minimize | Constant (with GP/SN) |
| Hinge | ≥ 1 | ≤ -1 | Piecewise: active below margin |

## Advantages

### 1. Stable Training

- No saturation issues
- Bounded gradient magnitude
- Works well with spectral normalization

### 2. Margin-Based Learning

- D doesn't waste capacity on already-classified samples
- Focuses on samples near decision boundary
- Can improve sample diversity

### 3. Simple Implementation

- Easy to understand and implement
- No log computations (numerically stable)
- Same generator loss as WGAN

## Disadvantages

### 1. Less Theoretical Foundation

- Not derived from specific divergence
- Harder to interpret loss values

### 2. May Undertrain Discriminator

- Samples past margin receive zero gradient
- D may not learn fine distinctions

## Used In Major Architectures

The hinge loss is the default in many state-of-the-art GANs:

1. **SN-GAN** (Spectral Normalization GAN)
2. **BigGAN** (Large-scale image generation)
3. **StyleGAN** variants (face generation)

## Hinge Loss with Spectral Normalization

The most common combination:

```python
from torch.nn.utils import spectral_norm

class SNHingeDiscriminator(nn.Module):
    """Spectrally normalized discriminator with hinge loss."""
    
    def __init__(self, image_channels=3, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(image_channels, feature_maps, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps * 8, 1, 4, 1, 0)),
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
```

## Summary

| Aspect | Hinge Loss |
|--------|------------|
| **D Loss** | E[max(0, 1-D(x))] + E[max(0, 1+D(G(z)))] |
| **G Loss** | -E[D(G(z))] |
| **D Output** | Unbounded score |
| **Margin** | Real ≥ 1, Fake ≤ -1 |
| **Best Paired With** | Spectral Normalization |
| **Used In** | SN-GAN, BigGAN, StyleGAN |

The hinge loss has become a standard choice for modern GANs due to its simplicity, stability, and excellent performance when combined with spectral normalization.
