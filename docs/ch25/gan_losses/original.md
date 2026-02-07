# Original GAN Loss

The original GAN loss, introduced by Goodfellow et al. in 2014, formulates generative modeling as a minimax game with binary cross-entropy classification.

## Mathematical Formulation

### Value Function

The original GAN objective is:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### Discriminator Loss

The discriminator maximizes $V(D, G)$, equivalent to minimizing:

$$\mathcal{L}_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

This is **binary cross-entropy** with:
- Real samples labeled as 1
- Fake samples labeled as 0

### Generator Loss

The generator minimizes $V(D, G)$:

$$\mathcal{L}_G = \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

The generator wants $D(G(z)) \to 1$, so $\log(1 - D(G(z))) \to -\infty$.

## Implementation

```python
import torch
import torch.nn as nn

class OriginalGANLoss:
    """Original GAN loss (Goodfellow et al., 2014)."""
    
    def __init__(self):
        self.criterion = nn.BCELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        """
        Discriminator loss: -E[log D(x)] - E[log(1 - D(G(z)))]
        """
        batch_size_real = d_real.size(0)
        batch_size_fake = d_fake.size(0)
        
        real_labels = torch.ones(batch_size_real, 1, device=d_real.device)
        fake_labels = torch.zeros(batch_size_fake, 1, device=d_fake.device)
        
        real_loss = self.criterion(d_real, real_labels)
        fake_loss = self.criterion(d_fake, fake_labels)
        
        return real_loss + fake_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item()
        }
    
    def generator_loss(self, d_fake):
        """
        Original generator loss: E[log(1 - D(G(z)))]
        """
        batch_size = d_fake.size(0)
        fake_labels = torch.zeros(batch_size, 1, device=d_fake.device)
        return -self.criterion(d_fake, fake_labels)
```

## Gradient Analysis

### The Saturation Problem

When D is confident on fake data ($D(G(z)) \approx 0$):

$$\nabla_{\theta_G} \mathcal{L}_G = -\mathbb{E}_z\left[\frac{\nabla_{\theta_G} D(G(z))}{1 - D(G(z))}\right]$$

The denominator $1 - D(G(z)) \approx 1$, giving small gradients when G needs strong learning signal.

## Theoretical Properties

### Optimal Discriminator

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

### Jensen-Shannon Divergence

With optimal D, the generator minimizes:

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

## Limitations

1. **Gradient saturation** - Weak G gradients early in training
2. **Mode collapse** - JSD can plateau when distributions don't overlap
3. **Training instability** - Requires careful hyperparameter tuning

## Summary

| Aspect | Original GAN Loss |
|--------|-------------------|
| D Loss | BCE (real=1, fake=0) |
| G Loss | min log(1 - D(G(z))) |
| Divergence | Jensen-Shannon |
| Issue | Gradient saturation |

---

# Non-Saturating Loss

The non-saturating loss is a practical modification to the original GAN generator loss that provides stronger gradients early in training when the generator needs them most.

## Motivation

### The Saturation Problem

In the original GAN, the generator minimizes:

$$\mathcal{L}_G^{\text{original}} = \mathbb{E}_z[\log(1 - D(G(z)))]$$

Early in training when $D(G(z)) \approx 0$:
- $\log(1 - D(G(z))) \approx \log(1) = 0$
- Gradient $\frac{-1}{1-D(G(z))} \approx -1$ (bounded)

The gradient is weak precisely when G is poor and needs strong learning signal.

### The Solution

Instead of minimizing $\log(1 - D(G(z)))$, maximize $\log D(G(z))$:

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_z[\log D(G(z))]$$

When $D(G(z)) \approx 0$:
- Gradient $\frac{-1}{D(G(z))} \to -\infty$

Strong gradient signal when G needs improvement!

## Mathematical Comparison

### Original Loss

$$\mathcal{L}_G^{\text{original}} = \mathbb{E}_z[\log(1 - D(G(z)))]$$

Gradient with respect to $D(G(z))$:
$$\frac{\partial \mathcal{L}_G^{\text{original}}}{\partial D(G(z))} = \frac{-1}{1 - D(G(z))}$$

### Non-Saturating Loss

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_z[\log D(G(z))]$$

Gradient with respect to $D(G(z))$:
$$\frac{\partial \mathcal{L}_G^{\text{NS}}}{\partial D(G(z))} = \frac{-1}{D(G(z))}$$

## Gradient Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_gradients():
    """Compare gradient magnitudes of original vs non-saturating loss."""
    
    d_gz = np.linspace(0.001, 0.999, 1000)
    
    # Original: gradient is -1/(1-D)
    grad_original = -1 / (1 - d_gz)
    
    # Non-saturating: gradient is -1/D
    grad_ns = -1 / d_gz
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Losses
    loss_original = np.log(1 - d_gz)
    loss_ns = -np.log(d_gz)
    
    axes[0].plot(d_gz, loss_original, label='Original: log(1-D)')
    axes[0].plot(d_gz, loss_ns, label='Non-saturating: -log(D)')
    axes[0].set_xlabel('D(G(z))')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss Functions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 5)
    
    # Gradient magnitudes
    axes[1].plot(d_gz, np.abs(grad_original), label='|∂(log(1-D))/∂D|')
    axes[1].plot(d_gz, np.abs(grad_ns), label='|∂(-log D)/∂D|')
    axes[1].set_xlabel('D(G(z))')
    axes[1].set_ylabel('Gradient Magnitude')
    axes[1].set_title('Gradient Magnitudes')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Gradient ratio
    ratio = np.abs(grad_ns) / np.abs(grad_original)
    axes[2].plot(d_gz, ratio)
    axes[2].set_xlabel('D(G(z))')
    axes[2].set_ylabel('|NS gradient| / |Original gradient|')
    axes[2].set_title('Gradient Ratio (NS / Original)')
    axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Implementation

### Loss Class

```python
import torch
import torch.nn as nn

class NonSaturatingGANLoss:
    """
    Non-saturating GAN loss.
    
    Generator maximizes log(D(G(z))) instead of minimizing log(1 - D(G(z))).
    Provides stronger gradients when G is poor.
    """
    
    def __init__(self):
        self.criterion = nn.BCELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        """
        Standard discriminator loss (same as original GAN).
        
        L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
        """
        batch_real = d_real.size(0)
        batch_fake = d_fake.size(0)
        
        real_labels = torch.ones(batch_real, 1, device=d_real.device)
        fake_labels = torch.zeros(batch_fake, 1, device=d_fake.device)
        
        real_loss = self.criterion(d_real, real_labels)
        fake_loss = self.criterion(d_fake, fake_labels)
        
        total_loss = real_loss + fake_loss
        
        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item()
        }
    
    def generator_loss(self, d_fake):
        """
        Non-saturating generator loss.
        
        L_G = -E[log D(G(z))]
        
        Instead of minimizing log(1-D), we minimize -log(D).
        This is implemented as BCE with real labels.
        """
        batch_size = d_fake.size(0)
        
        # Treat fake samples as real for generator training
        real_labels = torch.ones(batch_size, 1, device=d_fake.device)
        
        # BCE(D(G(z)), 1) = -log(D(G(z)))
        loss = self.criterion(d_fake, real_labels)
        
        return loss
```

### Alternative Implementation (Without BCE)

```python
class NonSaturatingGANLossManual:
    """Non-saturating loss with explicit log computation."""
    
    def discriminator_loss(self, d_real, d_fake):
        """D loss: -E[log D(x)] - E[log(1 - D(G(z)))]"""
        eps = 1e-8  # For numerical stability
        
        real_loss = -torch.log(d_real + eps).mean()
        fake_loss = -torch.log(1 - d_fake + eps).mean()
        
        return real_loss + fake_loss, {
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item()
        }
    
    def generator_loss(self, d_fake):
        """G loss: -E[log D(G(z))]"""
        eps = 1e-8
        return -torch.log(d_fake + eps).mean()
```

## Training with Non-Saturating Loss

```python
def train_step_nonsaturating(G, D, real_data, latent_dim, 
                             g_optimizer, d_optimizer, device):
    """Training step using non-saturating loss."""
    
    loss_fn = NonSaturatingGANLoss()
    batch_size = real_data.size(0)
    
    # ==================
    # Train Discriminator
    # ==================
    d_optimizer.zero_grad()
    
    # Real data
    d_real = D(real_data)
    
    # Fake data
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_data = G(z)
    d_fake = D(fake_data.detach())
    
    d_loss, d_info = loss_fn.discriminator_loss(d_real, d_fake)
    d_loss.backward()
    d_optimizer.step()
    
    # ===============
    # Train Generator (Non-Saturating)
    # ===============
    g_optimizer.zero_grad()
    
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_data = G(z)
    d_fake = D(fake_data)
    
    g_loss = loss_fn.generator_loss(d_fake)
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss.item(), d_loss.item(), d_info
```

## Theoretical Analysis

### Same Fixed Point

Both losses have the same optimal generator:

**Original**: $\min_G \mathbb{E}_z[\log(1 - D^*(G(z)))]$

**Non-saturating**: $\min_G -\mathbb{E}_z[\log D^*(G(z))]$

At optimum, $p_g = p_{\text{data}}$ and $D^*(x) = 0.5$ for both.

### Different Optimization Dynamics

The gradient dynamics differ:

| D(G(z)) | Original Gradient | NS Gradient |
|---------|-------------------|-------------|
| 0.01 | -1.01 | -100 |
| 0.1 | -1.11 | -10 |
| 0.5 | -2.0 | -2.0 |
| 0.9 | -10 | -1.11 |
| 0.99 | -100 | -1.01 |

At crossover point $D(G(z)) = 0.5$, both have equal gradient magnitude.

### Implicit Divergence

The non-saturating loss minimizes a different divergence:

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_{x \sim p_g}[\log D^*(x)]$$

With optimal discriminator:

$$= -\mathbb{E}_{x \sim p_g}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right]$$

This is related to the **reverse KL divergence** rather than JSD.

## Advantages

### 1. Stronger Early Gradients

When G is poor and $D(G(z)) \approx 0$:
- Original: gradient $\approx -1$
- Non-saturating: gradient $\to -\infty$

### 2. More Stable Training

Empirically, non-saturating loss leads to:
- Faster initial learning
- More consistent training dynamics
- Better final sample quality

### 3. Simple Implementation

Just change the labels in BCE:

```python
# Original: minimize BCE(d_fake, 0) = -log(1 - D(G(z)))
# NS: minimize BCE(d_fake, 1) = -log(D(G(z)))
```

## Limitations

### 1. Can Still Collapse

Mode collapse is not prevented by non-saturating loss alone.

### 2. Different Divergence

Minimizes reverse KL, which has different mode-seeking behavior:
- KL(p_data || p_g): mode-covering (VAE-like)
- KL(p_g || p_data): mode-seeking (GAN-like)

### 3. Not a Zero-Sum Game

With non-saturating loss, G and D don't play a zero-sum game:
- D still maximizes original objective
- G maximizes different objective

This can affect convergence analysis.

## Practical Recommendations

```python
# Standard GAN training uses non-saturating loss by default
def recommended_generator_loss(discriminator, fake_data):
    """The recommended way to train GANs."""
    d_fake = discriminator(fake_data)
    
    # Non-saturating loss
    real_labels = torch.ones_like(d_fake)
    loss = nn.BCELoss()(d_fake, real_labels)
    
    return loss
```

## Summary

| Aspect | Original Loss | Non-Saturating Loss |
|--------|---------------|---------------------|
| **G Objective** | min E[log(1-D)] | min -E[log D] |
| **Early Gradients** | Weak | Strong |
| **Divergence** | JSD | Related to reverse KL |
| **Zero-Sum Game** | Yes | No |
| **Usage** | Theoretical | Practical (default) |

The non-saturating loss is the **standard choice** for GAN training. It maintains the same fixed point as the original loss while providing more practical optimization dynamics.
