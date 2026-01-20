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
