# Spectral Normalization

Spectral normalization constrains the discriminator's Lipschitz constant by normalizing weight matrices by their spectral norm (largest singular value).

## Motivation

### Why Lipschitz Constraint?

For stable GAN training, the discriminator should be Lipschitz continuous:

$$\|D(x_1) - D(x_2)\| \leq K \|x_1 - x_2\|$$

This prevents D from producing extreme values and vanishing/exploding gradients.

### Spectral Norm Definition

The spectral norm of a matrix W is its largest singular value:

$$\sigma(W) = \max_{h \neq 0} \frac{\|Wh\|}{\|h\|} = \sigma_{\max}(W)$$

## Implementation

### Power Iteration Method

Compute spectral norm efficiently without full SVD:

```python
class SpectralNorm(nn.Module):
    def __init__(self, module, n_power_iterations=1):
        super().__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations
        
        # Initialize u and v vectors
        w = module.weight.data
        h, w_size = w.size(0), w.view(w.size(0), -1).size(1)
        
        self.register_buffer('u', torch.randn(h).normal_(0, 1))
        self.register_buffer('v', torch.randn(w_size).normal_(0, 1))
    
    def _power_iteration(self, w):
        """Estimate spectral norm via power iteration."""
        u = self.u
        v = self.v
        
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(w.t(), u), dim=0)
            u = F.normalize(torch.mv(w, v), dim=0)
        
        sigma = torch.dot(u, torch.mv(w, v))
        return sigma, u, v
    
    def forward(self, *args, **kwargs):
        w = self.module.weight
        w_mat = w.view(w.size(0), -1)
        
        sigma, u, v = self._power_iteration(w_mat)
        
        # Update buffers
        self.u = u.detach()
        self.v = v.detach()
        
        # Normalize weight
        self.module.weight.data = w / sigma
        
        return self.module(*args, **kwargs)
```

### PyTorch Built-in

```python
from torch.nn.utils import spectral_norm

# Apply to a layer
layer = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))

# Apply to entire model
def apply_spectral_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            spectral_norm(module)
```

## Spectrally Normalized Discriminator

```python
class SNDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, feature_maps, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(feature_maps*4, 1, 4, 1, 0)),
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
```

## Benefits

| Benefit | Description |
|---------|-------------|
| Stability | Bounds gradient magnitude |
| No hyperparameter | Unlike WGAN-GP's λ |
| Efficiency | Single power iteration per step |
| Quality | Often better FID scores |

## Comparison with Other Methods

| Method | Lipschitz Enforcement |
|--------|----------------------|
| Weight clipping | Crude, reduces capacity |
| Gradient penalty | Computationally expensive |
| Spectral norm | Efficient, principled |

## When to Use

- **Always in D**: Stabilizes training
- **Sometimes in G**: Can help with BigGAN-style models
- **With any loss**: Works with BCE, hinge, Wasserstein

## Summary

Spectral normalization is a simple, efficient, and effective technique for stabilizing GAN training by controlling the discriminator's Lipschitz constant.
