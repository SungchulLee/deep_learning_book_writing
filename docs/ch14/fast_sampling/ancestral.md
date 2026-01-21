# Ancestral Sampling

## Introduction

**Ancestral sampling** refers to the original stochastic DDPM sampling and its variants. It maintains randomness at each step, which can improve sample diversity.

## Comparison with DDIM

| Aspect | Ancestral | DDIM |
|--------|-----------|------|
| Randomness | Every step | Optional (η) |
| Diversity | Higher | Lower |
| Consistency | Different each run | Reproducible |
| Quality | Slightly better | Slightly worse |

## Implementation

```python
"""
Ancestral Sampling
==================
"""

import torch
import torch.nn as nn


class AncestralSampler:
    """Ancestral (DDPM-style) sampler."""
    
    def __init__(self, model, alphas_cumprod, device):
        self.model = model
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.device = device
    
    @torch.no_grad()
    def sample_step(self, x_t, t, t_prev):
        """Single ancestral sampling step."""
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        eps_pred = self.model(x_t, t_tensor)
        
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
        
        # Ancestral update (always adds noise)
        sigma_t = torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        
        mean = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev - sigma_t**2) * eps_pred
        
        noise = torch.randn_like(x_t) if t_prev > 0 else 0
        return mean + sigma_t * noise
```

## Summary

Ancestral sampling is the original DDPM approach with stochasticity at each step, providing diverse but less reproducible samples.

## Navigation

- **Previous**: [DDIM](ddim.md)
- **Next**: [Progressive Distillation](distillation.md)
