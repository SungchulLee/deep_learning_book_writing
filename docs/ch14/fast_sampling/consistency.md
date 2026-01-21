# Consistency Models

## Introduction

**Consistency models** (Song et al., 2023) enable single-step generation by learning to map any point on a trajectory directly to the clean data.

## Key Idea

### Consistency Property

For any point $x_t$ on the probability flow ODE trajectory:

$$f_\theta(x_t, t) = f_\theta(x_s, s) = x_0 \quad \forall t, s$$

The model outputs the same $x_0$ regardless of where on the trajectory you are.

## Training Approaches

### Consistency Distillation

Distill from pre-trained diffusion model:

$$\mathcal{L} = \|f_\theta(x_{t+1}, t+1) - f_{\theta^-}(x_t, t)\|^2$$

### Consistency Training

Train directly without teacher:

$$\mathcal{L} = \|f_\theta(x_{t+\Delta}, t+\Delta) - f_{\theta^-}(x_t, t)\|^2$$

## Implementation

```python
"""
Consistency Model Sampling
==========================
"""

import torch

class ConsistencyModel:
    """Consistency model for single-step generation."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def sample(self, shape, n_steps=1):
        """Generate samples (can use 1 step!)."""
        x = torch.randn(shape, device=self.device)
        
        # Even with n_steps=1, we get good samples
        t = torch.ones(shape[0], device=self.device)
        x = self.model(x, t)
        
        return x
```

## Results

| Model | Steps | FID |
|-------|-------|-----|
| DDPM | 1000 | 2.4 |
| Consistency | 1 | 3.5 |
| Consistency | 2 | 2.9 |

## Summary

Consistency models achieve near-diffusion quality in a single step, making real-time generation practical.

## Navigation

- **Previous**: [Progressive Distillation](distillation.md)
- **Next**: [Classifier Guidance](../conditional/classifier_guidance.md)
