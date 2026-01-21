# Sliced Score Matching

## Introduction

**Sliced Score Matching (SSM)** is an alternative to explicit score matching that avoids computing full Jacobians by using random projections. It provides a scalable way to train score networks when denoising score matching is not applicable.

!!! info "Core Idea"
    Instead of computing the full Jacobian trace, project both the model and data scores onto random directions and match these 1D projections.

## The Sliced Score Matching Objective

### Derivation

Starting from the explicit score matching objective:

$$\mathcal{L}_{\text{ESM}} = \mathbb{E}_p\left[\frac{1}{2}\|s_\theta(x)\|^2 + \text{tr}(\nabla_x s_\theta(x))\right]$$

The expensive term is $\text{tr}(\nabla_x s_\theta(x))$. Using random projections:

$$\text{tr}(A) = \mathbb{E}_{v}[v^\top A v]$$

where $v$ is drawn from a distribution with $\mathbb{E}[vv^\top] = I$.

### SSM Objective

$$\mathcal{L}_{\text{SSM}} = \mathbb{E}_{p(x), p(v)}\left[\frac{1}{2}(v^\top s_\theta(x))^2 + v^\top \nabla_x s_\theta(x) \cdot v\right]$$

The key observation: $v^\top \nabla_x s_\theta(x) \cdot v$ is a directional derivative, computable with a single backward pass!

## PyTorch Implementation

```python
"""
Sliced Score Matching Implementation
====================================
"""

import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Literal


def sample_rademacher(shape: tuple, device: torch.device) -> torch.Tensor:
    """Sample Rademacher random vectors (+1 or -1)."""
    return torch.randint(0, 2, shape, device=device).float() * 2 - 1


def ssm_loss(
    score_net: nn.Module,
    x: torch.Tensor,
    n_projections: int = 1,
    projection_type: Literal['rademacher', 'gaussian'] = 'rademacher'
) -> torch.Tensor:
    """
    Compute sliced score matching loss.
    
    L = E_x,v [(v·s(x))²/2 + v·∇_x(v·s(x))]
    """
    x = x.requires_grad_(True)
    batch_size, dim = x.shape
    device = x.device
    
    if projection_type == 'rademacher':
        v = sample_rademacher((n_projections, batch_size, dim), device)
    else:
        v = torch.randn(n_projections, batch_size, dim, device=device)
    
    total_loss = 0.0
    
    for i in range(n_projections):
        vi = v[i]
        score = score_net(x)
        
        # Term 1: (v·s(x))² / 2
        score_proj = (score * vi).sum(dim=1)
        term1 = 0.5 * (score_proj ** 2)
        
        # Term 2: v·∇_x(v·s(x)) - directional derivative
        grad_score_proj = grad(
            score_proj.sum(), x,
            create_graph=True, retain_graph=True
        )[0]
        term2 = (grad_score_proj * vi).sum(dim=1)
        
        total_loss += (term1 + term2).mean()
    
    return total_loss / n_projections


class SlicedScoreMatchingTrainer:
    """Trainer for sliced score matching."""
    
    def __init__(
        self,
        score_net: nn.Module,
        lr: float = 1e-3,
        n_projections: int = 1
    ):
        self.score_net = score_net
        self.optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
        self.n_projections = n_projections
        
    def train_step(self, x: torch.Tensor) -> dict:
        self.optimizer.zero_grad()
        loss = ssm_loss(self.score_net, x, self.n_projections)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
```

## Comparison with Other Methods

| Method | Jacobian Cost | Random Vectors | Noise Required |
|--------|---------------|----------------|----------------|
| Explicit SM | $O(d)$ backward | No | No |
| Sliced SM | $O(k)$ backward | Yes ($k$ vectors) | No |
| Denoising SM | $O(1)$ backward | No | Yes |

## When to Use SSM

**Use SSM when:**
- You cannot add noise to data (DSM not applicable)
- Dimension is moderate (10-100)
- You need exact score matching guarantees

**Prefer DSM when:**
- High-dimensional data (images, audio)
- Noise perturbation is acceptable
- Training efficiency is critical

## Summary

Sliced score matching provides a middle ground between expensive explicit score matching and noise-dependent denoising score matching. It's particularly useful when exact score matching is needed without data augmentation.

## References

1. Song et al. (2019). Sliced Score Matching: A Scalable Approach to Density and Score Estimation.
2. Hutchinson (1989). A Stochastic Estimator of the Trace of the Influence Matrix.

## Navigation

- **Previous**: [Denoising Score Matching](denoising_score_matching.md)
- **Next**: [Noise Conditional Score Networks](ncsn.md)
