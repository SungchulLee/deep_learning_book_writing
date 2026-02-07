# Score-Based Attacks

## Introduction

**Score-based attacks** exploit access to the model's output probability distribution $p(y|\mathbf{x})$ or logits to construct adversarial examples without gradient access. These attacks occupy a middle ground between white-box (full gradient access) and decision-based (hard labels only) methods, offering significantly better query efficiency than decision-based approaches.

## Mathematical Framework

### Gradient Estimation with Scores

Given access to the loss function evaluated via model scores, the gradient can be estimated using various sampling strategies.

**Coordinate-wise estimation:**

$$
\hat{g}_i = \frac{\mathcal{L}(f(\mathbf{x} + h\mathbf{e}_i)) - \mathcal{L}(f(\mathbf{x} - h\mathbf{e}_i))}{2h}
$$

This requires $2d$ queries for a complete gradient estimate in $d$ dimensions.

**Random direction estimation:**

$$
\hat{\nabla} \mathcal{L} \approx \frac{d}{n\sigma} \sum_{i=1}^n \left[\mathcal{L}(f(\mathbf{x} + \sigma \mathbf{u}_i)) - \mathcal{L}(f(\mathbf{x}))\right] \mathbf{u}_i
$$

with $\mathbf{u}_i$ drawn from the unit sphere. This requires only $n$ queries regardless of dimension.

### Bandit Optimization with Priors

Ilyas et al. (2019) improved query efficiency by incorporating **data-dependent priors** into gradient estimation. The key insight is that adversarial perturbations often have spatial structure (nearby pixels tend to be perturbed similarly).

The gradient estimate uses a time-dependent prior $\mathbf{p}^{(t)}$:

$$
\hat{\nabla} \mathcal{L} \approx \frac{1}{\sigma}\left[\mathcal{L}(f(\mathbf{x} + \sigma \mathbf{q})) - \mathcal{L}(f(\mathbf{x}))\right] \mathbf{q}
$$

where $\mathbf{q} = \beta \mathbf{p}^{(t)} + (1-\beta)\mathbf{u}$ blends the prior with a random direction, and the prior is updated based on successful attack directions.

## Square Attack

**Square Attack** (Andriushchenko et al., 2020) is a score-based black-box attack that avoids gradient estimation entirely, using a **random search** strategy with localized square-shaped updates.

### Algorithm

At each iteration:

1. Select a random square region of the input
2. Sample a perturbation for that region
3. Accept the update only if it increases the loss

The square size starts large and decreases over iterations, implementing a coarse-to-fine search strategy.

### Key Properties

- **No gradient estimation**: Avoids the overhead of sampling-based gradient methods
- **Highly query-efficient**: Often requires only 1,000-5,000 queries
- **Localized updates**: Exploits spatial structure of images
- **Component of AutoAttack**: Serves as the black-box component

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class SimpleSquareAttack:
    """
    Simplified Square Attack for score-based black-box setting.
    
    Uses random square-shaped perturbations with accept/reject
    based on loss improvement.
    
    Parameters
    ----------
    model : nn.Module
        Target model
    epsilon : float
        Linf perturbation budget
    max_queries : int
        Maximum number of queries
    p_init : float
        Initial fraction of image to perturb
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        max_queries: int = 5000,
        p_init: float = 0.8,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.max_queries = max_queries
        self.p_init = p_init
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def _get_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute margin loss (negative of correct class logit margin)."""
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            # Margin loss: max other logit - true logit
            true_logit = logits.gather(1, y.view(-1, 1)).squeeze()
            mask = torch.ones_like(logits).scatter_(1, y.view(-1, 1), 0)
            max_other = (logits * mask - (1 - mask) * 1e9).max(dim=1)[0]
            return max_other - true_logit
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Generate adversarial examples via random square search."""
        x = x.to(self.device)
        y = y.to(self.device)
        N, C, H, W = x.shape
        
        # Initialize with random Linf perturbation
        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        best_loss = self._get_loss(x_adv, y)
        queries = torch.ones(N, device=self.device)
        
        for i in range(self.max_queries):
            # Adaptive square size: decreases over iterations
            p = self.p_init * (1 - i / self.max_queries)
            s = max(int(round(p * min(H, W))), 1)
            
            # Random square location
            r = torch.randint(0, H - s + 1, (N,))
            c_pos = torch.randint(0, W - s + 1, (N,))
            
            # Propose update: random values in square region
            x_new = x_adv.clone()
            for b in range(N):
                # Random perturbation for the square
                patch = torch.empty(C, s, s, device=self.device).uniform_(
                    -self.epsilon, self.epsilon
                )
                x_new[b, :, r[b]:r[b]+s, c_pos[b]:c_pos[b]+s] = \
                    x[b, :, r[b]:r[b]+s, c_pos[b]:c_pos[b]+s] + patch
            
            x_new = torch.clamp(x_new, 0, 1)
            # Ensure Linf constraint
            delta = torch.clamp(x_new - x, -self.epsilon, self.epsilon)
            x_new = torch.clamp(x + delta, 0, 1)
            
            # Accept if loss improves
            new_loss = self._get_loss(x_new, y)
            improved = new_loss > best_loss
            x_adv[improved] = x_new[improved]
            best_loss[improved] = new_loss[improved]
            queries += 1
        
        return x_adv.detach()
```

## Comparison of Score-Based Methods

| Method | Queries/Example | Requires Probabilities | Key Advantage |
|--------|----------------|----------------------|---------------|
| NES | 5,000-20,000 | Yes | Full gradient estimate |
| Bandits | 2,000-10,000 | Yes | Prior-guided efficiency |
| Square Attack | 1,000-5,000 | Yes (or logits) | No gradient estimation |
| SimBA | 1,000-10,000 | Yes | Simple random search |

## Summary

Score-based attacks leverage model output probabilities to construct adversarial examples without gradient access. Square Attack represents the state-of-the-art in query efficiency, while NES and bandit methods provide more principled gradient estimation. For financial applications where query budgets are limited by API rate limits, Square Attack's efficiency makes it the preferred choice.

## References

1. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.
2. Ilyas, A., Engstrom, L., & Madry, A. (2019). "Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors." ICLR.
3. Guo, C., et al. (2019). "Simple Black-Box Adversarial Attacks." ICML.
