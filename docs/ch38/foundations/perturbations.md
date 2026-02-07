# Perturbation Types

## Introduction

The choice of perturbation constraint defines the "budget" an adversary operates within. Different norms capture different notions of imperceptibility and lead to different attack strategies. This section formalizes the standard perturbation types, their geometric properties, and their projection operators used in constrained optimization.

## $\ell_\infty$ Norm

$$
\|\boldsymbol{\delta}\|_\infty = \max_i |\delta_i| \leq \varepsilon
$$

**Interpretation:** Each coordinate changes by at most $\varepsilon$. This is the most commonly used constraint in adversarial robustness research.

**Properties:**

- Constrains the maximum per-feature change
- Natural for images: bounds the per-pixel perturbation
- The $\ell_\infty$ ball is a hypercube in $\mathbb{R}^d$
- Standard benchmarks: $\varepsilon = 8/255 \approx 0.031$ for CIFAR-10, $\varepsilon = 4/255$ for ImageNet

**Projection operator:**

$$
\Pi_\varepsilon^{\infty}(\boldsymbol{\delta})_i = \text{clip}(\delta_i, -\varepsilon, \varepsilon)
$$

This is computationally trivial—simply clamp each coordinate independently.

**Financial interpretation:** In feature-space attacks on financial models, $\ell_\infty$ bounds the maximum change in any single input feature (e.g., no single financial metric changes by more than $\varepsilon$).

## $\ell_2$ Norm

$$
\|\boldsymbol{\delta}\|_2 = \sqrt{\sum_i \delta_i^2} \leq \varepsilon
$$

**Interpretation:** The total Euclidean magnitude of the perturbation is bounded. Individual coordinates may change by more than $\varepsilon$, provided the overall vector length remains small.

**Properties:**

- Allows larger changes in fewer dimensions
- Natural for physical perturbations and continuous signals
- The $\ell_2$ ball is a hypersphere in $\mathbb{R}^d$
- Standard benchmarks: $\varepsilon = 0.5$ for CIFAR-10, $\varepsilon = 3.0$ for ImageNet

**Projection operator:**

$$
\Pi_\varepsilon^{2}(\boldsymbol{\delta}) = 
\begin{cases}
\boldsymbol{\delta} & \text{if } \|\boldsymbol{\delta}\|_2 \leq \varepsilon \\
\varepsilon \cdot \frac{\boldsymbol{\delta}}{\|\boldsymbol{\delta}\|_2} & \text{otherwise}
\end{cases}
$$

This normalizes the perturbation to lie on the surface of the $\varepsilon$-ball when it exceeds the budget.

## $\ell_1$ Norm

$$
\|\boldsymbol{\delta}\|_1 = \sum_i |\delta_i| \leq \varepsilon
$$

**Interpretation:** The total absolute change across all coordinates is bounded.

**Properties:**

- Encourages **sparse** perturbations (few coordinates modified significantly)
- The $\ell_1$ ball is a cross-polytope (diamond shape)
- Harder to optimize than $\ell_\infty$ or $\ell_2$
- Relates to $\ell_0$ through convex relaxation

**Projection operator (Duchi et al., 2008):**

The projection onto the $\ell_1$ ball requires a sorting-based algorithm:

```python
def project_l1(v: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Project vector v onto L1 ball of given radius.
    Uses the algorithm from Duchi et al. (2008).
    """
    if torch.norm(v, p=1) <= radius:
        return v
    
    # Sort absolute values in descending order
    u = torch.abs(v)
    sorted_u, _ = torch.sort(u, descending=True)
    
    # Find the threshold via cumulative sum
    cumsum = torch.cumsum(sorted_u, dim=0)
    indices = torch.arange(1, len(u) + 1, device=v.device, dtype=v.dtype)
    rho = torch.where(
        sorted_u > (cumsum - radius) / indices,
        indices,
        torch.zeros_like(indices)
    ).max().long()
    
    theta = (cumsum[rho - 1] - radius) / rho
    
    # Apply soft thresholding
    return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
```

## $\ell_0$ "Norm"

$$
\|\boldsymbol{\delta}\|_0 = |\{i : \delta_i \neq 0\}| \leq k
$$

**Interpretation:** At most $k$ features (pixels) may be modified, but those features can change by an arbitrary amount (usually still clipped to valid input range).

**Properties:**

- Extreme sparsity: only $k$ coordinates are perturbed
- Not a true norm (violates homogeneity)
- Combinatorially hard to optimize (NP-hard in general)
- Often relaxed to $\ell_1$ in practice
- Models pixel-level attacks (e.g., placing a few adversarial pixels)

## Geometric Comparison

The unit balls of different norms have distinct shapes in $\mathbb{R}^d$:

| Norm | Unit Ball Shape | Sparsity | Optimization | Vertices |
|------|-----------------|----------|--------------|----------|
| $\ell_\infty$ | Hypercube | None | Easy (element-wise) | $2^d$ |
| $\ell_2$ | Hypersphere | None | Easy (gradient) | Continuous |
| $\ell_1$ | Cross-polytope | Moderate | Moderate | $2d$ |
| $\ell_0$ | Discrete set | Extreme | NP-hard | Combinatorial |

### Norm Relationships

For any vector $\boldsymbol{\delta} \in \mathbb{R}^d$:

$$
\|\boldsymbol{\delta}\|_\infty \leq \|\boldsymbol{\delta}\|_2 \leq \|\boldsymbol{\delta}\|_1 \leq \sqrt{d} \|\boldsymbol{\delta}\|_2 \leq d \|\boldsymbol{\delta}\|_\infty
$$

These inequalities allow converting perturbation budgets between norms, though the conversions become loose in high dimensions.

## PyTorch Implementation: Unified Projection

```python
import torch
import torch.nn as nn
from typing import Literal

class PerturbationProjector:
    """
    Unified projection onto Lp epsilon-balls.
    
    Handles Linf, L2, and L1 projections with valid range clipping.
    
    Parameters
    ----------
    norm : str
        Norm type ('linf', 'l2', 'l1')
    epsilon : float
        Perturbation budget
    clip_min : float
        Minimum valid input value
    clip_max : float
        Maximum valid input value
    """
    
    def __init__(
        self,
        norm: Literal['linf', 'l2', 'l1'] = 'linf',
        epsilon: float = 8/255,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        self.norm = norm
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def project(
        self,
        delta: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Project perturbation onto epsilon-ball and clip to valid range.
        
        Parameters
        ----------
        delta : torch.Tensor
            Perturbation tensor, shape (N, ...)
        x : torch.Tensor
            Original input (for valid range clipping)
            
        Returns
        -------
        delta_proj : torch.Tensor
            Projected perturbation
        """
        if self.norm == 'linf':
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        
        elif self.norm == 'l2':
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            factor = torch.clamp(norms / self.epsilon, min=1.0)
            delta_flat = delta_flat / factor
            delta = delta_flat.view(delta.shape)
        
        elif self.norm == 'l1':
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            for i in range(batch_size):
                if torch.norm(delta_flat[i], p=1) > self.epsilon:
                    delta_flat[i] = self._project_l1_single(
                        delta_flat[i], self.epsilon
                    )
            delta = delta_flat.view(delta.shape)
        
        # Clip to ensure x + delta is in valid range
        delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
        return delta
    
    @staticmethod
    def _project_l1_single(v: torch.Tensor, radius: float) -> torch.Tensor:
        """Project a single vector onto L1 ball."""
        u = torch.abs(v)
        sorted_u, _ = torch.sort(u, descending=True)
        cumsum = torch.cumsum(sorted_u, dim=0)
        indices = torch.arange(1, len(u) + 1, device=v.device, dtype=v.dtype)
        rho = torch.where(
            sorted_u > (cumsum - radius) / indices,
            indices,
            torch.zeros_like(indices)
        ).max().long()
        theta = (cumsum[rho - 1] - radius) / rho
        return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
    
    def random_init(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Generate random initialization within the epsilon-ball.
        
        Parameters
        ----------
        shape : tuple
            Shape of the perturbation tensor
        device : torch.device
            Device for the tensor
            
        Returns
        -------
        delta : torch.Tensor
            Random perturbation within epsilon-ball
        """
        if self.norm == 'linf':
            return torch.empty(shape, device=device).uniform_(
                -self.epsilon, self.epsilon
            )
        elif self.norm == 'l2':
            delta = torch.randn(shape, device=device)
            delta_flat = delta.view(shape[0], -1)
            norms = delta_flat.norm(p=2, dim=1, keepdim=True)
            delta_flat = delta_flat / norms  # Unit sphere
            # Scale uniformly within ball
            r = torch.rand(shape[0], 1, device=device) ** (1.0 / delta_flat.shape[1])
            delta_flat = delta_flat * r * self.epsilon
            return delta_flat.view(shape)
        elif self.norm == 'l1':
            # Approximate: sample uniformly and project
            delta = torch.empty(shape, device=device).uniform_(-1, 1)
            delta_flat = delta.view(shape[0], -1)
            for i in range(shape[0]):
                delta_flat[i] = self._project_l1_single(
                    delta_flat[i], self.epsilon
                )
            return delta_flat.view(shape)
```

## Perturbation Budgets in Practice

### Standard Benchmarks

| Dataset | $\ell_\infty$ | $\ell_2$ | Justification |
|---------|---------------|----------|---------------|
| MNIST | $0.3$ | $2.0$ | Digit remains recognizable |
| CIFAR-10 | $8/255 \approx 0.031$ | $0.5$ | Changes imperceptible to humans |
| ImageNet | $4/255 \approx 0.016$ | $3.0$ | Higher resolution, tighter budget |

### Financial Applications

For tabular financial data, perturbation budgets must be domain-specific:

| Feature Type | Suggested $\ell_\infty$ Budget | Rationale |
|-------------|-------------------------------|-----------|
| Normalized prices | 0.01-0.05 | Small price manipulation |
| Log returns | 0.005-0.02 | Realistic market noise |
| Binary indicators | 0 (fixed) | Cannot perturb categorical |
| Continuous ratios | 0.01-0.1 | Feature-dependent |

## Summary

| Norm | Constraint | Best For | Projection Complexity |
|------|-----------|----------|----------------------|
| $\ell_\infty$ | Max per-coordinate | Image attacks, uniform changes | $O(d)$ |
| $\ell_2$ | Euclidean magnitude | Physical perturbations | $O(d)$ |
| $\ell_1$ | Total absolute change | Sparse perturbations | $O(d \log d)$ |
| $\ell_0$ | Number of changes | Pixel attacks | NP-hard |

The choice of norm should match the threat model's notion of imperceptibility. For images, $\ell_\infty$ and $\ell_2$ are standard. For financial features, domain-specific constraints often supersede any single norm.

## References

1. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
2. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P.
3. Duchi, J., et al. (2008). "Efficient Projections onto the L1-Ball for Learning in High Dimensions." ICML.
4. Croce, F., & Hein, M. (2021). "Mind the Box: L1-APGD for Sparse Adversarial Attacks on Image Classifiers." ICML.
