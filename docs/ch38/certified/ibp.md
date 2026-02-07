# Interval Bound Propagation (IBP)

## Introduction

**Interval Bound Propagation (IBP)** (Gowal et al., 2019) provides certified $\ell_\infty$ robustness by propagating interval bounds through the network. Unlike randomized smoothing (which certifies $\ell_2$ robustness), IBP directly certifies that no $\ell_\infty$ perturbation within the budget can change the prediction.

## Mathematical Foundation

### Core Idea

Given an input $\mathbf{x}$ and perturbation budget $\varepsilon$, the input lies in the interval:

$$
\mathbf{x} \in [\mathbf{x} - \varepsilon, \mathbf{x} + \varepsilon] = [\underline{\mathbf{x}}_0, \overline{\mathbf{x}}_0]
$$

IBP propagates these bounds through each layer of the network to obtain bounds on the final logits:

$$
[\underline{\mathbf{z}}_L, \overline{\mathbf{z}}_L] = \text{IBP}(f_\theta, [\underline{\mathbf{x}}_0, \overline{\mathbf{x}}_0])
$$

### Propagation Through Linear Layers

For a linear layer $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ with input bounds $[\underline{\mathbf{x}}, \overline{\mathbf{x}}]$:

$$
\begin{aligned}
\underline{z}_j &= \sum_i \min(W_{ji} \underline{x}_i, W_{ji} \overline{x}_i) + b_j \\
&= \sum_i [W_{ji}]_+ \underline{x}_i + [W_{ji}]_- \overline{x}_i + b_j
\end{aligned}
$$

$$
\overline{z}_j = \sum_i [W_{ji}]_+ \overline{x}_i + [W_{ji}]_- \underline{x}_i + b_j
$$

where $[a]_+ = \max(a, 0)$ and $[a]_- = \min(a, 0)$.

In matrix form:

$$
\begin{aligned}
\underline{\mathbf{z}} &= \mathbf{W}^+ \underline{\mathbf{x}} + \mathbf{W}^- \overline{\mathbf{x}} + \mathbf{b} \\
\overline{\mathbf{z}} &= \mathbf{W}^+ \overline{\mathbf{x}} + \mathbf{W}^- \underline{\mathbf{x}} + \mathbf{b}
\end{aligned}
$$

### Propagation Through ReLU

For ReLU activation $z = \max(0, x)$ with bounds $[\underline{x}, \overline{x}]$:

$$
\underline{z} = \max(0, \underline{x}), \quad \overline{z} = \max(0, \overline{x})
$$

### Certification Condition

A prediction is certified robust if the lower bound of the true class logit exceeds the upper bound of all other classes:

$$
\underline{z}_y > \max_{k \neq y} \overline{z}_k \implies \text{certified robust}
$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class IBPBounds:
    """
    Interval Bound Propagation for certified Linf robustness.
    
    Propagates interval bounds through a feedforward network
    to certify predictions.
    """
    
    @staticmethod
    def propagate_linear(
        layer: nn.Linear,
        lb: torch.Tensor,
        ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through a linear layer."""
        W = layer.weight
        b = layer.bias if layer.bias is not None else 0
        
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        
        new_lb = lb @ W_pos.t() + ub @ W_neg.t() + b
        new_ub = ub @ W_pos.t() + lb @ W_neg.t() + b
        
        return new_lb, new_ub
    
    @staticmethod
    def propagate_relu(
        lb: torch.Tensor, ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through ReLU."""
        return torch.clamp(lb, min=0), torch.clamp(ub, min=0)
    
    @staticmethod
    def propagate_conv2d(
        layer: nn.Conv2d,
        lb: torch.Tensor,
        ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through Conv2d."""
        W = layer.weight
        b = layer.bias
        
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        
        new_lb = (nn.functional.conv2d(lb, W_pos, bias=b,
                    stride=layer.stride, padding=layer.padding) +
                  nn.functional.conv2d(ub, W_neg, bias=None,
                    stride=layer.stride, padding=layer.padding))
        new_ub = (nn.functional.conv2d(ub, W_pos, bias=b,
                    stride=layer.stride, padding=layer.padding) +
                  nn.functional.conv2d(lb, W_neg, bias=None,
                    stride=layer.stride, padding=layer.padding))
        
        return new_lb, new_ub


def certify_ibp(
    model: nn.Sequential,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float
) -> Dict[str, float]:
    """
    Certify predictions using IBP.
    
    Parameters
    ----------
    model : nn.Sequential
        Network with alternating Linear/Conv2d and ReLU layers
    x : torch.Tensor
        Input batch
    y : torch.Tensor
        True labels
    epsilon : float
        Linf perturbation budget
    
    Returns
    -------
    results : dict with certified_accuracy and clean_accuracy
    """
    ibp = IBPBounds()
    
    # Initial bounds
    lb = torch.clamp(x - epsilon, 0, 1)
    ub = torch.clamp(x + epsilon, 0, 1)
    
    # Propagate through layers
    for layer in model:
        if isinstance(layer, nn.Linear):
            lb, ub = ibp.propagate_linear(layer, lb, ub)
        elif isinstance(layer, nn.Conv2d):
            lb, ub = ibp.propagate_conv2d(layer, lb, ub)
        elif isinstance(layer, nn.ReLU):
            lb, ub = ibp.propagate_relu(lb, ub)
        elif isinstance(layer, nn.Flatten):
            lb = lb.flatten(1)
            ub = ub.flatten(1)
    
    # Check certification
    # Certified if: lower_bound[y] > upper_bound[k] for all k ≠ y
    true_lb = lb.gather(1, y.view(-1, 1))  # Lower bound of true class
    
    # Set true class upper bound to -inf for comparison
    ub_others = ub.clone()
    ub_others.scatter_(1, y.view(-1, 1), float('-inf'))
    max_other_ub = ub_others.max(dim=1)[0]
    
    certified = (true_lb.squeeze() > max_other_ub)
    
    # Clean accuracy
    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
        clean_correct = (clean_pred == y)
    
    return {
        'clean_accuracy': clean_correct.float().mean().item(),
        'certified_accuracy': (certified & clean_correct).float().mean().item(),
        'certification_rate': certified.float().mean().item()
    }
```

## IBP Training

To improve certified accuracy, models can be trained with IBP-based loss:

$$
\mathcal{L}_{\text{IBP}} = \text{CE}(\underline{\mathbf{z}}_y - \max_{k \neq y} \overline{\mathbf{z}}_k, \mathbf{1})
$$

This encourages the network to maintain separation between the true class lower bound and other class upper bounds.

### Warm-Up Schedule

IBP training requires careful scheduling: start with standard training and gradually increase the weight of the IBP loss (and $\varepsilon$) to avoid training instability.

## Limitations

- **Loose bounds**: IBP bounds become increasingly loose with network depth, limiting certification to small networks
- **Small $\varepsilon$**: Practical for small perturbation budgets; bounds explode for large $\varepsilon$
- **Architecture constraints**: Works best with simple feedforward networks; harder for complex architectures

## Comparison with Randomized Smoothing

| Aspect | IBP | Randomized Smoothing |
|--------|-----|---------------------|
| Certified norm | $\ell_\infty$ | $\ell_2$ |
| Guarantee type | Deterministic | Probabilistic |
| Bound tightness | Loose for deep nets | Independent of depth |
| Scalability | Small networks | Any architecture |
| Training cost | Moderate | Low (noise augmentation) |

## References

1. Gowal, S., et al. (2019). "Scalable Verified Training for Provably Robust Image Classification." ICCV.
2. Mirman, M., Gehr, T., & Vechev, M. (2018). "Differentiable Abstract Interpretation for Provably Robust Neural Networks." ICML.
3. Wong, E., & Kolter, J. Z. (2018). "Provable Defenses Against Adversarial Examples via the Convex Outer Adversarial Polytope." ICML.
