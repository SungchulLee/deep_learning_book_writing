# Projected Gradient Descent (PGD) Attack

## Introduction

**Projected Gradient Descent (PGD)** extends FGSM to a multi-step iterative attack, making it significantly more powerful. Introduced by Madry et al. (2018), PGD is considered the de facto standard for evaluating adversarial robustness and is the foundation of robust adversarial training.

## Mathematical Foundation

### From FGSM to Iterative Attacks

FGSM takes a single large step in the gradient direction. This is suboptimal because:

1. The linear approximation degrades far from $\mathbf{x}$
2. A single step may overshoot or undershoot the optimum
3. The loss landscape is non-convex

PGD addresses these issues by taking **multiple small steps**, re-computing gradients at each iteration.

### PGD Formulation

PGD solves the constrained optimization problem:

$$
\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

using projected gradient ascent. The iterative update is:

$$
\mathbf{x}^{(t+1)} = \Pi_{\mathbf{x} + \mathcal{S}}\left( \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)) \right)
$$

where:
- $\Pi_{\mathbf{x} + \mathcal{S}}$ projects back onto the $\varepsilon$-ball around $\mathbf{x}$
- $\alpha$ is the step size
- $\mathcal{S} = \{\boldsymbol{\delta} : \|\boldsymbol{\delta}\|_p \leq \varepsilon\}$ is the allowed perturbation set

### Algorithm

**Algorithm: PGD Attack**

**Input:** Clean example $\mathbf{x}$, label $y$, model $f_\theta$, epsilon $\varepsilon$, step size $\alpha$, iterations $T$

**Output:** Adversarial example $\mathbf{x}_{\text{adv}}$

1. **Initialize:** $\mathbf{x}^{(0)} = \mathbf{x} + \mathbf{u}$ where $\mathbf{u} \sim \text{Uniform}[-\varepsilon, \varepsilon]^d$
2. **For** $t = 0, 1, \ldots, T-1$:
   - Compute gradient: $\mathbf{g} = \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)$
   - Update: $\tilde{\mathbf{x}}^{(t+1)} = \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\mathbf{g})$
   - Project: $\mathbf{x}^{(t+1)} = \Pi_{\varepsilon}(\tilde{\mathbf{x}}^{(t+1)}, \mathbf{x})$
   - Clip to valid range: $\mathbf{x}^{(t+1)} = \text{clip}(\mathbf{x}^{(t+1)}, 0, 1)$
3. **Return:** $\mathbf{x}_{\text{adv}} = \mathbf{x}^{(T)}$

### Projection Operators

**$\ell_\infty$ Projection:**

$$
\Pi_\varepsilon^{\infty}(\tilde{\mathbf{x}}, \mathbf{x})_i = \text{clip}(\tilde{x}_i, x_i - \varepsilon, x_i + \varepsilon)
$$

**$\ell_2$ Projection:**

$$
\Pi_\varepsilon^{2}(\tilde{\mathbf{x}}, \mathbf{x}) = 
\begin{cases}
\tilde{\mathbf{x}} & \text{if } \|\tilde{\mathbf{x}} - \mathbf{x}\|_2 \leq \varepsilon \\
\mathbf{x} + \varepsilon \cdot \frac{\tilde{\mathbf{x}} - \mathbf{x}}{\|\tilde{\mathbf{x}} - \mathbf{x}\|_2} & \text{otherwise}
\end{cases}
$$

### Step Size Selection

| Strategy | Formula | Notes |
|----------|---------|-------|
| **Linear** | $\alpha = \varepsilon / T$ | Conservative |
| **Scaled linear** | $\alpha = 2\varepsilon / T$ | Standard (Madry et al.) |
| **Aggressive** | $\alpha = 2.5\varepsilon / T$ | Faster convergence |

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict

class PGD:
    """
    Projected Gradient Descent (PGD) Attack.
    
    Parameters
    ----------
    model : nn.Module
        Neural network to attack
    epsilon : float
        Maximum perturbation magnitude
    alpha : float
        Step size per iteration
    num_iter : int
        Number of PGD iterations
    norm : str
        Norm constraint ('linf' or 'l2')
    random_init : bool
        Whether to use random initialization
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: Optional[float] = None,
        num_iter: int = 10,
        norm: Literal['linf', 'l2'] = 'linf',
        random_init: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha else 2 * epsilon / num_iter
        self.num_iter = num_iter
        self.norm = norm
        self.random_init = random_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.model.eval()
        self.model.to(self.device)
    
    def _project(self, x_adv: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project onto epsilon-ball."""
        delta = x_adv - x
        
        if self.norm == 'linf':
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        elif self.norm == 'l2':
            delta_flat = delta.view(delta.shape[0], -1)
            norm = delta_flat.norm(p=2, dim=1, keepdim=True)
            factor = torch.clamp(norm / self.epsilon, min=1.0)
            delta = (delta_flat / factor).view(delta.shape)
        
        return torch.clamp(x + delta, self.clip_min, self.clip_max)
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate adversarial examples using PGD."""
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Initialize
        if self.random_init:
            if self.norm == 'linf':
                delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            else:
                delta = torch.randn_like(x)
                delta = delta / delta.view(len(x), -1).norm(p=2, dim=1, keepdim=True).view(-1,1,1,1)
                delta = delta * self.epsilon * torch.rand(len(x), 1, 1, 1, device=self.device)
            x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        else:
            x_adv = x.clone()
        
        # PGD iterations
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            
            if targeted:
                loss = -self.loss_fn(logits, target_labels.to(self.device))
            else:
                loss = self.loss_fn(logits, y)
            
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad.data
            
            # Gradient step
            if self.norm == 'linf':
                x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            else:
                grad_norm = grad.view(len(x), -1).norm(p=2, dim=1, keepdim=True).view(-1,1,1,1)
                x_adv = x_adv.detach() + self.alpha * grad / (grad_norm + 1e-8)
            
            x_adv = self._project(x_adv, x)
        
        return x_adv.detach()
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, x_adv: torch.Tensor) -> Dict[str, float]:
        """Evaluate attack effectiveness."""
        with torch.no_grad():
            clean_acc = (self.model(x.to(self.device)).argmax(1) == y.to(self.device)).float().mean()
            robust_acc = (self.model(x_adv.to(self.device)).argmax(1) == y.to(self.device)).float().mean()
            delta = (x_adv - x).view(len(x), -1)
            
        return {
            'clean_accuracy': clean_acc.item(),
            'robust_accuracy': robust_acc.item(),
            'attack_success_rate': 1 - robust_acc.item(),
            'avg_linf': delta.abs().max(dim=1)[0].mean().item(),
            'avg_l2': delta.norm(p=2, dim=1).mean().item()
        }
```

## Comparison: FGSM vs PGD

| Aspect | FGSM | PGD |
|--------|------|-----|
| Steps | 1 | $T$ (10-100) |
| Initialization | None | Random |
| Strength | Weak | Strong |
| Speed | Fast | $T \times$ slower |

**Typical Results (CIFAR-10, ε=8/255):**

| Method | Success Rate |
|--------|--------------|
| FGSM | ~65% |
| PGD-10 | ~85% |
| PGD-40 | ~92% |

## Variants

### Momentum Iterative FGSM (MI-FGSM)

$$
\mathbf{g}^{(t)} = \mu \cdot \mathbf{g}^{(t-1)} + \frac{\nabla_\mathbf{x} \mathcal{L}}{\|\nabla_\mathbf{x} \mathcal{L}\|_1}
$$

Improves transferability by accumulating gradient momentum.

### Auto-PGD (APGD)

Adapts step size based on loss improvement—key component of AutoAttack.

## Connection to Adversarial Training

PGD is the standard attack for robust training:

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y)}\left[\max_{\|\boldsymbol{\delta}\|_\infty \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)\right]
$$

The inner max is approximated by PGD.

## References

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Dong, Y., et al. (2018). "Boosting Adversarial Attacks with Momentum." CVPR.
