# Robustness Definitions

## Introduction

Rigorous definitions of robustness are essential for meaningful evaluation and comparison of defense methods. This section formalizes the key metrics, establishes the mathematical framework for measuring adversarial vulnerability, and provides a reusable PyTorch base class for implementing attacks.

## Core Definitions

### Pointwise Robustness

A classifier $f$ is **pointwise robust** at input $\mathbf{x}$ with label $y$ under $\ell_p$ norm with radius $\varepsilon$ if:

$$
\forall \boldsymbol{\delta} \in \mathbb{R}^d, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon \implies f(\mathbf{x} + \boldsymbol{\delta}) = y
$$

The classifier correctly classifies every point in the $\varepsilon$-ball around $\mathbf{x}$.

### Minimum Adversarial Perturbation

The **minimum adversarial perturbation** for input $\mathbf{x}$ is the smallest perturbation that causes misclassification:

$$
\varepsilon^*(\mathbf{x}) = \min \left\{ \|\boldsymbol{\delta}\|_p : f(\mathbf{x} + \boldsymbol{\delta}) \neq y \right\}
$$

This measures the "distance to the decision boundary" and is the most fundamental robustness metric for individual examples.

### Robust Risk

The **robust risk** (or adversarial risk) generalizes standard classification risk to the worst-case setting:

$$
R_{\text{rob}}(f, \varepsilon) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathbf{1}[f(\mathbf{x} + \boldsymbol{\delta}) \neq y]\right]
$$

The standard risk is the special case with $\varepsilon = 0$:

$$
R_{\text{std}}(f) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\mathbf{1}[f(\mathbf{x}) \neq y]]
$$

## Attack Success Metrics

### Attack Success Rate (ASR)

The fraction of examples where the attack causes misclassification:

$$
\text{ASR} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) \neq y_i]
$$

For targeted attacks toward class $y_{\text{target}}$:

$$
\text{ASR}_{\text{targeted}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) = y_{\text{target}}]
$$

### Robust Accuracy

Accuracy under the strongest possible adversarial attack within the perturbation budget:

$$
\text{Robust Acc}(\varepsilon) = \frac{1}{N} \sum_{i=1}^N \min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}) = y_i]
$$

**Relationship to ASR:** For untargeted attacks, $\text{ASR} = 1 - \text{Robust Acc}$.

In practice, exact robust accuracy is intractable (requires solving a hard inner maximization). We approximate it using strong attacks like PGD or AutoAttack.

### Perturbation Magnitude Metrics

**Average $\ell_p$ distance of successful attacks:**

$$
\bar{d}_p = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \|\boldsymbol{\delta}_i^*\|_p
$$

where $\mathcal{S} = \{i : f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) \neq y_i\}$ is the set of successful attacks.

**Minimum required perturbation per example:**

$$
\varepsilon^*_i = \min \{\varepsilon : \exists \boldsymbol{\delta}, \|\boldsymbol{\delta}\|_p \leq \varepsilon, f_\theta(\mathbf{x}_i + \boldsymbol{\delta}) \neq y_i\}
$$

### Certified Accuracy

For defenses with provable guarantees, **certified accuracy** at radius $r$ is:

$$
\text{Certified Acc}(r) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f(\mathbf{x}_i) = y_i \text{ and } R_i \geq r]
$$

where $R_i$ is the certified robustness radius at example $i$. This is a **lower bound** on true robust accuracy.

## PyTorch Implementation: Attack Base Class

```python
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Literal, Dict

class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks.
    
    All attacks share:
    - A target model
    - A perturbation budget (epsilon)
    - A norm constraint
    - Methods for generating and evaluating adversarial examples
    
    Parameters
    ----------
    model : nn.Module
        Neural network to attack
    epsilon : float
        Perturbation budget
    norm : str
        Norm constraint ('linf', 'l2', 'l1')
    clip_min : float
        Minimum valid input value
    clip_max : float
        Maximum valid input value
    device : torch.device, optional
        Computation device
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        norm: Literal['linf', 'l2', 'l1'] = 'linf',
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
    
    @abstractmethod
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples.
        
        Parameters
        ----------
        x : torch.Tensor
            Clean inputs, shape (N, C, H, W) or (N, D)
        y : torch.Tensor
            True labels, shape (N,)
        targeted : bool
            Whether to perform targeted attack
        target_labels : torch.Tensor, optional
            Target labels for targeted attack
            
        Returns
        -------
        x_adv : torch.Tensor
            Adversarial examples
        """
        pass
    
    def project(
        self,
        delta: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Project perturbation onto epsilon-ball and clip to valid range.
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
                delta_flat[i] = self._project_l1(delta_flat[i], self.epsilon)
            delta = delta_flat.view(delta.shape)
        
        # Clip to ensure x + delta is in valid range
        delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
        return delta
    
    @staticmethod
    def _project_l1(v: torch.Tensor, radius: float) -> torch.Tensor:
        """Project vector v onto L1 ball of given radius."""
        if torch.norm(v, p=1) <= radius:
            return v
        u = torch.abs(v)
        sorted_u, _ = torch.sort(u, descending=True)
        cumsum = torch.cumsum(sorted_u, dim=0)
        rho = torch.where(
            sorted_u > (cumsum - radius) / (torch.arange(len(u), device=v.device) + 1),
            torch.arange(len(u), device=v.device),
            torch.zeros_like(torch.arange(len(u), device=v.device))
        ).max()
        theta = (cumsum[rho] - radius) / (rho + 1)
        return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate attack effectiveness.
        
        Returns
        -------
        metrics : dict
            - clean_accuracy: Accuracy on clean examples
            - robust_accuracy: Accuracy on adversarial examples
            - attack_success_rate: Fraction of successful attacks
            - avg_perturbation: Average perturbation magnitude
        """
        with torch.no_grad():
            clean_pred = self.model(x.to(self.device)).argmax(dim=1)
            clean_correct = (clean_pred == y.to(self.device)).sum().item()
            
            adv_pred = self.model(x_adv.to(self.device)).argmax(dim=1)
            adv_correct = (adv_pred == y.to(self.device)).sum().item()
            
            delta = (x_adv - x).view(len(x), -1)
            
            if self.norm == 'linf':
                avg_pert = delta.abs().max(dim=1)[0].mean().item()
            elif self.norm == 'l2':
                avg_pert = torch.norm(delta, p=2, dim=1).mean().item()
            elif self.norm == 'l1':
                avg_pert = torch.norm(delta, p=1, dim=1).mean().item()
        
        n = len(y)
        return {
            'clean_accuracy': clean_correct / n,
            'robust_accuracy': adv_correct / n,
            'attack_success_rate': 1 - adv_correct / n,
            'avg_perturbation': avg_pert
        }
```

## Robustness Curves

A **robustness curve** plots robust accuracy as a function of perturbation budget $\varepsilon$:

$$
\varepsilon \mapsto \text{Robust Acc}(\varepsilon)
$$

This curve is always non-increasing (larger budgets give the attacker more power) and approaches zero as $\varepsilon \to \infty$. The area under the robustness curve provides a single scalar summary of robustness across all perturbation levels.

```python
def compute_robustness_curve(
    attack_class,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilons: list,
    **attack_kwargs
) -> dict:
    """
    Compute robustness curve: robust accuracy vs epsilon.
    
    Parameters
    ----------
    attack_class : type
        Attack class (e.g., PGD, FGSM)
    model : nn.Module
        Model to evaluate
    x, y : torch.Tensor
        Test inputs and labels
    epsilons : list[float]
        List of perturbation budgets to evaluate
    
    Returns
    -------
    results : dict
        Maps epsilon to robust accuracy
    """
    results = {}
    for eps in epsilons:
        attack = attack_class(model, epsilon=eps, **attack_kwargs)
        x_adv = attack.generate(x, y)
        metrics = attack.evaluate(x, y, x_adv)
        results[eps] = metrics['robust_accuracy']
    return results
```

## Summary

| Metric | Formula | Measures |
|--------|---------|----------|
| Attack Success Rate | $\frac{1}{N}\sum \mathbf{1}[f(\mathbf{x}_i + \boldsymbol{\delta}_i) \neq y_i]$ | Attack effectiveness |
| Robust Accuracy | $\frac{1}{N}\sum \min_\delta \mathbf{1}[f(\mathbf{x}_i + \boldsymbol{\delta}_i) = y_i]$ | Model resilience |
| Certified Accuracy | Robust acc with provable guarantee | Worst-case resilience |
| Min. Perturbation | $\min\{\varepsilon : \exists \boldsymbol{\delta}, f(\mathbf{x}+\boldsymbol{\delta}) \neq y\}$ | Decision boundary distance |

These definitions provide the foundation for all attack and defense evaluations in subsequent sections.

## References

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
