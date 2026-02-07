# Carlini-Wagner (C&W) Attack

## Introduction

The **Carlini-Wagner (C&W) attack** is an optimization-based adversarial attack that reformulates adversarial example generation as an unconstrained optimization problem. Introduced by Carlini and Wagner (2017), it is significantly stronger than gradient-based methods and often finds smaller perturbations that still cause misclassification.

## Mathematical Foundation

### Problem Reformulation

Unlike FGSM/PGD which directly constrain perturbation magnitude, C&W formulates the attack as:

$$
\min_{\boldsymbol{\delta}} \|\boldsymbol{\delta}\|_p + c \cdot f(\mathbf{x} + \boldsymbol{\delta})
$$

where:
- $\|\boldsymbol{\delta}\|_p$ is the perturbation norm (typically $p=2$)
- $c > 0$ is a trade-off constant
- $f(\cdot)$ is an objective function encouraging misclassification

### Logit-Based Objective

C&W uses a carefully designed objective based on logits $Z(\mathbf{x})$ (pre-softmax outputs):

**Untargeted attack:**
$$
f(\mathbf{x}') = \max\left(\max_{i \neq y} Z(\mathbf{x}')_i - Z(\mathbf{x}')_y, -\kappa\right)
$$

**Targeted attack (target class $t$):**
$$
f(\mathbf{x}') = \max\left(Z(\mathbf{x}')_y - Z(\mathbf{x}')_t, -\kappa\right)
$$

where $\kappa \geq 0$ is a **confidence parameter**:
- $\kappa = 0$: Just misclassify
- $\kappa > 0$: Misclassify with confidence margin

### Change of Variables

To handle box constraints $\mathbf{x}' \in [0, 1]^d$, C&W uses:

$$
\mathbf{x}' = \frac{1}{2}(\tanh(\mathbf{w}) + 1)
$$

where $\mathbf{w}$ is the unconstrained optimization variable. This ensures:
- $\tanh(\mathbf{w}) \in (-1, 1)$
- $\mathbf{x}' \in (0, 1)$ automatically

The perturbation becomes:
$$
\boldsymbol{\delta} = \frac{1}{2}(\tanh(\mathbf{w}) + 1) - \mathbf{x}
$$

### Final Optimization

Optimize over $\mathbf{w}$:
$$
\min_{\mathbf{w}} \left\|\frac{1}{2}(\tanh(\mathbf{w}) + 1) - \mathbf{x}\right\|_2^2 + c \cdot f\left(\frac{1}{2}(\tanh(\mathbf{w}) + 1)\right)
$$

### Binary Search for $c$

The optimal $c$ is unknown a priori. C&W uses binary search:

1. Initialize: $c_{\text{low}} = 0$, $c_{\text{high}} = 10^{10}$
2. For each step:
   - Set $c = (c_{\text{low}} + c_{\text{high}}) / 2$
   - Run optimization
   - If attack succeeds: $c_{\text{high}} = c$ (try smaller)
   - If attack fails: $c_{\text{low}} = c$ (try larger)
3. Return smallest successful $c$

## Algorithm

**Algorithm: C&W L2 Attack**

**Input:** Clean example $\mathbf{x}$, label $y$, model $f_\theta$

**Output:** Adversarial example $\mathbf{x}_{\text{adv}}$

1. Initialize $\mathbf{w}$ such that $\frac{1}{2}(\tanh(\mathbf{w}) + 1) = \mathbf{x}$
2. Binary search on $c$:
   - For each $c$:
     - Run Adam optimizer on $\mathbf{w}$ for $T$ iterations
     - Track best adversarial example (smallest $\|\boldsymbol{\delta}\|_2$ that succeeds)
3. Return best adversarial example

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict

class CarliniWagnerL2:
    """
    Carlini-Wagner L2 Attack.
    
    Optimization-based attack that minimizes L2 perturbation
    while ensuring misclassification.
    
    Parameters
    ----------
    model : nn.Module
        Neural network to attack
    c : float
        Initial trade-off constant
    kappa : float
        Confidence parameter (0 = just misclassify)
    learning_rate : float
        Learning rate for Adam optimizer
    max_iter : int
        Maximum optimization iterations
    binary_search_steps : int
        Number of binary search steps for c
    """
    
    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        kappa: float = 0.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        binary_search_steps: int = 9,
        initial_const: float = 1e-3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
    
    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [0,1] to tanh space."""
        # x = 0.5 * (tanh(w) + 1) => w = arctanh(2x - 1)
        x_scaled = x * 2 - 1  # Scale to [-1, 1]
        x_scaled = torch.clamp(x_scaled, -0.999999, 0.999999)  # Avoid infinity
        return torch.atanh(x_scaled)
    
    def _from_tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        """Convert from tanh space to [0,1]."""
        return 0.5 * (torch.tanh(w) + 1)
    
    def _f_objective(
        self,
        x_adv: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
        target_labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the C&W objective function f(x').
        
        Returns value that is:
        - Positive when attack hasn't succeeded
        - Negative (or zero) when attack has succeeded
        """
        logits = self.model(x_adv)
        
        # Get logit of true class
        true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
        
        if targeted:
            # Targeted: want target logit > true logit
            target_logit = logits.gather(1, target_labels.view(-1, 1)).squeeze(1)
            f = torch.clamp(true_logit - target_logit, min=-self.kappa)
        else:
            # Untargeted: want max other logit > true logit
            # Create mask for true class
            mask = torch.ones_like(logits).scatter_(1, y.view(-1, 1), 0)
            other_logits = logits * mask - (1 - mask) * 1e9
            max_other_logit = other_logits.max(dim=1)[0]
            f = torch.clamp(true_logit - max_other_logit, min=-self.kappa)
        
        return f
    
    def _optimize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: float,
        targeted: bool,
        target_labels: Optional[torch.Tensor]
    ) -> tuple:
        """Run optimization for a fixed c value."""
        batch_size = x.shape[0]
        
        # Initialize w in tanh space
        w = self._to_tanh_space(x).clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([w], lr=self.learning_rate)
        
        best_adv = x.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            
            # Convert to image space
            x_adv = self._from_tanh_space(w)
            
            # Compute perturbation L2 norm
            delta = x_adv - x
            l2_dist = delta.view(batch_size, -1).norm(p=2, dim=1)
            
            # Compute C&W objective
            f = self._f_objective(x_adv, y, targeted, target_labels)
            
            # Total loss: L2 + c * f
            loss = l2_dist.sum() + c * f.sum()
            
            loss.backward()
            optimizer.step()
            
            # Track best adversarial examples
            with torch.no_grad():
                # Check which examples are successful
                logits = self.model(x_adv)
                pred = logits.argmax(dim=1)
                
                if targeted:
                    success = (pred == target_labels)
                else:
                    success = (pred != y)
                
                # Update best if successful and smaller L2
                improved = success & (l2_dist < best_l2)
                best_adv[improved] = x_adv[improved]
                best_l2[improved] = l2_dist[improved]
        
        return best_adv, best_l2
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples using C&W attack.
        
        Parameters
        ----------
        x : torch.Tensor
            Clean images
        y : torch.Tensor
            True labels
        targeted : bool
            Perform targeted attack
        target_labels : torch.Tensor, optional
            Target labels for targeted attack
            
        Returns
        -------
        x_adv : torch.Tensor
            Adversarial examples
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if targeted and target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        batch_size = x.shape[0]
        
        # Initialize binary search bounds
        c_low = torch.zeros(batch_size, device=self.device)
        c_high = torch.full((batch_size,), 1e10, device=self.device)
        
        # Track overall best
        best_adv = x.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Binary search on c
        for step in range(self.binary_search_steps):
            if step == 0:
                c = torch.full((batch_size,), self.initial_const, device=self.device)
            else:
                c = (c_low + c_high) / 2
            
            c_mean = c.mean().item()
            
            # Run optimization
            adv, l2 = self._optimize(x, y, c_mean, targeted, target_labels)
            
            # Update best
            improved = l2 < best_l2
            best_adv[improved] = adv[improved]
            best_l2[improved] = l2[improved]
            
            # Check success and update bounds
            with torch.no_grad():
                pred = self.model(adv).argmax(dim=1)
                if targeted:
                    success = (pred == target_labels)
                else:
                    success = (pred != y)
            
            # Update binary search bounds
            c_high[success] = c[success]
            c_low[~success] = c[~success]
        
        return best_adv
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate C&W attack effectiveness."""
        with torch.no_grad():
            x, y, x_adv = x.to(self.device), y.to(self.device), x_adv.to(self.device)
            
            clean_pred = self.model(x).argmax(dim=1)
            adv_pred = self.model(x_adv).argmax(dim=1)
            
            clean_acc = (clean_pred == y).float().mean().item()
            robust_acc = (adv_pred == y).float().mean().item()
            
            delta = (x_adv - x).view(len(x), -1)
            l2_norms = delta.norm(p=2, dim=1)
        
        metrics = {
            'clean_accuracy': clean_acc,
            'robust_accuracy': robust_acc,
            'attack_success_rate': 1 - robust_acc,
            'avg_l2': l2_norms.mean().item(),
            'median_l2': l2_norms.median().item(),
            'max_linf': delta.abs().max().item()
        }
        
        if verbose:
            print("=" * 50)
            print("C&W L2 Attack Results")
            print("=" * 50)
            print(f"Clean Accuracy:      {metrics['clean_accuracy']:.2%}")
            print(f"Robust Accuracy:     {metrics['robust_accuracy']:.2%}")
            print(f"Attack Success Rate: {metrics['attack_success_rate']:.2%}")
            print(f"Avg L2 Perturbation: {metrics['avg_l2']:.4f}")
            print(f"Median L2:           {metrics['median_l2']:.4f}")
            print("=" * 50)
        
        return metrics
```

## Comparison: PGD vs C&W

| Aspect | PGD | C&W |
|--------|-----|-----|
| **Constraint** | Hard ($\|\boldsymbol{\delta}\| \leq \varepsilon$) | Soft (regularization) |
| **Optimization** | Gradient ascent | Adam on unconstrained |
| **Objective** | Maximize loss | Minimize perturbation + misclassify |
| **Finds minimal perturbation** | No | Yes |
| **Speed** | Fast | Slow |
| **Strength** | Strong | Very strong |

## Key Properties

### Strengths

1. **Finds minimal perturbations**: Unlike fixed-$\varepsilon$ attacks
2. **Bypasses gradient masking**: More robust optimization
3. **Highly effective**: Defeats many defenses
4. **Flexible**: Different norms ($\ell_0$, $\ell_2$, $\ell_\infty$)

### Limitations

1. **Computationally expensive**: Binary search × many iterations
2. **Hyperparameter sensitive**: $c$, learning rate, iterations
3. **Overkill for evaluation**: PGD often sufficient

## When to Use C&W

- **Defense evaluation**: When PGD might be fooled by gradient masking
- **Minimal perturbation**: Finding imperceptibility bounds
- **Targeted attacks**: Where precise control is needed
- **Research**: Understanding decision boundary geometry

## References

1. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P.
2. Chen, P. Y., et al. (2018). "EAD: Elastic-Net Attacks." AAAI.
