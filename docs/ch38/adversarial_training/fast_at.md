# Fast Adversarial Training

## Introduction

**Fast Adversarial Training** (Wong et al., 2020) achieves robust training using **single-step FGSM** with random initialization, approaching PGD-AT robustness at a fraction of the computational cost. The key discovery is that FGSM-based training—previously thought insufficient—works well when combined with proper random initialization and careful hyperparameter selection.

## Background

Early work by Goodfellow et al. (2015) proposed training on FGSM adversarial examples, but this approach was found to produce models that were robust to FGSM specifically but vulnerable to stronger multi-step attacks. Madry et al. (2018) showed that PGD-based training was necessary for genuine robustness but at 10× computational cost.

Wong et al. (2020) revisited single-step training and found that the failures were due to **catastrophic overfitting**, not an inherent limitation of single-step attacks.

## Catastrophic Overfitting

### The Problem

During FGSM adversarial training, robust accuracy can suddenly collapse from ~45% to ~0% within a single epoch. The model becomes a **FGSM specialist**: robust against FGSM but completely vulnerable to PGD.

### The Cause

The model learns to mask its gradients specifically against FGSM, creating a false sense of robustness. The FGSM attack, being a single gradient step, is easily fooled by gradient masking.

### The Solution

**Random initialization** before the FGSM step prevents catastrophic overfitting:

$$
\begin{aligned}
\boldsymbol{\delta}_0 &\sim \text{Uniform}[-\varepsilon, \varepsilon]^d \\
\boldsymbol{\delta}_{\text{FGSM}} &= \Pi_\varepsilon\left(\boldsymbol{\delta}_0 + \alpha \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}_0), y))\right)
\end{aligned}
$$

The random start ensures diverse adversarial examples during training, preventing the model from specializing against a deterministic attack direction.

## Algorithm

**Algorithm: Fast Adversarial Training**

```
For each epoch:
    For each mini-batch (x, y):
        1. Random init: δ ~ Uniform[-ε, ε]
        2. Compute gradient: g = ∇_x L(f_θ(x + δ), y)
        3. FGSM step: δ ← Π_ε(δ + α · sign(g))
        4. Compute loss: L = CE(f_θ(clip(x + δ, 0, 1)), y)
        5. Update: θ ← θ - η · ∇_θ L
```

Total cost: **2 forward-backward passes per batch** (1 for gradient, 1 for parameter update).

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm

class FastAdversarialTrainer:
    """
    Fast Adversarial Training with FGSM + random init.
    
    Achieves near-PGD robustness at ~2× standard training cost.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    epsilon : float
        Perturbation budget
    alpha : float
        FGSM step size (typically 1.25 × epsilon)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 10/255,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha  # Slightly larger than epsilon
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def _fgsm_with_random_start(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """FGSM attack with uniform random initialization."""
        # Random initialization within epsilon-ball
        delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)
        
        # Forward pass with perturbed input
        logits = self.model(torch.clamp(x + delta, 0, 1))
        loss = F.cross_entropy(logits, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # FGSM step
        with torch.no_grad():
            delta = delta + self.alpha * delta.grad.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        
        return torch.clamp(x + delta, 0, 1).detach()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Fast AT')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Generate adversarial examples with FGSM + random init
            x_adv = self._fgsm_with_random_start(x, y)
            
            # Update model on adversarial examples
            optimizer.zero_grad()
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{correct/total:.2%}'
            })
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
```

## Preventing Catastrophic Overfitting

### Early Stopping on Robust Accuracy

Monitor PGD robust accuracy during training and stop if it drops suddenly:

```python
def check_catastrophic_overfitting(
    model, test_loader, epsilon, device, prev_robust_acc
):
    """Detect if catastrophic overfitting has occurred."""
    model.eval()
    # Quick PGD-10 evaluation on subset
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        if total > 1000:
            break
        # PGD-10 attack
        x_adv = pgd_attack(model, x, y, epsilon, steps=10)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    robust_acc = correct / total
    
    # Flag if robust accuracy drops by >10%
    if prev_robust_acc - robust_acc > 0.1:
        print(f"WARNING: Catastrophic overfitting detected! "
              f"{prev_robust_acc:.2%} -> {robust_acc:.2%}")
        return True, robust_acc
    
    return False, robust_acc
```

## Comparison of Efficient AT Methods

| Method | Passes/Batch | Relative Cost | Clean Acc | Robust Acc |
|--------|-------------|---------------|-----------|------------|
| PGD-AT (10 steps) | 11 | 10× | 85% | 48% |
| Free AT ($m=8$) | 8 | ~1.2× | 83% | 43% |
| **Fast AT** | **2** | **~2×** | **83%** | **43%** |
| FGSM-AT (no random init) | 2 | 2× | 90% | 0%* |

*Catastrophic overfitting occurs without random initialization.

## Summary

Fast AT demonstrates that single-step adversarial training is viable when combined with random initialization. At only 2× the cost of standard training, it provides a practical option for achieving robustness in computationally constrained settings, including large-scale financial model training.

## References

1. Wong, E., Rice, L., & Kolter, J. Z. (2020). "Fast is Better than Free: Revisiting Adversarial Training." ICLR.
2. Andriushchenko, M., & Flammarion, N. (2020). "Understanding and Improving Fast Adversarial Training." NeurIPS.
