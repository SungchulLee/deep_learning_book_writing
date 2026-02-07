# MART: Misclassification-Aware Robust Training

## Introduction

**MART** (Misclassification-Aware adversarial Robust Training; Wang et al., 2020) improves adversarial training by assigning different weights to training examples based on how difficult they are to classify correctly. The key insight is that not all examples contribute equally to robustness: examples that are already difficult to classify naturally deserve more attention during robust training.

## Motivation

Standard adversarial training and TRADES treat all examples equally. However:

- **Easy examples** (high clean confidence $p_y(\mathbf{x}) \approx 1$): Already well-classified; the model has large margin and is less likely to be fooled
- **Hard examples** (low clean confidence $p_y(\mathbf{x}) \approx 0$): Near the decision boundary; adversarial perturbations easily cause misclassification

MART focuses defense effort on hard examples where it is most needed.

## Mathematical Formulation

### MART Loss

$$
\mathcal{L}_{\text{MART}} = \text{BCE}(f_\theta(\mathbf{x}_{\text{adv}}), y) + \lambda \cdot (1 - p_y(\mathbf{x})) \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

where:
- $\text{BCE}$ is the Binary Cross-Entropy (boosted cross-entropy) on adversarial examples
- $p_y(\mathbf{x}) = \text{softmax}(f_\theta(\mathbf{x}))_y$ is the clean probability of the true class
- $(1 - p_y(\mathbf{x}))$ is the **misclassification-aware weight**
- $\lambda$ is the regularization strength

### Weight Interpretation

The weight $(1 - p_y(\mathbf{x}))$ acts as an attention mechanism:

| $p_y(\mathbf{x})$ | Weight | Interpretation |
|-------------------|--------|----------------|
| ~1.0 (confident correct) | ~0 | Low regularization—model is robust here |
| ~0.5 (uncertain) | ~0.5 | Moderate regularization |
| ~0 (misclassified) | ~1.0 | High regularization—most vulnerable |

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm

class MARTTrainer:
    """
    MART: Misclassification-Aware Robust Training.
    
    Focuses adversarial training effort on hard examples
    by weighting the consistency loss by (1 - p_y(x)).
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    lam : float
        Regularization strength for KL term
    epsilon : float
        Perturbation budget
    alpha : float
        PGD step size
    num_iter : int
        PGD iterations
    """
    
    def __init__(
        self,
        model: nn.Module,
        lam: float = 6.0,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iter: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.lam = lam
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def _pgd_attack(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Generate PGD adversarial examples."""
        x_adv = x + torch.empty_like(x).uniform_(
            -self.epsilon, self.epsilon
        )
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def _mart_loss(
        self,
        logits_adv: torch.Tensor,
        logits_clean: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MART loss.
        
        BCE(adv, y) + λ · (1 - p_y(x)) · KL(clean || adv)
        """
        # Boosted cross-entropy on adversarial examples
        adv_probs = F.softmax(logits_adv, dim=1)
        # BCE formulation: -log(p_y(x_adv)) scaled
        tmp = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(
            tmp[:, -1] == y, tmp[:, -2], tmp[:, -1]
        )
        loss_adv = F.cross_entropy(logits_adv, y) + \
                   F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        
        # Misclassification-aware weight
        with torch.no_grad():
            p_clean = F.softmax(logits_clean, dim=1)
            p_y = p_clean.gather(1, y.view(-1, 1)).squeeze()
            weight = 1.0 - p_y  # High weight for misclassified
        
        # Weighted KL divergence
        kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction='none'
        ).sum(dim=1)
        
        loss_kl = (weight * kl).mean()
        
        return loss_adv + self.lam * loss_kl
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch using MART loss."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='MART Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Clean forward pass
            logits_clean = self.model(x)
            
            # Generate adversarial examples
            x_adv = self._pgd_attack(x, y)
            
            # Adversarial forward pass
            logits_adv = self.model(x_adv)
            
            # MART loss
            loss = self._mart_loss(logits_adv, logits_clean, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (logits_clean.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({'loss': f'{total_loss/total:.4f}'})
        
        return {
            'loss': total_loss / total,
            'clean_accuracy': correct / total
        }
```

## Comparison of AT Methods

| Method | Key Idea | Clean Acc | Robust Acc | Hyperparameter |
|--------|----------|-----------|------------|----------------|
| Standard AT | Max-loss on adversarial | 85% | 48% | $\varepsilon$ |
| TRADES | Explicit trade-off | 87% | 46% | $\beta$ |
| MART | Focus on hard examples | 86% | 49% | $\lambda$ |

Results approximate for CIFAR-10, $\varepsilon = 8/255$.

## Summary

MART's misclassification-aware weighting provides targeted robustness improvement where it matters most—on examples that the model finds difficult. This makes MART particularly effective for datasets with class imbalance or varying difficulty levels, common in financial applications.

## References

1. Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020). "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." ICLR.
