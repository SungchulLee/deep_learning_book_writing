# TRADES: Theoretically Principled Trade-off

## Introduction

**TRADES** (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization; Zhang et al., 2019) explicitly decomposes the robust optimization objective to control the trade-off between clean accuracy and adversarial robustness. Unlike standard adversarial training which implicitly balances these goals, TRADES provides a tunable parameter $\beta$ that directly controls this trade-off.

## Mathematical Foundation

### Decomposition of Robust Risk

The key insight of TRADES is that robust risk can be decomposed:

$$
R_{\text{rob}}(f) \leq R_{\text{std}}(f) + R_{\text{boundary}}(f)
$$

where $R_{\text{boundary}}$ measures the probability that a perturbation can push a correctly-classified point across the decision boundary. This suggests optimizing both terms separately.

### TRADES Formulation

TRADES minimizes:

$$
\mathcal{L}_{\text{TRADES}} = \underbrace{\mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}), y)}_{\text{clean accuracy}} + \beta \cdot \underbrace{\text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))}_{\text{local smoothness}}
$$

where:
- The first term ensures predictions are correct on clean inputs (standard cross-entropy)
- The second term encourages **consistent predictions** between clean and perturbed inputs via KL divergence
- $\beta > 0$ controls the trade-off: larger $\beta$ emphasizes robustness at the cost of clean accuracy

### Adversarial Example Generation in TRADES

The adversarial examples $\mathbf{x}_{\text{adv}}$ are generated to **maximize** the KL divergence (not cross-entropy):

$$
\mathbf{x}_{\text{adv}} = \arg\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x} + \boldsymbol{\delta}))
$$

This is solved approximately via PGD on the KL objective.

## Comparison with Standard AT

| Aspect | Standard AT | TRADES |
|--------|-------------|--------|
| Training loss | $\mathcal{L}_{\text{CE}}(f(\mathbf{x}_{\text{adv}}), y)$ | $\mathcal{L}_{\text{CE}}(f(\mathbf{x}), y) + \beta \cdot \text{KL}$ |
| Attack target | Maximize CE loss | Maximize KL divergence |
| Trade-off control | Implicit (via $\varepsilon$) | Explicit (via $\beta$) |
| Clean accuracy | Lower | Higher |
| Robust accuracy | Slightly higher | Slightly lower |
| Theoretical justification | Min-max | Surrogate loss decomposition |

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm

class TRADESTrainer:
    """
    TRADES: Theoretically Principled Trade-off.
    
    Loss = L_CE(f(x), y) + β · KL(f(x) || f(x_adv))
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    beta : float
        Trade-off parameter (1-6 typical; 6 = more robust)
    epsilon : float
        Perturbation budget
    alpha : float
        PGD step size for generating x_adv
    num_iter : int
        PGD iterations for generating x_adv
    """
    
    def __init__(
        self,
        model: nn.Module,
        beta: float = 6.0,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iter: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def _generate_trades_adversary(
        self,
        x: torch.Tensor,
        logits_clean: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples by maximizing KL divergence.
        
        Unlike standard AT which maximizes CE, TRADES maximizes
        the KL divergence between clean and adversarial predictions.
        """
        x_adv = x.clone().detach()
        x_adv += torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # Detached clean softmax (target distribution)
        p_clean = F.softmax(logits_clean.detach(), dim=1)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            
            # KL divergence: KL(clean || adversarial)
            loss_kl = F.kl_div(
                F.log_softmax(self.model(x_adv), dim=1),
                p_clean,
                reduction='batchmean'
            )
            
            self.model.zero_grad()
            loss_kl.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch using TRADES loss."""
        self.model.train()
        total_loss = 0
        total_natural = 0
        total_robust = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='TRADES Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Clean forward pass
            logits_clean = self.model(x)
            loss_natural = F.cross_entropy(logits_clean, y)
            
            # Generate adversarial examples (maximize KL)
            x_adv = self._generate_trades_adversary(x, logits_clean)
            
            # TRADES loss
            logits_adv = self.model(x_adv)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction='batchmean'
            )
            
            loss = loss_natural + self.beta * loss_robust
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            total_natural += loss_natural.item() * len(y)
            total_robust += loss_robust.item() * len(y)
            correct += (logits_clean.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'nat': f'{total_natural/total:.4f}',
                'rob': f'{total_robust/total:.4f}'
            })
        
        return {
            'total_loss': total_loss / total,
            'natural_loss': total_natural / total,
            'robust_loss': total_robust / total,
            'clean_accuracy': correct / total
        }
```

## Choosing $\beta$

The $\beta$ parameter provides direct control over the accuracy-robustness trade-off:

| $\beta$ | Clean Accuracy | Robust Accuracy | Use Case |
|---------|---------------|----------------|----------|
| 1.0 | ~89% | ~43% | Prioritize clean accuracy |
| 3.0 | ~88% | ~45% | Balanced |
| 6.0 | ~87% | ~46% | Standard recommendation |
| 10.0 | ~85% | ~47% | Prioritize robustness |

Results on CIFAR-10 with $\varepsilon = 8/255$.

## Summary

TRADES provides an explicit, principled mechanism for controlling the clean-robust accuracy trade-off. Its decomposition of robust risk into natural and boundary components enables fine-grained control via $\beta$, making it the preferred method when clean accuracy must be preserved alongside robustness.

## References

1. Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML.
