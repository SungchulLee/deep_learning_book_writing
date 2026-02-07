# Fraud Detection Robustness

## Introduction

Fraud detection systems are inherently adversarial: fraudsters actively attempt to craft transactions that evade detection while maintaining fraudulent intent. This makes adversarial robustness not a theoretical concern but an operational necessity. Unlike image classification where adversarial examples are a research curiosity, fraud detection faces **real adversaries** who continuously adapt their strategies.

## Threat Model for Fraud Detection

### Adversary Profile

Fraudsters operate under specific constraints:

- **Knowledge**: Typically black-box or gray-box—fraudsters observe accept/reject decisions but rarely have model access
- **Goal**: Targeted evasion—make fraudulent transactions appear legitimate
- **Constraints**: Must maintain the fraudulent economic objective (e.g., money must actually transfer, stolen goods must be received)
- **Query budget**: Limited by the cost of each attempted fraud and risk of detection

### Formal Framework

Let $f_\theta: \mathbb{R}^d \to \{0, 1\}$ be a fraud detector where $f(\mathbf{x}) = 1$ indicates fraud. The adversary seeks:

$$
\mathbf{x}_{\text{evasion}} = \arg\min_{\mathbf{x}' \in \mathcal{C}} f_\theta(\mathbf{x}')
$$

subject to the constraint set $\mathcal{C}$ that preserves fraudulent intent (the transaction must still achieve the adversary's economic goal).

### Feature-Space Perturbations

Unlike image attacks with $\ell_p$ norms, fraud attacks operate in **feature space** with domain-specific constraints:

| Feature Type | Perturbable? | Constraint |
|-------------|-------------|------------|
| Transaction amount | Partially | Must achieve economic goal |
| Merchant category | Yes | Choose from valid categories |
| Time of day | Yes | Within operating hours |
| Device fingerprint | Yes | Spoof or use new device |
| IP geolocation | Yes | Use VPN/proxy |
| Transaction velocity | Partially | Must complete transactions |
| Card-present indicator | Fixed | Physical constraint |

## Adversarial Training for Fraud Detection

### Adapted AT Framework

Standard adversarial training must be adapted for tabular financial data:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class FraudRobustTrainer:
    """
    Adversarial training adapted for fraud detection.
    
    Key differences from image AT:
    - Feature-specific perturbation budgets
    - Constraint-aware perturbations (categorical features, valid ranges)
    - Asymmetric loss (false negatives are costlier than false positives)
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_budgets: torch.Tensor,
        categorical_mask: torch.Tensor,
        num_iter: int = 10,
        alpha_scale: float = 2.0,
        fn_weight: float = 10.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.feature_budgets = feature_budgets  # Per-feature epsilon
        self.categorical_mask = categorical_mask  # 1 for categorical
        self.num_iter = num_iter
        self.alpha_scale = alpha_scale
        self.fn_weight = fn_weight  # Weight for false negatives
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def _constrained_pgd(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        PGD with feature-specific constraints.
        
        Continuous features: perturbed within per-feature epsilon
        Categorical features: held fixed (or perturbed to valid values)
        """
        eps = self.feature_budgets.to(self.device)
        cat_mask = self.categorical_mask.to(self.device)
        alpha = self.alpha_scale * eps / self.num_iter
        
        # Initialize
        delta = torch.zeros_like(x)
        cont_mask = 1 - cat_mask
        
        # Random init for continuous features only
        delta = delta + cont_mask * torch.empty_like(x).uniform_(-1, 1) * eps
        
        for _ in range(self.num_iter):
            x_adv = (x + delta).requires_grad_(True)
            logits = self.model(x_adv)
            
            # Weighted loss: penalize evasion (fraudulent classified as legit)
            loss = F.cross_entropy(logits, y, reduction='none')
            # Upweight fraud examples (adversary tries to evade)
            weights = torch.where(y == 1, self.fn_weight, 1.0)
            loss = (weights * loss).mean()
            
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad.data
            
            with torch.no_grad():
                # Update only continuous features
                delta = delta + cont_mask * alpha * grad.sign()
                delta = torch.clamp(delta, -eps, eps) * cont_mask
        
        return torch.clamp(x + delta, 0, 1).detach()
    
    def train_epoch(self, train_loader, optimizer):
        """Train one epoch with fraud-aware adversarial training."""
        self.model.train()
        total_loss = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            # Generate adversarial examples
            x_adv = self._constrained_pgd(x, y)
            
            # Train on adversarial examples
            optimizer.zero_grad()
            logits = self.model(x_adv)
            
            # Asymmetric loss
            weights = torch.where(y == 1, self.fn_weight, 1.0)
            loss = (weights * F.cross_entropy(
                logits, y, reduction='none'
            )).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            total += len(y)
        
        return {'loss': total_loss / total}
```

## Evaluation Metrics for Robust Fraud Detection

Standard accuracy is insufficient for fraud detection. Relevant metrics under adversarial conditions:

| Metric | Definition | Target |
|--------|-----------|--------|
| Robust TPR | True positive rate under adversarial evasion | Maximize |
| FPR at threshold | False positive rate at operating threshold | Minimize |
| Robust AUPRC | Area under precision-recall curve under attack | Maximize |
| Evasion rate | Fraction of frauds that evade detection | Minimize |

## Practical Recommendations

1. **Use feature-specific budgets**: Not all features are equally perturbable
2. **Preserve categorical constraints**: Adversaries cannot arbitrarily change discrete features
3. **Asymmetric training**: Weight false negatives (missed fraud) much higher than false positives
4. **Monitor evasion patterns**: Track which features adversaries manipulate most
5. **Ensemble defenses**: Combine rule-based and ML-based detection for robustness

## References

1. Cartella, F., et al. (2021). "Adversarial Attacks on Fraud Detection Systems." Future Generation Computer Systems.
2. Chen, H., et al. (2020). "Robustness of Machine Learning Based Fraud Detection." ACM SIGKDD Workshop.
