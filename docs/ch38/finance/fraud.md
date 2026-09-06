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

## Exercises

**Exercise 1.**
For a linear classifier $f(x) = w^T x + b$, compute the minimal $\ell_\infty$ perturbation needed to change the predicted class. Relate this to the robustness of neural networks.

??? success "Solution to Exercise 1"
    For a linear classifier, the distance to the decision boundary under $\ell_\infty$ norm is $\frac{|w^T x + b|}{\|w\|_1}$. The minimal perturbation is $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$. For neural networks, the local linear approximation $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$ explains why FGSM (which uses the sign of the gradient) is effective. The vulnerability of high-dimensional models comes from the fact that $\|w\|_1$ grows with dimension while $|w^T x + b|$ does not necessarily, making the robustness margin shrink. $\square$

---

**Exercise 2.**
Implement the attack or defense described in this section for a ResNet-18 model on CIFAR-10. Report clean accuracy and robust accuracy under PGD-20 attack with $\epsilon = 8/255$.

??? success "Solution to Exercise 2"
    A standard ResNet-18 achieves $\sim$93% clean accuracy but $\sim$0% robust accuracy under PGD-20 ($\epsilon = 8/255$, step size $2/255$). After applying the method from this section, typical results depend on the specific technique: adversarial training achieves $\sim$83% clean / $\sim$50% robust; certified defenses achieve lower but provable bounds. The accuracy-robustness trade-off is fundamental: improving robustness typically costs 5--15% clean accuracy. Report results averaged over 3 random seeds with standard errors. $\square$

---

**Exercise 3.**
Prove that no defense can simultaneously achieve high accuracy on clean data and high robustness against $\ell_\infty$ perturbations without increasing model capacity, under the assumption that the data distribution has overlapping class-conditional supports within the perturbation ball.

??? success "Solution to Exercise 3"
    If two classes have support overlap within distance $\epsilon$ (i.e., $\exists x_1 \in \text{class 1}, x_2 \in \text{class 2}$ with $\|x_1 - x_2\|_\infty \leq 2\epsilon$), then any classifier robust at both $x_1$ and $x_2$ must misclassify at least one of them (the perturbation balls overlap). This is the fundamental accuracy-robustness trade-off: the fraction of overlapping support determines the unavoidable accuracy loss. For natural image distributions, significant overlap exists at $\epsilon = 8/255$, explaining the observed 10--15% accuracy drop. Increased model capacity (wider networks) can better approximate the complex robust decision boundary, partially mitigating the trade-off. $\square$

---

**Exercise 4.**
Discuss how adversarial robustness concerns manifest in a financial machine learning system (e.g., fraud detection or trading signal generation). How does the threat model differ from computer vision?

??? success "Solution to Exercise 4"
    In finance, adversaries are strategic agents (fraudsters, market manipulators) who actively adapt to detection systems. Key differences from vision: (1) the perturbation space is constrained by what is economically feasible (a fraudster cannot change their entire transaction history); (2) attacks are sequential and adaptive (the adversary observes the system's response and adjusts); (3) the cost of false positives and false negatives is asymmetric (blocking a legitimate transaction vs. missing fraud); (4) $\ell_p$ norms are not meaningful -- domain-specific perturbation models are needed. Defenses must be robust to adaptive adversaries, which rules out many detection-based approaches that can be evaded once the detection criterion is known. $\square$
