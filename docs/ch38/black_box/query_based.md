# Query-Based Attacks
## Introduction

**Query-based black-box attacks** generate adversarial examples by iteratively querying the target model and using the responses to guide the perturbation search. Unlike transfer attacks (zero-query), these methods directly interact with the target model but require no knowledge of its internals. The key challenge is achieving high attack success rates within a limited **query budget**.

## Attack Taxonomy

Query-based attacks are categorized by the information available from each query:

| Type | Information per Query | Query Efficiency | Attack Strength |
|------|----------------------|------------------|-----------------|
| **Score-based** | Full probability vector $p(y|\mathbf{x})$ | High | Strong |
| **Decision-based** | Hard label only $\hat{y} = \arg\max_y p(y|\mathbf{x})$ | Low | Moderate |
| **Top-$k$ based** | Top $k$ classes and scores | Medium | Strong |

## Gradient Estimation via Finite Differences

The core idea of score-based attacks is to **estimate gradients** using only function evaluations. Given access to the loss $\mathcal{L}(f(\mathbf{x}), y)$, we can estimate partial derivatives:

**Forward differences:**

$$
\frac{\partial \mathcal{L}}{\partial x_i} \approx \frac{\mathcal{L}(f(\mathbf{x} + h\mathbf{e}_i), y) - \mathcal{L}(f(\mathbf{x}), y)}{h}
$$

**Central differences (more accurate):**

$$
\frac{\partial \mathcal{L}}{\partial x_i} \approx \frac{\mathcal{L}(f(\mathbf{x} + h\mathbf{e}_i), y) - \mathcal{L}(f(\mathbf{x} - h\mathbf{e}_i), y)}{2h}
$$

**Cost:** $O(d)$ queries per gradient estimate for $d$-dimensional input. For images with $d \approx 3 \times 32 \times 32 = 3{,}072$ (CIFAR-10), this is expensive but feasible.

### Random Direction Estimation (NES)

Natural Evolution Strategies (NES) estimate gradients more efficiently using random directions:

$$
\nabla_\mathbf{x} \mathcal{L} \approx \frac{1}{n\sigma} \sum_{i=1}^n \mathcal{L}(f(\mathbf{x} + \sigma \mathbf{u}_i), y) \cdot \mathbf{u}_i
$$

where $\mathbf{u}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ are random Gaussian directions. This requires only $n$ queries (typically $n = 50\text{-}200$) regardless of input dimension.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class NESAttack:
    """
    Natural Evolution Strategies (NES) based black-box attack.
    
    Estimates gradients using random direction sampling,
    then performs PGD-style updates.
    
    Parameters
    ----------
    model : nn.Module
        Target model (treated as black-box query oracle)
    epsilon : float
        Perturbation budget (Linf)
    num_samples : int
        Number of random directions per gradient estimate
    sigma : float
        Sampling standard deviation for NES
    step_size : float
        Attack step size
    max_queries : int
        Maximum total queries allowed
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        num_samples: int = 100,
        sigma: float = 0.001,
        step_size: float = 0.01,
        max_queries: int = 10000,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.sigma = sigma
        self.step_size = step_size
        self.max_queries = max_queries
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
        self.total_queries = 0
    
    def _query(self, x: torch.Tensor) -> torch.Tensor:
        """Query model and return logits (simulates API call)."""
        self.total_queries += x.shape[0]
        with torch.no_grad():
            return self.model(x.to(self.device))
    
    def _estimate_gradient(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate gradient using NES with antithetic sampling.
        
        Uses pairs (u, -u) for variance reduction.
        """
        batch_size = x.shape[0]
        grad_estimate = torch.zeros_like(x)
        
        for _ in range(self.num_samples // 2):
            # Random direction
            u = torch.randn_like(x)
            
            # Forward and backward queries
            x_plus = x + self.sigma * u
            x_minus = x - self.sigma * u
            
            logits_plus = self._query(x_plus)
            logits_minus = self._query(x_minus)
            
            # Loss difference
            loss_plus = F.cross_entropy(logits_plus, y, reduction='none')
            loss_minus = F.cross_entropy(logits_minus, y, reduction='none')
            
            # NES gradient estimate (antithetic)
            diff = (loss_plus - loss_minus).view(-1, 1, 1, 1)
            grad_estimate += diff * u
        
        grad_estimate /= (self.num_samples * self.sigma)
        return grad_estimate
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Generate adversarial examples using NES-PGD."""
        x = x.to(self.device)
        y = y.to(self.device)
        self.total_queries = 0
        
        x_adv = x.clone()
        
        max_iter = self.max_queries // self.num_samples
        
        for t in range(max_iter):
            if self.total_queries >= self.max_queries:
                break
            
            # Check which examples still need attacking
            with torch.no_grad():
                pred = self.model(x_adv).argmax(dim=1)
                still_correct = (pred == y)
            
            if not still_correct.any():
                break
            
            # Estimate gradient
            grad = self._estimate_gradient(x_adv, y)
            
            # PGD-style update
            x_adv = x_adv + self.step_size * grad.sign()
            
            # Project
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate attack with query count."""
        with torch.no_grad():
            clean_pred = self.model(x.to(self.device)).argmax(1)
            adv_pred = self.model(x_adv.to(self.device)).argmax(1)
            y_dev = y.to(self.device)
        
        return {
            'clean_accuracy': (clean_pred == y_dev).float().mean().item(),
            'robust_accuracy': (adv_pred == y_dev).float().mean().item(),
            'attack_success_rate': (adv_pred != y_dev).float().mean().item(),
            'total_queries': self.total_queries,
            'avg_queries_per_example': self.total_queries / len(x)
        }
```

## Query Efficiency Comparison

| Method | Queries per Example | Success Rate | Gradient Access |
|--------|---------------------|--------------|-----------------|
| Coordinate-wise FD | $O(d)$ per iteration | High | Full estimate |
| NES (100 samples) | ~100 per iteration | High | Approximate |
| Bandits (prior) | ~50 per iteration | High | Data-dependent |
| Square Attack | ~1 per iteration | Moderate-High | None |

## Practical Considerations

### Query Budget in Financial Settings

Financial APIs typically impose rate limits. Realistic query budgets for financial models:

| Setting | Typical Budget | Rationale |
|---------|---------------|-----------|
| Public credit API | 100-1000 queries | Rate-limited endpoints |
| Trading signal API | 10-100 queries | Real-time latency constraints |
| Internal model audit | 10,000+ queries | Controlled evaluation |

### Defenses Against Query Attacks

- **Query detection**: Monitor for suspicious patterns of similar inputs
- **Rate limiting**: Restrict the number of queries per user/session
- **Output perturbation**: Add noise to model outputs to hinder gradient estimation
- **Stateful detection**: Track query sequences and flag anomalous access patterns

## References

1. Ilyas, A., et al. (2018). "Black-Box Adversarial Attacks with Limited Queries and Information." ICML.
2. Ilyas, A., Engstrom, L., & Madry, A. (2019). "Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors." ICLR.
3. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.

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
