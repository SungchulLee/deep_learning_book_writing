# Decision-Based Attacks
## Introduction

**Decision-based attacks** operate in the most restrictive black-box setting: the adversary observes only the model's **hard label** prediction $\hat{y} = \arg\max_y f(\mathbf{x})$, without access to confidence scores or probabilities. Despite this severe information limitation, decision-based attacks can find adversarial examples with competitive perturbation magnitudes, albeit requiring more queries than score-based methods.

## Mathematical Framework

### Problem Setting

Given only a decision oracle:

$$
\mathcal{O}(\mathbf{x}) = \arg\max_y f_\theta(\mathbf{x})_y
$$

the attacker must find $\mathbf{x}_{\text{adv}}$ such that $\mathcal{O}(\mathbf{x}_{\text{adv}}) \neq y$ and $\|\mathbf{x}_{\text{adv}} - \mathbf{x}\|_p$ is minimized.

No gradient information or continuous loss signal is available—only a binary signal of whether a given point is classified differently from the true label.

## Boundary Attack

The **Boundary Attack** (Brendel et al., 2018) performs a random walk along the decision boundary, progressively reducing the distance to the original input.

### Algorithm

1. Start from a point known to be adversarial (e.g., a random image from a different class)
2. At each iteration:
    - **Orthogonal step**: Perturb in a direction orthogonal to the line connecting $\mathbf{x}_{\text{adv}}$ and $\mathbf{x}_0$
    - **Step toward source**: Move slightly toward $\mathbf{x}_0$
    - **Accept/reject**: Keep the update only if the new point remains adversarial
3. Adapt step sizes to maintain approximately 50% acceptance rate

The boundary attack requires $O(10^3\text{-}10^5)$ queries per example but requires no score information.

## HopSkipJumpAttack

**HopSkipJump** (Chen et al., 2020) improves upon the Boundary Attack by estimating the **boundary normal** direction using only decision queries.

### Key Idea

At a point near the decision boundary, approximate the boundary normal via:

$$
\hat{\nabla} \phi(\mathbf{x}) \approx \frac{1}{B} \sum_{b=1}^B \text{sign}\left(\mathcal{O}(\mathbf{x} + \delta \mathbf{u}_b) \neq y\right) \cdot \mathbf{u}_b
$$

where $\mathbf{u}_b$ are random unit vectors and $\delta$ is a small step. This Monte Carlo estimate of the boundary normal enables gradient-like updates using only hard labels.

### Algorithm Steps

1. **Binary search**: Find a point on the decision boundary via binary search between $\mathbf{x}_0$ and $\mathbf{x}_{\text{adv}}$
2. **Gradient estimation**: Estimate boundary normal using random sampling
3. **Boundary step**: Move along the estimated gradient direction
4. **Binary search**: Project back to the boundary
5. Repeat with decreasing step sizes

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class BoundaryAttack:
    """
    Simplified Boundary Attack for decision-based black-box setting.
    
    Performs a random walk along the decision boundary,
    progressively reducing distance to the original input.
    
    Parameters
    ----------
    model : nn.Module
        Target model (only hard labels used)
    max_queries : int
        Maximum number of model queries
    init_delta : float
        Initial step size for orthogonal steps
    init_epsilon : float  
        Initial step size toward source
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_queries: int = 25000,
        init_delta: float = 0.1,
        init_epsilon: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.max_queries = max_queries
        self.delta = init_delta
        self.epsilon_step = init_epsilon
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.queries = 0
    
    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard label prediction (simulates decision oracle)."""
        self.queries += x.shape[0]
        with torch.no_grad():
            return self.model(x.to(self.device)).argmax(dim=1)
    
    def _is_adversarial(
        self, x: torch.Tensor, true_label: int
    ) -> bool:
        """Check if x is adversarial (different from true label)."""
        pred = self._predict(x.unsqueeze(0))
        return pred.item() != true_label
    
    def _binary_search_boundary(
        self, x_orig: torch.Tensor, x_adv: torch.Tensor,
        true_label: int, tol: float = 1e-4
    ) -> torch.Tensor:
        """Binary search to find point on decision boundary."""
        low, high = 0.0, 1.0
        
        for _ in range(20):  # ~20 queries for binary search
            mid = (low + high) / 2
            x_mid = (1 - mid) * x_orig + mid * x_adv
            
            if self._is_adversarial(x_mid, true_label):
                high = mid
            else:
                low = mid
        
        return (1 - high) * x_orig + high * x_adv
    
    def _attack_single(
        self, x: torch.Tensor, true_label: int
    ) -> torch.Tensor:
        """Attack a single example."""
        x = x.to(self.device)
        C, H, W = x.shape
        
        # Initialize: find an adversarial starting point
        # (random image from input space)
        for _ in range(100):
            x_init = torch.rand_like(x)
            if self._is_adversarial(x_init, true_label):
                break
        else:
            return x  # Could not find adversarial init
        
        # Binary search to boundary
        x_adv = self._binary_search_boundary(x, x_init, true_label)
        
        # Iterative refinement
        while self.queries < self.max_queries:
            # Orthogonal perturbation
            noise = torch.randn_like(x)
            # Remove component along x_adv - x
            direction = x_adv - x
            direction_flat = direction.view(-1)
            noise_flat = noise.view(-1)
            noise_flat = noise_flat - (noise_flat @ direction_flat) / \
                        (direction_flat @ direction_flat + 1e-8) * direction_flat
            noise = noise_flat.view(x.shape)
            noise = noise / (noise.norm() + 1e-8)
            
            # Step along boundary
            d_norm = (x_adv - x).norm()
            x_candidate = x_adv + self.delta * d_norm * noise
            
            # Step toward source
            x_candidate = (1 - self.epsilon_step) * x_candidate + \
                         self.epsilon_step * x
            x_candidate = torch.clamp(x_candidate, 0, 1)
            
            # Accept if still adversarial and closer
            if self._is_adversarial(x_candidate, true_label):
                new_dist = (x_candidate - x).norm()
                old_dist = (x_adv - x).norm()
                if new_dist < old_dist:
                    x_adv = x_candidate
        
        return x_adv
    
    def generate(
        self, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Generate adversarial examples for a batch."""
        x_adv = x.clone()
        self.queries = 0
        
        for i in range(len(x)):
            x_adv[i] = self._attack_single(x[i], y[i].item())
        
        return x_adv.detach()
```

## Query Complexity Comparison

| Method | Queries per Example | Information Used | Attack Strength |
|--------|---------------------|-----------------|-----------------|
| Boundary Attack | $10^3 - 10^5$ | Hard labels | Moderate |
| HopSkipJump | $10^3 - 10^4$ | Hard labels | Strong |
| Sign-OPT | $10^3 - 10^4$ | Hard labels | Strong |
| GeoDA | $10^2 - 10^3$ | Hard labels | Moderate |

## When Decision-Based Attacks Apply

Decision-based attacks are the relevant threat model when:

- The target system returns only final decisions (approve/reject, buy/sell)
- Confidence scores are not exposed to end users
- API responses are intentionally limited to prevent score-based attacks

### Financial Examples

- **Credit decisioning**: Applicant sees only approved/denied, not the score
- **Fraud alerts**: Transaction is flagged or cleared with no probability
- **Trading signals**: System outputs buy/hold/sell without confidence

## Summary

| Method | Queries | Labels Needed | Best For |
|--------|---------|---------------|----------|
| Boundary Attack | High | Hard only | Initial exploration |
| HopSkipJump | Moderate | Hard only | Efficient boundary attacks |
| Sign-OPT | Moderate | Hard only | Targeted attacks |

Decision-based attacks demonstrate that even the most restricted information access does not prevent adversarial attacks—it only increases the query cost. This has important implications for API design in security-sensitive applications.

## References

1. Brendel, W., Rauber, J., & Bethge, M. (2018). "Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models." ICLR.
2. Chen, J., et al. (2020). "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack." IEEE S&P.
3. Cheng, M., et al. (2019). "Sign-OPT: A Query-Efficient Hard-Label Adversarial Attack." ICLR.

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
