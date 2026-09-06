# Certified Accuracy
## Introduction

**Certified accuracy** quantifies the fraction of test examples for which a model's prediction is provably correct under any perturbation within a given budget. Unlike empirical robust accuracy (which depends on the attack strength), certified accuracy provides a guaranteed lower bound on true robustness.

## Formal Definition

### Certified Accuracy at Radius r

$$
\text{Certified Acc}(r) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left[f(\mathbf{x}_i) = y_i \text{ and } R(\mathbf{x}_i) \geq r\right]
$$

where $R(\mathbf{x}_i)$ is the certified radius at example $i$.

### Relationship to Other Metrics

$$
\text{Certified Acc}(r) \leq \text{True Robust Acc}(r) \leq \text{Empirical Robust Acc}(r)
$$

- **Certified accuracy** is a lower bound: some truly robust predictions may not be certifiable
- **Empirical robust accuracy** is an upper bound: attacks may not find the optimal adversarial example
- The gap measures the "certification gap"

## Computing Certified Accuracy

### For Randomized Smoothing

```python
import torch
from typing import Dict, List

def compute_certified_accuracy(
    smoother,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    radii: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    n: int = 10000,
    alpha: float = 0.001
) -> Dict[str, float]:
    """
    Compute certified accuracy at multiple radii.
    
    Parameters
    ----------
    smoother : RandomizedSmoothing
        Smoothed classifier with certification capability
    test_images, test_labels : torch.Tensor
        Test dataset
    radii : list[float]
        Radii at which to compute certified accuracy
    n : int
        Monte Carlo samples for certification
    alpha : float
        Confidence level
    
    Returns
    -------
    results : dict mapping radius to certified accuracy
    """
    num_examples = len(test_images)
    predictions = []
    certified_radii = []
    
    for i in range(num_examples):
        pred, cert_radius = smoother.certify(
            test_images[i], n=n, alpha=alpha
        )
        predictions.append(pred)
        certified_radii.append(cert_radius)
    
    predictions = torch.tensor(predictions)
    certified_radii = torch.tensor(certified_radii)
    correct = (predictions == test_labels)
    
    results = {'clean_accuracy': correct.float().mean().item()}
    
    for r in radii:
        certified_at_r = correct & (certified_radii >= r)
        results[f'certified_r={r}'] = certified_at_r.float().mean().item()
    
    if correct.any():
        results['avg_radius'] = certified_radii[correct].mean().item()
    else:
        results['avg_radius'] = 0.0
    
    return results
```

### For IBP/CROWN

```python
def certified_accuracy_ibp(model, test_loader, epsilon, device='cuda'):
    """Compute certified accuracy using IBP bounds."""
    certified = 0
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        lb, ub = compute_ibp_bounds(model, x, epsilon)
        
        with torch.no_grad():
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
        
        true_lb = lb.gather(1, y.view(-1, 1)).squeeze()
        ub_copy = ub.clone()
        ub_copy.scatter_(1, y.view(-1, 1), float('-inf'))
        max_other_ub = ub_copy.max(dim=1)[0]
        
        is_certified = (true_lb > max_other_ub) & (pred == y)
        certified += is_certified.sum().item()
        total += len(y)
    
    return {
        'clean_accuracy': correct / total,
        'certified_accuracy': certified / total
    }
```

## Benchmarks

### CIFAR-10 (L2, Randomized Smoothing)

| Method | $\sigma$ | Cert. @ $r{=}0.25$ | Cert. @ $r{=}0.5$ | Cert. @ $r{=}1.0$ |
|--------|----------|---------------------|--------------------|--------------------|
| Cohen et al. | 0.25 | 60% | 43% | — |
| Salman et al. | 0.25 | 68% | 49% | — |
| Cohen et al. | 0.50 | 54% | 41% | 26% |
| Salman et al. | 0.50 | 59% | 44% | 32% |

### CIFAR-10 (L-infinity, IBP/CROWN)

| Method | $\varepsilon$ | Certified Accuracy |
|--------|--------------|-------------------|
| IBP | 2/255 | 33% |
| CROWN-IBP | 2/255 | 38% |
| IBP | 8/255 | 7% |
| CROWN-IBP | 8/255 | 12% |

## Summary

| Metric | Guarantee | Cost | Tightness |
|--------|-----------|------|-----------|
| Empirical robust acc | None | Low-moderate | Upper bound |
| Certified acc (RS) | Probabilistic | High | Moderate |
| Certified acc (IBP) | Deterministic | Low | Loose |
| Certified acc (CROWN) | Deterministic | Moderate | Tighter |

Certified accuracy is the most rigorous robustness measure, providing guarantees that hold against any attack within the perturbation budget.

## References

1. Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.
2. Gowal, S., et al. (2019). "Scalable Verified Training for Provably Robust Image Classification." ICCV.
3. Salman, H., et al. (2019). "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." NeurIPS.

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
