# Certified Accuracy

## Introduction

**Certified accuracy** quantifies the fraction of test examples for which a model's prediction is provably correct under any perturbation within a given budget. Unlike empirical robust accuracy (which depends on the attack strength), certified accuracy provides a guaranteed lower bound on true robustness.

## Formal Definition

### Certified Accuracy at Radius $r$

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

### CIFAR-10 ($\ell_2$, Randomized Smoothing)

| Method | $\sigma$ | Cert. @ $r{=}0.25$ | Cert. @ $r{=}0.5$ | Cert. @ $r{=}1.0$ |
|--------|----------|---------------------|--------------------|--------------------|
| Cohen et al. | 0.25 | 60% | 43% | — |
| Salman et al. | 0.25 | 68% | 49% | — |
| Cohen et al. | 0.50 | 54% | 41% | 26% |
| Salman et al. | 0.50 | 59% | 44% | 32% |

### CIFAR-10 ($\ell_\infty$, IBP/CROWN)

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
