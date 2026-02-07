# Stability Evaluation

## Introduction

An explanation method is **stable** if similar inputs produce similar explanations. If adding imperceptible noise completely changes the explanation, practitioners cannot trust it. Stability is essential for regulated environments where explanations must be reproducible.

## Metrics

### Relative Input Stability (RIS)

$$
\text{RIS}(x, x') = \frac{\|E(x) - E(x')\|_2}{\|E(x)\|_2 \cdot \|x - x'\|_2}
$$

Lower RIS = more stable.

### Max-Sensitivity

$$
\text{MaxSens}(x) = \max_{\|\epsilon\| \leq r} \|E(x) - E(x + \epsilon)\|_2
$$

### Implementation

```python
import torch
import numpy as np

def compute_stability(
    explanation_fn, input_tensor, n_perturbations=50, noise_level=0.01
):
    """Measure explanation stability under input perturbations."""
    base_explanation = explanation_fn(input_tensor)
    base_norm = np.linalg.norm(base_explanation)
    
    relative_changes = []
    for _ in range(n_perturbations):
        noise = torch.randn_like(input_tensor) * noise_level
        perturbed_explanation = explanation_fn(input_tensor + noise)
        
        explanation_change = np.linalg.norm(base_explanation - perturbed_explanation)
        input_change = noise.norm().item()
        
        if base_norm > 0 and input_change > 0:
            relative_changes.append(explanation_change / (base_norm * input_change))
    
    return {
        'mean_ris': np.mean(relative_changes),
        'max_ris': np.max(relative_changes),
        'std_ris': np.std(relative_changes)
    }
```

## Method Stability Comparison

| Method | Typical Stability | Why |
|--------|------------------|-----|
| Vanilla Gradients | Low | Noisy by construction |
| SmoothGrad | High | Averaging reduces sensitivity |
| Integrated Gradients | Medium | Path-dependent, baseline matters |
| SHAP | Medium-High | Sampling introduces some variance |
| LIME | Low-Medium | Random sampling, kernel width sensitive |
| Grad-CAM | High | Spatial averaging smooths output |

## Summary

Stability evaluation ensures explanations are robust to minor input variations. Methods like SmoothGrad and Grad-CAM inherently improve stability through averaging.

## References

1. Alvarez-Melis, D., & Jaakkola, T. S. (2018). "On the Robustness of Interpretability Methods." *ICML Workshop*.
2. Yeh, C. K., et al. (2019). "On the (In)fidelity and Sensitivity of Explanations." *NeurIPS*.
