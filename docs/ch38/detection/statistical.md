# Statistical Detection of Adversarial Examples

## Introduction

Rather than making models robust to adversarial perturbations, an alternative approach is to **detect** adversarial inputs before they reach the classifier. Statistical detection methods analyze properties of inputs and model internals to distinguish clean from adversarial examples.

## Detection Approaches

### Feature Distribution Analysis

Adversarial examples often produce **unusual activation patterns** in intermediate network layers. Detection methods compare the activation statistics of a test input against the distribution of clean data.

**Mahalanobis Distance Detector** (Lee et al., 2018):

For each class $c$ and layer $\ell$, fit a Gaussian to clean activations:

$$
(\boldsymbol{\mu}_c^\ell, \boldsymbol{\Sigma}^\ell) = \text{fit}(\{h^\ell(\mathbf{x}) : y = c\})
$$

The detection score combines Mahalanobis distances across layers:

$$
M(\mathbf{x}) = \sum_\ell \max_c \left[ -(h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell)^\top (\boldsymbol{\Sigma}^\ell)^{-1} (h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell) \right]
$$

Adversarial examples tend to have higher Mahalanobis distances (lower scores).

### Prediction Consistency

Check whether the model's predictions are **consistent** under input transformations that should not change the true class:

```python
import torch
import torch.nn as nn
from typing import Dict

class ConsistencyDetector:
    """
    Detect adversarial examples via prediction consistency
    under random transformations.
    
    Clean inputs maintain consistent predictions under
    small transformations; adversarial examples do not.
    """
    
    def __init__(
        self, model: nn.Module, num_transforms: int = 20,
        noise_std: float = 0.05, threshold: float = 0.7
    ):
        self.model = model
        self.num_transforms = num_transforms
        self.noise_std = noise_std
        self.threshold = threshold
        self.model.eval()
    
    def detect(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect adversarial examples.
        
        Returns consistency scores and binary detection decisions.
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            # Original prediction
            base_pred = self.model(x).argmax(dim=1)
            
            # Predictions under random noise
            consistent = torch.zeros(len(x), device=device)
            for _ in range(self.num_transforms):
                noise = torch.randn_like(x) * self.noise_std
                noisy_pred = self.model(x + noise).argmax(dim=1)
                consistent += (noisy_pred == base_pred).float()
            
            consistency_score = consistent / self.num_transforms
        
        return {
            'consistency_score': consistency_score,
            'is_adversarial': consistency_score < self.threshold,
            'base_prediction': base_pred
        }
```

### Logit Analysis

Adversarial examples often produce logit distributions that differ from clean inputs:

- **Higher entropy**: Less confident predictions (for some attacks)
- **Unusual logit gaps**: Abnormal margins between top classes
- **Different softmax distributions**: Detectable via statistical tests

## Limitations

Statistical detectors face a fundamental challenge: they can themselves be attacked. An **adaptive adversary** who knows the detection mechanism can craft adversarial examples that also fool the detector. This creates an arms race that favors the attacker in the white-box setting.

## Summary

Statistical detection provides a complementary layer of defense, particularly effective against non-adaptive attacks. However, it should not be relied upon as the sole defense mechanism, especially against sophisticated adversaries.

## References

1. Lee, K., et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS.
2. Ma, X., et al. (2018). "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality." ICLR.
3. Carlini, N., & Wagner, D. (2017). "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods." ACM Workshop on AI Security.
