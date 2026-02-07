# Platt Scaling

## Overview

Platt scaling fits a logistic regression model to map uncalibrated scores to calibrated probabilities. Originally designed for SVMs, it applies a sigmoid transformation with two learnable parameters.

## Mathematical Formulation

For binary classification with score $s$:

$$P(y = 1 | s) = \frac{1}{1 + \exp(as + b)}$$

Parameters $a$ and $b$ are fit by minimizing the negative log-likelihood on a calibration set.

For multiclass, Platt scaling generalizes to learning an affine transformation of the logit vector:

$$\hat{p}_c = \frac{\exp(a_c z_c + b_c)}{\sum_j \exp(a_j z_j + b_j)}$$

When $a_c = 1/T$ and $b_c = 0$ for all classes, this reduces to temperature scaling.

## Implementation

```python
import torch
import torch.nn as nn


class PlattScaling(nn.Module):
    """
    Platt scaling with per-class affine transformation.
    
    Generalizes temperature scaling by allowing different
    scale and shift per class.
    """
    
    def __init__(self, n_classes: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scaled = logits * self.a + self.b
        return torch.softmax(scaled, dim=-1)
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 200):
        """Fit on validation set."""
        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled = logits * self.a + self.b
            loss = criterion(scaled, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return self.a.data.clone(), self.b.data.clone()
```

## Comparison with Other Methods

| Method | Parameters | Flexibility | Data Needed |
|--------|-----------|-------------|-------------|
| Temperature Scaling | 1 | Low | Small (~500) |
| Platt Scaling | 2K | Medium | Medium (~1000) |
| Isotonic Regression | Non-parametric | High | Large (~2000+) |

## References

- Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines."
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
