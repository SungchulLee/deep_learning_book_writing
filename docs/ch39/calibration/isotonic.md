# Isotonic Regression for Calibration

## Overview

Isotonic regression is a non-parametric calibration method that fits a monotonically non-decreasing function mapping predicted probabilities to calibrated probabilities. Unlike temperature scaling (single parameter) or Platt scaling (two parameters), isotonic regression is more flexible but requires more calibration data.

## Mathematical Formulation

Given calibration data $(\hat{p}_i, y_i)$ where $\hat{p}_i$ is the model's predicted confidence and $y_i \in \{0,1\}$ is the correctness indicator, isotonic regression solves:

$$\min_{z_1 \leq z_2 \leq \ldots \leq z_n} \sum_{i=1}^n (z_i - y_i)^2$$

subject to the ordering constraint $z_{\pi(1)} \leq z_{\pi(2)} \leq \ldots$ where $\pi$ sorts by $\hat{p}_i$.

## Implementation

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch


class IsotonicCalibrator:
    """
    Isotonic regression calibration.
    
    Fits a non-parametric monotonic mapping from
    predicted probabilities to calibrated probabilities.
    """
    
    def __init__(self):
        self.calibrators = {}
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Fit per-class isotonic regression on validation set."""
        probs = torch.softmax(logits, dim=-1).numpy()
        labels_np = labels.numpy()
        n_classes = probs.shape[1]
        
        for c in range(n_classes):
            binary_labels = (labels_np == c).astype(float)
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs[:, c], binary_labels)
            self.calibrators[c] = ir
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply calibration to new predictions."""
        probs = torch.softmax(logits, dim=-1).numpy()
        calibrated = np.zeros_like(probs)
        
        for c, ir in self.calibrators.items():
            calibrated[:, c] = ir.predict(probs[:, c])
        
        # Renormalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-10)
        
        return torch.tensor(calibrated, dtype=torch.float32)
```

## When to Use Isotonic Regression

**Advantages**: Non-parametric, more flexible than temperature/Platt scaling, handles non-linear miscalibration.

**Disadvantages**: Requires more calibration data (~1000+ samples), can overfit with small calibration sets, multiclass extension requires per-class fitting.

**Recommendation**: Use when calibration set is large (>2000 samples) and miscalibration is non-linear.

## References

- Zadrozny, B., & Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." KDD.
