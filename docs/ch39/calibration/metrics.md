# Calibration Metrics

## Expected Calibration Error (ECE)

The most common calibration metric, measuring the average gap between confidence and accuracy across bins:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

```python
import torch
import numpy as np
from typing import Tuple, Dict


def compute_ece(
    confidences: np.ndarray, predictions: np.ndarray,
    labels: np.ndarray, n_bins: int = 15
) -> Tuple[float, Dict]:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = (predictions[mask] == labels[mask]).mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()
            ece += (bin_count / len(confidences)) * abs(bin_acc - bin_conf)
            bin_data.append({'acc': bin_acc, 'conf': bin_conf, 'count': bin_count})
    
    return ece, bin_data
```

## Maximum Calibration Error (MCE)

$$\text{MCE} = \max_{m \in \{1,\ldots,M\}} |\text{acc}(B_m) - \text{conf}(B_m)|$$

Captures the worst-case calibration gap. Important for safety-critical applications.

## Brier Score

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2$$

A proper scoring rule that jointly evaluates calibration and sharpness.

```python
def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))
```

## Adaptive ECE

Standard ECE with equal-width bins can be misleading when most predictions fall in a narrow confidence range. Adaptive ECE uses equal-count bins instead:

```python
def compute_adaptive_ece(confidences, predictions, labels, n_bins=15):
    """ECE with equal-count (adaptive) binning."""
    sorted_idx = np.argsort(confidences)
    bin_size = len(confidences) // n_bins
    
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(confidences)
        idx = sorted_idx[start:end]
        
        if len(idx) > 0:
            bin_acc = (predictions[idx] == labels[idx]).mean()
            bin_conf = confidences[idx].mean()
            ece += (len(idx) / len(confidences)) * abs(bin_acc - bin_conf)
    
    return ece
```

## Comparison of Metrics

| Metric | Type | Captures | Range |
|--------|------|----------|-------|
| ECE | Binned | Average miscalibration | [0, 1] |
| MCE | Binned | Worst-case miscalibration | [0, 1] |
| Brier Score | Proper scoring | Calibration + sharpness | [0, 2] |
| NLL | Proper scoring | Calibration + sharpness | [0, ∞) |

## References

- Naeini, M. P., et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." AAAI.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
