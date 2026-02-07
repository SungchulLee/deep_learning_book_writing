# Reliability Diagrams

## Overview

Reliability diagrams visualize calibration by plotting observed frequency against predicted probability. A perfectly calibrated model produces points along the diagonal.

## Construction

1. Bin predictions by confidence level (typically 10-15 bins)
2. For each bin, compute the average confidence and the actual accuracy
3. Plot accuracy vs confidence; deviation from diagonal indicates miscalibration

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def plot_reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = 'Reliability Diagram'
) -> Tuple[plt.Figure, float]:
    """
    Plot reliability diagram with ECE annotation.
    
    Below diagonal = overconfident
    Above diagonal = underconfident
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies, avg_confidences, counts = [], [], []
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            accuracies.append((predictions[mask] == labels[mask]).mean())
            avg_confidences.append(confidences[mask].mean())
            counts.append(mask.sum())
        else:
            accuracies.append(0)
            avg_confidences.append(bin_centers[i])
            counts.append(0)
    
    # Compute ECE
    total = sum(counts)
    ece = sum(c * abs(a - conf) for a, conf, c in 
              zip(accuracies, avg_confidences, counts) if c > 0) / total
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers, accuracies, width=0.08, alpha=0.7, 
           label='Observed accuracy')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\nECE = {ece:.4f}')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return fig, ece
```

## Interpreting Reliability Diagrams

- **Bars below diagonal**: Model is overconfident in that range
- **Bars above diagonal**: Model is underconfident
- **Gap area**: Proportional to ECE
- **Uniform bar heights**: Well-calibrated model

## References

- DeGroot, M. H., & Fienberg, S. E. (1983). "The Comparison and Evaluation of Forecasters." The Statistician.
