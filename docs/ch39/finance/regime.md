# Uncertainty-Based Regime Detection

## Overview

Rising epistemic uncertainty in a model's predictions can signal distributional shift—an indication that the current market regime differs from training data. This provides an early warning system for regime changes.

## The Principle

During familiar market conditions, an ensemble's predictions will agree (low epistemic uncertainty). When conditions shift to a new regime, ensemble members—each having captured slightly different aspects of the training distribution—will disagree, producing elevated epistemic uncertainty.

## Implementation

```python
import torch
import numpy as np
from typing import Dict, List


class UncertaintyRegimeDetector:
    """
    Detect market regime changes via rising epistemic uncertainty.
    """
    
    def __init__(self, lookback: int = 60, threshold_z: float = 2.0):
        self.lookback = lookback
        self.threshold_z = threshold_z
        self.uncertainty_history = []
    
    def update(self, epistemic_uncertainty: float) -> Dict[str, float]:
        """
        Update with new uncertainty observation and check for regime change.
        """
        self.uncertainty_history.append(epistemic_uncertainty)
        
        if len(self.uncertainty_history) < self.lookback + 1:
            return {'regime_signal': 0.0, 'z_score': 0.0}
        
        # Compare current uncertainty to recent history
        recent = np.array(self.uncertainty_history[-self.lookback-1:-1])
        current = epistemic_uncertainty
        
        mu = recent.mean()
        sigma = recent.std() + 1e-8
        z_score = (current - mu) / sigma
        
        # Regime change signal
        regime_signal = 1.0 if z_score > self.threshold_z else 0.0
        
        return {
            'regime_signal': regime_signal,
            'z_score': z_score,
            'uncertainty_mean': mu,
            'uncertainty_current': current,
        }
    
    def get_regime_history(self) -> np.ndarray:
        """Return full uncertainty time series."""
        return np.array(self.uncertainty_history)


def ensemble_regime_analysis(
    ensemble_predictions: torch.Tensor,
    window: int = 20
) -> Dict[str, torch.Tensor]:
    """
    Analyze ensemble disagreement over time for regime detection.
    
    Args:
        ensemble_predictions: (M, T) predictions from M members over T steps
    """
    M, T = ensemble_predictions.shape
    
    # Rolling disagreement
    disagreement = ensemble_predictions.std(dim=0)  # (T,)
    
    # Rolling statistics
    rolling_mean = torch.zeros(T)
    rolling_z = torch.zeros(T)
    
    for t in range(window, T):
        window_data = disagreement[t-window:t]
        rolling_mean[t] = window_data.mean()
        z = (disagreement[t] - window_data.mean()) / (window_data.std() + 1e-8)
        rolling_z[t] = z
    
    return {
        'disagreement': disagreement,
        'rolling_mean': rolling_mean,
        'z_score': rolling_z,
        'regime_change_flags': (rolling_z > 2.0).float(),
    }
```

## Practical Application

1. **Train ensemble** on historical data spanning multiple regimes
2. **Monitor disagreement** in real-time predictions
3. **Flag regime changes** when disagreement exceeds historical norms
4. **Reduce exposure** during flagged periods until the model adapts

## Combining with Traditional Indicators

Uncertainty-based regime detection complements traditional methods (hidden Markov models, structural breaks). The combination of statistical tests and model-based uncertainty provides more robust detection.

## References

- Ovadia, Y., et al. (2019). "Can You Trust Your Model's Uncertainty?" NeurIPS.
- Maddox, W., et al. (2019). "A Simple Baseline for Bayesian Inference in Deep Learning." NeurIPS.
