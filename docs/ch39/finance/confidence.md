# Model Confidence Scoring

## Overview

Model confidence scores enable automated decision-making: trade only when the model is confident, abstain or reduce size otherwise. This section covers confidence scoring methods tailored for financial applications.

## Selective Trading with Uncertainty

```python
import torch
from typing import Dict


class SelectiveTrader:
    """
    Trade only when model confidence exceeds threshold.
    Uncertainty-based position sizing for remaining trades.
    """
    
    def __init__(self, model, confidence_threshold: float = 0.7,
                 max_position: float = 1.0):
        self.model = model
        self.threshold = confidence_threshold
        self.max_position = max_position
    
    def generate_signals(
        self, features: torch.Tensor, n_mc_samples: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Generate trading signals with confidence filtering."""
        self.model.eval()
        
        # MC predictions
        preds = []
        for _ in range(n_mc_samples):
            with torch.no_grad():
                pred = self.model(features)
                preds.append(pred)
        
        preds = torch.stack(preds)  # (T, batch, 1)
        
        mean_pred = preds.mean(dim=0).squeeze()
        pred_std = preds.std(dim=0).squeeze()
        
        # Confidence = inverse of coefficient of variation
        confidence = mean_pred.abs() / (pred_std + 1e-8)
        
        # Position sizing: proportional to confidence
        position = mean_pred.sign() * torch.clamp(
            confidence / confidence.median(), 0, self.max_position
        )
        
        # Zero out low-confidence positions
        position[confidence < self.threshold] = 0
        
        return {
            'position': position,
            'confidence': confidence,
            'prediction': mean_pred,
            'uncertainty': pred_std,
            'n_trades': (position != 0).sum().item(),
            'coverage': (position != 0).float().mean().item(),
        }
```

## Confidence Calibration for Trading

Raw model confidence scores need calibration to be useful. The key metric is: "Among signals with confidence $c$, what is the empirical hit rate?"

If the model says 70% confident in direction, we should observe approximately 70% correct directional predictions in that confidence bucket.

## References

- Geifman, Y., & El-Yaniv, R. (2017). "Selective Classification for Deep Neural Networks." NeurIPS.
