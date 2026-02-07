# Market Manipulation Detection

## Introduction

Market manipulation—the deliberate attempt to interfere with free market operation—represents a uniquely adversarial domain for machine learning. Unlike standard adversarial robustness where perturbations are mathematical constructs, market manipulation involves real economic agents who strategically modify observable signals to deceive both human and algorithmic participants.

## Manipulation as an Adversarial Attack

### Parallels to Adversarial ML

| Adversarial ML Concept | Market Manipulation Analog |
|------------------------|---------------------------|
| Input perturbation | Spoofing orders, wash trading |
| Targeted attack | Triggering specific algorithmic trading signals |
| Evasion attack | Avoiding surveillance system detection |
| Perturbation budget | Regulatory risk, capital constraints |
| Black-box attack | Manipulating without knowing surveillance model |

### Formal Framework

A market manipulator modifies observable market signals $\mathbf{x}_t$ (prices, volumes, order book) to achieve an objective:

$$
\mathbf{x}_t^{\text{manip}} = \mathbf{x}_t + \boldsymbol{\delta}_t, \quad \text{subject to } \boldsymbol{\delta}_t \in \mathcal{C}_t
$$

where $\mathcal{C}_t$ encodes economic and regulatory constraints:
- Orders must be executable (valid price/quantity)
- Manipulation cost must be recoverable from the strategy's profit
- Pattern must not trigger simple rule-based surveillance

## Types of Manipulation

### Spoofing and Layering

Placing orders with intent to cancel before execution, creating false impressions of supply/demand:

- **Adversarial effect**: Fools order-book-based ML models into predicting price movements
- **Detection challenge**: Distinguishes from legitimate order modification
- **Robustness requirement**: Detection models must be robust to adversarial order patterns

### Wash Trading

Self-dealing transactions that inflate volume without genuine economic activity:

- **Adversarial effect**: Inflates volume-based signals used by trading algorithms
- **Detection challenge**: Transactions appear individually legitimate
- **Robustness requirement**: Models must identify coordinated patterns across transactions

### Pump-and-Dump

Coordinated campaigns to inflate asset prices through false promotion:

- **Adversarial effect**: Manipulates sentiment and momentum signals
- **Detection challenge**: Combines legitimate-looking market activity with social media manipulation
- **Robustness requirement**: Multi-modal robustness across price, volume, and text data

## Robust Surveillance Systems

### Architecture for Robust Detection

```python
import torch
import torch.nn as nn
from typing import Dict, List

class RobustSurveillanceSystem:
    """
    Market manipulation detection with adversarial robustness.
    
    Combines multiple detection signals with adversarial
    training to resist adaptive manipulators.
    """
    
    def __init__(
        self,
        order_model: nn.Module,
        pattern_model: nn.Module,
        ensemble_weights: List[float] = [0.5, 0.5]
    ):
        self.order_model = order_model      # Order-level features
        self.pattern_model = pattern_model    # Sequence-level patterns
        self.weights = ensemble_weights
    
    def detect(
        self,
        order_features: torch.Tensor,
        sequence_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-signal manipulation detection.
        
        Ensembles order-level and pattern-level models
        for robustness against single-signal manipulation.
        """
        with torch.no_grad():
            score_order = torch.sigmoid(self.order_model(order_features))
            score_pattern = torch.sigmoid(self.pattern_model(sequence_features))
            
            # Weighted ensemble
            combined = (
                self.weights[0] * score_order +
                self.weights[1] * score_pattern
            )
        
        return {
            'manipulation_score': combined,
            'order_score': score_order,
            'pattern_score': score_pattern,
            'alert': combined > 0.5
        }
```

### Defense Strategies

1. **Multi-signal detection**: Combine price, volume, order book, and behavioral signals so that manipulating one channel is insufficient
2. **Temporal consistency**: Check that patterns are consistent across multiple time scales
3. **Cross-market validation**: Verify signals across related markets and instruments
4. **Adversarial training**: Train detection models against simulated manipulation strategies

## Summary

Market manipulation detection is adversarial robustness applied to a domain with genuine adversaries. The key insight is that manipulation must be economically viable, which constrains the adversary's perturbation space in ways that can be exploited for more effective detection.

## References

1. Cao, Y., et al. (2021). "Adversarial Attacks on Machine Learning-Based Market Surveillance." Journal of Financial Data Science.
2. Golmohammadi, K., & Zaiane, O. R. (2015). "Time Series Contextual Anomaly Detection for Detecting Market Manipulation in Stock Market." IEEE DSAA.
