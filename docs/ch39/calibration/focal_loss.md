# Focal Loss for Calibration

## Overview

Focal loss (Lin et al., 2017) was originally designed for class imbalance in object detection but also improves calibration by reducing the loss contribution from well-classified examples. This prevents the model from becoming overconfident on easy examples.

## Mathematical Formulation

Standard cross-entropy: $\mathcal{L}_{CE} = -\log(p_t)$

Focal loss: $\mathcal{L}_{FL} = -(1 - p_t)^\gamma \log(p_t)$

where $p_t$ is the predicted probability of the true class and $\gamma \geq 0$ is the focusing parameter.

**Effect of $\gamma$**:

- $\gamma = 0$: Standard cross-entropy
- $\gamma = 1$: Moderate down-weighting of easy examples
- $\gamma = 2$: Strong down-weighting (recommended default)
- $\gamma > 3$: Very aggressive, may underfit

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for training calibrated classifiers.
    
    Down-weights loss from well-classified examples, preventing
    overconfidence and improving calibration.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional class weights
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        p_t = (probs * targets_one_hot).sum(dim=-1)  # prob of true class
        focal_weight = (1 - p_t) ** self.gamma
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

## Calibration Benefits

Mukhoti et al. (2020) showed that focal loss with $\gamma \approx 3$ produces well-calibrated models without post-hoc calibration, achieving ECE comparable to temperature-scaled cross-entropy models.

The mechanism: focal loss prevents the logit magnitudes from growing unboundedly on easy examples, which is the primary source of overconfidence.

## References

- Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
- Mukhoti, J., et al. (2020). "Calibrating Deep Neural Networks using Focal Loss." NeurIPS.
