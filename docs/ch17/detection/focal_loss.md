# Focal Loss

## Learning Objectives

By the end of this section, you will be able to:

- Understand the class imbalance problem in dense object detection
- Derive focal loss from standard cross-entropy with the focusing parameter
- Implement focal loss for both binary and multi-class settings
- Select appropriate hyperparameters (alpha, gamma) for different scenarios

## Motivation: The Easy Example Problem

In dense detection, the vast majority of candidate locations are easy negatives (background). Standard cross-entropy treats all examples equally, so the cumulative loss from thousands of easy negatives overwhelms the signal from the few hard, informative examples.

## Focal Loss Definition

Focal loss adds a modulating factor $(1 - p_t)^\gamma$ to cross-entropy:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's estimated probability for the correct class.

### Effect of the Focusing Parameter $\gamma$

| $p_t$ (confidence) | CE Loss | FL ($\gamma=2$) | Reduction |
|---------------------|---------|-----------------|-----------|
| 0.9 (easy) | 0.105 | 0.001 | **100×** |
| 0.5 (medium) | 0.693 | 0.173 | 4× |
| 0.1 (hard) | 2.303 | 1.867 | 1.2× |

At $\gamma = 2$, well-classified examples are down-weighted by 100× or more, while hard examples are barely affected. This automatically focuses training on informative examples without explicit hard negative mining.

### The Alpha Balancing Factor

$\alpha_t$ provides class-level weighting independent of focal weighting. Typical value: $\alpha = 0.25$ (down-weights the more frequent background class).

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Balancing factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               gamma=0 reduces to standard cross-entropy
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        return (alpha_weight * focal_weight * bce).mean()


class MultiClassFocalLoss(nn.Module):
    """Multi-class focal loss for segmentation and detection."""
    
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none',
                             ignore_index=self.ignore_index)
        
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1).clamp(0, logits.size(1) - 1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        valid_mask = (targets != self.ignore_index).float()
        focal_loss = focal_weight * ce * valid_mask
        
        return focal_loss.sum() / (valid_mask.sum() + 1e-6)
```

## Hyperparameter Guidelines

| Setting | $\gamma$ | $\alpha$ | Notes |
|---------|----------|----------|-------|
| Mild imbalance | 1.0 | 0.5 | Slight focusing |
| Standard detection | 2.0 | 0.25 | RetinaNet default |
| Extreme imbalance | 3.0–5.0 | 0.1 | Medical, small objects |
| Balanced dataset | 0.0 | 0.5 | Reduces to CE |

## Summary

Focal loss addresses the fundamental class imbalance in dense detection by down-weighting easy examples. With $\gamma = 2$ and $\alpha = 0.25$, it enabled RetinaNet to match two-stage detector accuracy without any sampling heuristics—demonstrating that the accuracy gap was caused by class imbalance, not architectural limitations.

## References

1. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. ICCV.
