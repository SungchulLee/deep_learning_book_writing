# Label Smoothing

## Overview

Label smoothing replaces hard one-hot targets with soft targets: $y_k^{\text{smooth}} = (1-\epsilon) \cdot \mathbb{1}[k=c] + \epsilon/K$, where $\epsilon$ is typically 0.1 and $K$ is the number of classes.

## Effect on Training

The smoothed loss decomposes as $(1-\epsilon) \cdot \text{CE}(p, y_{\text{hard}}) + \epsilon \cdot \text{KL}(p \| u)$, encouraging the model to maintain probability mass across all classes and avoid overconfident predictions.

## Implementation

```python
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth_loss).mean()
```

## Impact on Calibration

Label smoothing improves model calibration (alignment between predicted probabilities and actual correctness) by compressing the output distribution, yielding better Expected Calibration Error (ECE). The value $\epsilon = 0.1$ is nearly universal in transformer training.
