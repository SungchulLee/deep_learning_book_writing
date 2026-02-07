# Softmax Baseline for OOD Detection

## Overview

The Maximum Softmax Probability (MSP) baseline (Hendrycks & Gimpel, 2017) is the simplest OOD detector: use $1 - \max_c p(y=c|\mathbf{x})$ as the OOD score. Despite its simplicity, it provides a surprisingly strong baseline.

## Method

$$s_{\text{MSP}}(\mathbf{x}) = 1 - \max_c \text{softmax}(f_\theta(\mathbf{x}))_c$$

In-distribution samples tend to have higher maximum softmax probability (lower score); OOD samples tend to have lower maximum probability (higher score).

## Implementation

```python
import torch
import torch.nn.functional as F


def msp_ood_score(model, x):
    """Maximum Softmax Probability OOD score."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values
    return 1 - max_prob  # Higher = more likely OOD
```

## Limitations

MSP is limited by neural network overconfidence: models often assign high softmax probabilities to OOD inputs, reducing detection performance. Energy-based and distance-based methods address this limitation.

## References

- Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks." ICLR.
