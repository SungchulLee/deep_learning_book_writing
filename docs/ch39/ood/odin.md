# ODIN: Out-of-Distribution Detector

## Overview

ODIN (Liang et al., 2018) improves OOD detection by combining temperature scaling with input perturbation. The key insight is that in-distribution and OOD samples respond differently to these transformations.

## Method

1. **Temperature scaling**: Apply high temperature $T$ to soften the softmax
2. **Input perturbation**: Add a small adversarial perturbation in the direction of increasing confidence

$$\tilde{\mathbf{x}} = \mathbf{x} - \epsilon \cdot \text{sign}(\nabla_{\mathbf{x}} \max_c \log p(y=c|\mathbf{x}; T))$$

3. **Score**: Maximum softmax probability of the perturbed input with temperature

## Implementation

```python
import torch
import torch.nn.functional as F


def odin_score(model, x, temperature=1000.0, epsilon=0.001):
    """
    ODIN OOD score: temperature + input perturbation.
    Returns: 1 - max_softmax (higher = more likely OOD)
    """
    model.eval()
    x.requires_grad_(True)
    
    # Forward pass with temperature
    logits = model(x)
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    
    # Compute gradient for perturbation
    loss = max_probs.sum()
    loss.backward()
    
    # Perturb input
    gradient = x.grad.data.sign()
    x_perturbed = x.data - epsilon * gradient
    
    # Score perturbed input
    with torch.no_grad():
        logits_p = model(x_perturbed)
        probs_p = F.softmax(logits_p / temperature, dim=-1)
        score = 1 - probs_p.max(dim=-1).values
    
    x.requires_grad_(False)
    return score
```

## Hyperparameter Selection

- **Temperature $T$**: Typically 1000. Higher values help separate in-distribution from OOD.
- **Perturbation $\epsilon$**: Typically 0.001-0.01. Tuned on a small OOD validation set.

## References

- Liang, S., et al. (2018). "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks." ICLR.
