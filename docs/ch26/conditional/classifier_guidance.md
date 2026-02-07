# Classifier Guidance

## Introduction

**Classifier guidance** uses an external classifier to steer diffusion sampling toward desired classes, trading diversity for fidelity.

## Method

### Modified Score

Combine unconditional score with classifier gradient:

$$\nabla_x \log p(x_t | y) = \nabla_x \log p(x_t) + \nabla_x \log p(y | x_t)$$

### Guidance Scale

Amplify classifier influence:

$$\tilde{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t) - s \cdot \sigma_t \nabla_{x_t} \log p_\phi(y | x_t)$$

where $s$ is the guidance scale (typically 1-10).

## Implementation

```python
"""
Classifier Guidance
===================
"""

import torch
import torch.nn as nn


class ClassifierGuidedSampler:
    """Diffusion sampling with classifier guidance."""
    
    def __init__(self, diffusion_model, classifier, device):
        self.model = diffusion_model
        self.classifier = classifier
        self.device = device
    
    @torch.no_grad()
    def guided_step(self, x_t, t, y, guidance_scale=3.0):
        """Single guided sampling step."""
        # Get classifier gradient
        x_t = x_t.requires_grad_(True)
        with torch.enable_grad():
            logits = self.classifier(x_t, t)
            log_prob = torch.log_softmax(logits, dim=-1)
            selected = log_prob[range(len(y)), y]
            grad = torch.autograd.grad(selected.sum(), x_t)[0]
        x_t = x_t.detach()
        
        # Unconditional noise prediction
        eps_pred = self.model(x_t, t)
        
        # Guide toward class
        eps_guided = eps_pred - guidance_scale * grad
        
        # Standard DDPM update with guided epsilon
        # ... (rest of sampling step)
        
        return x_t_prev
```

## Trade-offs

| Guidance Scale | Diversity | Fidelity |
|----------------|-----------|----------|
| 0 | High | Low |
| 1 | Medium | Medium |
| 5+ | Low | High |

## Limitations

1. Requires training a separate classifier
2. Classifier must handle noisy inputs
3. Gradient computation adds overhead

## Summary

Classifier guidance enables controlled generation but requires an additional noisy classifier.

