# Comprehensiveness Evaluation

## Introduction

**Comprehensiveness** measures whether an explanation captures all the important features, not just some. Its complement, **sufficiency**, tests whether the identified features alone reproduce the prediction.

## Metrics

### Comprehensiveness Score

Remove top-$k$ features and measure prediction change:

$$
\text{Comprehensiveness}(E, x) = f(x) - f(x_{\setminus E})
$$

Higher = the explanation captures features that matter.

### Sufficiency Score

Keep only top-$k$ features and measure prediction preservation:

$$
\text{Sufficiency}(E, x) = f(x) - f(x_E)
$$

Lower = the identified features are sufficient.

### Implementation

```python
import torch
import numpy as np

def comprehensiveness_sufficiency(
    model, input_tensor, attribution, target_class,
    k_values=[0.1, 0.2, 0.3, 0.5]
):
    """Compute comprehensiveness and sufficiency at various thresholds."""
    model.eval()
    
    with torch.no_grad():
        base_score = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()
    
    attr_flat = attribution.flatten()
    sorted_idx = np.argsort(np.abs(attr_flat))[::-1]
    n_features = len(attr_flat)
    
    results = {}
    for k in k_values:
        n_top = int(k * n_features)
        top_indices = sorted_idx[:n_top]
        
        # Comprehensiveness: remove top-k
        removed = input_tensor.clone().flatten()
        removed[top_indices] = 0
        with torch.no_grad():
            removed_score = torch.softmax(model(removed.reshape(input_tensor.shape)), dim=1)[0, target_class].item()
        
        # Sufficiency: keep only top-k
        kept = torch.zeros_like(input_tensor).flatten()
        kept[top_indices] = input_tensor.flatten()[top_indices]
        with torch.no_grad():
            kept_score = torch.softmax(model(kept.reshape(input_tensor.shape)), dim=1)[0, target_class].item()
        
        results[k] = {
            'comprehensiveness': base_score - removed_score,
            'sufficiency': base_score - kept_score
        }
    
    return results
```

## Interpretation

A good explanation should be both **comprehensive** (high comprehensiveness score) and **sufficient** (low sufficiency score). These metrics are complementary: comprehensiveness alone can be gamed by always selecting all features.

## Summary

Comprehensiveness and sufficiency quantify whether explanations capture all relevant features and whether those features alone reproduce the prediction.

## References

1. DeYoung, J., et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *ACL*.
2. Carton, S., Rathore, A., & Tan, C. (2020). "Evaluating and Characterizing Human Rationales." *EMNLP*.
