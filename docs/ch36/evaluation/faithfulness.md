# Faithfulness Evaluation

## Introduction

**Faithfulness** measures whether an explanation accurately reflects the model's actual decision process. A faithful explanation should identify features that, when removed or added, demonstrably change the model's prediction. This is the most critical evaluation dimension—an unfaithful explanation is worse than no explanation.

## Insertion and Deletion Curves

### Deletion (Most Important First)

Remove features in order of decreasing attribution. A faithful explanation causes rapid prediction decline. **Lower deletion AUC = more faithful.**

### Insertion (Most Important First)

Start from a blank baseline and add features in order of decreasing attribution. **Higher insertion AUC = more faithful.**

### Implementation

```python
import torch
import numpy as np

def insertion_deletion_curves(
    model, input_tensor, attribution_map, target_class,
    steps=100, baseline='blur'
):
    """Compute insertion and deletion AUC for faithfulness evaluation."""
    model.eval()
    attr_flat = attribution_map.flatten()
    sorted_idx = np.argsort(attr_flat)[::-1]
    n_pixels = len(attr_flat)
    step_size = max(1, n_pixels // steps)
    
    if baseline == 'blur':
        from torchvision.transforms.functional import gaussian_blur
        baseline_tensor = gaussian_blur(input_tensor, kernel_size=51)
    else:
        baseline_tensor = torch.zeros_like(input_tensor)
    
    input_flat = input_tensor.flatten().clone()
    baseline_flat = baseline_tensor.flatten().clone()
    
    deletion_scores, insertion_scores = [], []
    del_current, ins_current = input_flat.clone(), baseline_flat.clone()
    
    for i in range(0, n_pixels, step_size):
        with torch.no_grad():
            del_score = torch.softmax(model(del_current.reshape(input_tensor.shape)), dim=1)[0, target_class].item()
            ins_score = torch.softmax(model(ins_current.reshape(input_tensor.shape)), dim=1)[0, target_class].item()
        deletion_scores.append(del_score)
        insertion_scores.append(ins_score)
        
        end_idx = min(i + step_size, n_pixels)
        del_current[sorted_idx[i:end_idx]] = baseline_flat[sorted_idx[i:end_idx]]
        ins_current[sorted_idx[i:end_idx]] = input_flat[sorted_idx[i:end_idx]]
    
    return np.trapz(insertion_scores) / len(insertion_scores), np.trapz(deletion_scores) / len(deletion_scores)
```

## ROAR (RemOve And Retrain)

ROAR provides a more rigorous faithfulness test by retraining after removing attributed features. A faithful method causes greater accuracy drops when its top features are removed.

## Summary

Faithfulness is the most critical property. Insertion/deletion curves provide quick metrics; ROAR provides gold-standard evaluation at higher computational cost.

## References

1. Petsiuk, V., Das, A., & Saenko, K. (2018). "RISE: Randomized Input Sampling for Explanation of Black-box Models." *BMVC*.
2. Hooker, S., et al. (2019). "A Benchmark for Interpretability Methods in Deep Neural Networks." *NeurIPS*.
