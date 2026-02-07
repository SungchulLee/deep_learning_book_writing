# Class-Incremental Learning

Class-incremental learning (Class-IL) is the most challenging continual learning scenario. The model must classify among all classes seen so far without knowing which task a test example belongs to.

## Problem Setting

After training on tasks $\mathcal{T}_1, ..., \mathcal{T}_t$, the model must predict:

$$\hat{y} = \arg\max_{c \in \bigcup_{i=1}^{t} \mathcal{C}_i} f(x; \theta)$$

over the union of all classes, using a single unified output head.

## Why Class-IL Is Hard

1. **Inter-task discrimination**: The model must distinguish classes from different tasks
2. **Output bias**: Recently learned classes receive higher logits (recency bias)
3. **No task ID**: Cannot route to task-specific heads

## Unified Head Architecture

```python
import torch
import torch.nn as nn


class ClassIncrementalModel(nn.Module):
    """Single expanding head for class-incremental learning."""
    
    def __init__(self, backbone, feature_dim=512):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = None
        self.num_classes = 0
    
    def expand_classifier(self, new_classes):
        """Expand the classifier to accommodate new classes."""
        old_num = self.num_classes
        self.num_classes += new_classes
        
        new_classifier = nn.Linear(self.feature_dim, self.num_classes)
        
        if self.classifier is not None:
            # Copy old weights
            with torch.no_grad():
                new_classifier.weight[:old_num] = self.classifier.weight
                new_classifier.bias[:old_num] = self.classifier.bias
        
        self.classifier = new_classifier
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(features)


def bias_correction(model, val_loader, device='cuda'):
    """
    Correct output bias toward recent classes.
    
    Recent classes tend to have larger logits because the model
    was most recently trained on them.
    """
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(y)
    
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    
    # Compute per-class mean logit
    num_classes = logits.shape[1]
    class_means = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            class_means[c] = logits[mask, c].mean()
    
    # Normalise to correct bias
    correction = class_means.mean() / (class_means + 1e-8)
    
    return correction
```

## Key Techniques for Class-IL

| Technique | Approach | Examples |
|-----------|----------|---------|
| Replay | Store/generate old examples | Experience Replay, DGR |
| Distillation | Preserve old predictions | LwF, iCaRL |
| Bias correction | Fix output head bias | BiC, WA |
| Feature replay | Replay in feature space | REMIND |

## References

1. Rebuffi, S.A., et al. (2017). "iCaRL: Incremental Classifier and Representation Learning." *CVPR*.
2. Wu, Y., et al. (2019). "Large Scale Incremental Learning." *CVPR*.
