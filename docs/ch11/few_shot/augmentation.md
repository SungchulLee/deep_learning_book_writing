# Data Augmentation for Few-Shot

## Standard Augmentations

Random cropping, flipping, and color jitter remain useful but aggressive augmentation can distort class-discriminative features when only $K$ examples are available.

## Few-Shot-Specific Strategies

**Feature-space augmentation**: augment in embedding space rather than pixel space, sampling around each class mean:

```python
def feature_augment(features, labels, n_aug=5):
    augmented = [features]
    for cls in labels.unique():
        cls_feat = features[labels == cls]
        mean, std = cls_feat.mean(0), cls_feat.std(0).clamp(min=1e-6)
        for _ in range(n_aug):
            augmented.append((mean + torch.randn_like(mean) * std * 0.5).unsqueeze(0))
    return torch.cat(augmented)
```

**Within-class mixup**: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$ where $y_i = y_j$.

**Test-time augmentation**: apply multiple augmentations to each query and average predictions.

**Hallucination-based**: learn a generator $G(x, z)$ to produce diverse variations of support examples.

## Meta-Training

During episode-based training, augmentation serves dual purposes: increasing task-level diversity (more distinct episodes) and example-level diversity (more varied support/query examples).
