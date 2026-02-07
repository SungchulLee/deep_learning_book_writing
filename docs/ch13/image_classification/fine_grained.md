# Fine-Grained Classification

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by Fine-Grained Classification (—)
- Identify how Fine-Grained Classification influenced subsequent architecture design

## Overview

**Year**: — | **Parameters**: — | **Key Innovation**: Distinguishing visually similar sub-categories

Fine-grained classification distinguishes between visually similar sub-categories (e.g., bird species, car models, plant diseases). This requires learning subtle discriminative features that general classifiers overlook.

## Challenges

1. **Small inter-class variance**: Different species may differ only in bill shape or wing pattern
2. **Large intra-class variance**: Same species varies by age, pose, lighting
3. **Limited training data**: Specialized domains have fewer labeled examples
4. **Expert knowledge required**: Annotations need domain expertise

## Key Techniques

### Bilinear Pooling

Captures second-order feature interactions for richer representations:

$$\mathbf{z} = \text{vec}(\mathbf{x}^T \mathbf{y})$$

where $\mathbf{x}$ and $\mathbf{y}$ are features from two branches.

### Part-Based Models

Automatically localize discriminative parts (e.g., bird head, wing) and classify based on part features.

### Transfer Learning with Fine-Tuning

Pre-trained models + careful fine-tuning with small learning rates for early layers and larger rates for the classifier.

```python
import torchvision.models as models

# Fine-grained classification with pre-trained backbone
model = models.resnet50(weights='DEFAULT')

# Freeze early layers, fine-tune later ones
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Replace classifier for fine-grained task
model.fc = torch.nn.Linear(model.fc.in_features, num_species)
```

## Common Datasets

| Dataset | Classes | Images | Domain |
|---------|---------|--------|--------|
| CUB-200-2011 | 200 | 11,788 | Bird species |
| Stanford Cars | 196 | 16,185 | Car models |
| FGVC-Aircraft | 100 | 10,000 | Aircraft variants |
| iNaturalist | 10,000+ | 500,000+ | Species |

## References

1. Lin, T.-Y., RoyChowdhury, A., & Maji, S. (2015). Bilinear CNN Models for Fine-grained Visual Recognition. ICCV.
2. Wei, X.-S., et al. (2021). Fine-Grained Image Analysis with Deep Learning: A Survey. IEEE TPAMI.
