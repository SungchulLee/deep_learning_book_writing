# Data Augmentation Basics

## Overview

Data augmentation applies random transformations to training samples, effectively expanding the dataset and serving as a powerful regularization technique. By presenting the model with varied views of the same data, augmentation encourages learning invariant representations and reduces overfitting.

## Why Augmentation Works

Augmentation encodes **prior knowledge about invariances** into the training process. A horizontally flipped cat is still a cat; a slightly rotated digit is still the same digit. By making these transformations explicit, we prevent the model from memorizing spurious spatial or intensity patterns.

Formally, augmentation approximates training on the distribution $p_{\text{aug}}(x, y)$ obtained by marginalizing over transformation parameters $\tau$:

$$p_{\text{aug}}(x, y) = \int p(\tau) \, p(x' | \tau) \, \delta(y) \, d\tau$$

This acts as a data-dependent regularizer that penalizes sensitivity to transformations the model should be invariant to.

## Standard Image Augmentations

```python
from torchvision import transforms

train_transform = transforms.Compose([
    # Spatial augmentations
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

    # Color augmentations
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.02),

    # Conversion and normalization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),

    # Tensor-space augmentations
    transforms.RandomErasing(p=0.1)
])
```

## Augmentation Strategies by Domain

**Natural images**: Flips, crops, color jitter, rotation are safe. Vertical flips may not be appropriate for scenes with gravitational semantics (e.g., architectural photos).

**Medical imaging**: Rotation, elastic deformations, intensity shifts are common. Flips depend on anatomical symmetry.

**Financial time series**: Standard spatial augmentations do not apply. Instead, consider noise injection, time warping, window jittering, and magnitude scaling:

```python
class TimeSeriesAugmentation:
    """Augmentations appropriate for financial time series."""
    def __init__(self, noise_std=0.001, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, x):
        # Additive noise
        x = x + torch.randn_like(x) * self.noise_std
        # Magnitude scaling
        scale = torch.empty(1).uniform_(*self.scale_range)
        x = x * scale
        return x
```

## Augmentation Intensity Guidelines

Too little augmentation provides minimal regularization. Too much augmentation distorts the data distribution and hurts convergence. The right intensity depends on dataset size and model capacity:

- **Small datasets** (< 10k samples): Aggressive augmentation is critical to prevent overfitting.
- **Large datasets** (> 100k samples): Moderate augmentation still helps but provides diminishing returns.
- **Pretrained models**: Lighter augmentation during fine-tuning, since the model already learned robust features.

## Common Pitfalls

**Label-dependent transforms**: Some augmentations invalidate labels. Vertical flips change the meaning of "6" vs "9" in digit recognition. Always verify that augmented samples remain correctly labeled.

**Train-only augmentation**: Stochastic augmentations must be applied only during training. Validation and test sets should use deterministic preprocessing.

**Information-destroying augmentations**: Heavy cropping or color distortion can remove the discriminative signal. Start with mild augmentations and increase gradually while monitoring validation performance.

## Key Takeaways

- Augmentation is a regularization technique that encodes invariance priors.
- Separate augmentation transforms from deterministic preprocessing; apply augmentation only during training.
- Augmentation strategies are domain-dependent—what works for natural images may be harmful for time series.
- Augmentation intensity should be tuned as a hyperparameter: too little or too much both degrade performance.
