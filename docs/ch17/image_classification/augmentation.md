# Data Augmentation for Images

## Overview

Data augmentation artificially expands the training set by applying label-preserving transformations to existing images. By exposing the model to variations it will encounter at test time—rotations, scale changes, color shifts, occlusions—augmentation acts as a powerful regularizer that reduces overfitting and improves generalization. This section covers geometric and photometric transforms, advanced augmentation strategies (Cutout, Mixup, CutMix), learned augmentation policies (AutoAugment, RandAugment), and test-time augmentation (TTA).

In quantitative finance, data augmentation principles extend to tabular and time-series data: synthetic oversampling for imbalanced fraud detection, noise injection for robust signal estimation, and regime perturbation for stress testing. The geometric intuition developed here transfers directly.

---

## 1. Why Augmentation Works

### 1.1 The Regularization Perspective

Augmentation implicitly enlarges the training distribution. If $\mathcal{T}$ is a set of label-preserving transforms and $p_{\text{data}}(x, y)$ is the original data distribution, augmentation trains on the expanded distribution:

$$p_{\text{aug}}(x, y) = \mathbb{E}_{t \sim \mathcal{T}}[p_{\text{data}}(t^{-1}(x), y)]$$

This smooths the loss landscape and reduces the gap between training and test performance. Empirically, augmentation provides regularization comparable to or stronger than dropout and weight decay, and it composes with both.

### 1.2 Invariance and Equivariance

Augmentation encodes **prior knowledge** about which transformations should not change the prediction:

- A horizontally flipped cat is still a cat → flip augmentation teaches horizontal invariance
- A slightly rotated digit is the same digit → rotation augmentation teaches rotation invariance
- A darker photo of a car is still a car → brightness augmentation teaches illumination invariance

The choice of augmentations should reflect the invariances present in the target task. Flipping is appropriate for natural images but not for text recognition; rotation is appropriate for aerial imagery but not for face detection.

---

## 2. Geometric Transforms

Geometric transforms alter the spatial arrangement of pixels while preserving semantic content.

### 2.1 Standard Geometric Augmentations

```python
import torch
import torchvision.transforms.v2 as T


geometric_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),           # Use for aerial/medical; skip for natural scenes
    T.RandomRotation(degrees=15),
    T.RandomResizedCrop(
        size=224,
        scale=(0.8, 1.0),                  # Crop 80-100% of area
        ratio=(0.9, 1.1),                  # Near-square aspect ratio
    ),
    T.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),              # Up to 10% translation
        scale=(0.9, 1.1),
        shear=5,
    ),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
])
```

### 2.2 Transform Reference

| Transform | Parameters | Effect | Typical Use |
|-----------|-----------|--------|-------------|
| `RandomHorizontalFlip` | `p=0.5` | Mirror left-right | Nearly universal |
| `RandomVerticalFlip` | `p=0.5` | Mirror top-bottom | Aerial, medical, satellite |
| `RandomRotation` | `degrees=(-15, 15)` | Rotate around center | General; avoid for text |
| `RandomResizedCrop` | `scale=(0.8, 1.0)` | Random crop and resize | Standard for ImageNet training |
| `RandomAffine` | `degrees, translate, scale, shear` | Combined affine transform | Moderate augmentation |
| `RandomPerspective` | `distortion_scale=0.2` | Perspective warp | Scene recognition |

### 2.3 Task-Specific Guidelines

Not all geometric transforms are appropriate for every task:

| Task | Recommended | Avoid |
|------|-------------|-------|
| Natural image classification | Flip, crop, rotation | Large rotation |
| Medical imaging | All flips, rotation, elastic | — |
| Document / OCR | Small rotation, scale | Horizontal flip |
| Satellite / aerial | All flips, rotation, perspective | — |
| Face recognition | Small crop, small rotation | Flip (asymmetry matters) |
| Fine-grained classification | Crop, flip | Heavy distortion |

---

## 3. Photometric Transforms

Photometric transforms alter pixel intensities without changing spatial layout.

### 3.1 Standard Photometric Augmentations

```python
photometric_transforms = T.Compose([
    T.ColorJitter(
        brightness=0.2,       # ±20% brightness
        contrast=0.2,         # ±20% contrast
        saturation=0.2,       # ±20% saturation
        hue=0.1,              # ±10% hue shift
    ),
    T.RandomGrayscale(p=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    T.RandomAutocontrast(p=0.2),
    T.RandomEqualize(p=0.2),
])
```

### 3.2 Normalization Placement

Photometric augmentations must be applied **before** normalization:

```python
train_transform = T.Compose([
    # 1. Geometric transforms (on PIL image or tensor)
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),

    # 2. Photometric transforms
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomGrayscale(p=0.1),

    # 3. Convert to tensor
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),

    # 4. Normalize LAST
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation: no augmentation, only resize + normalize
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 4. Erasing and Occlusion Augmentations

These methods simulate occlusion by masking regions of the input, forcing the model to rely on global context rather than single discriminative patches.

### 4.1 Cutout (Random Erasing)

Replace a random rectangular region with zeros (or random noise):

```python
# PyTorch built-in: applied AFTER ToTensor and Normalize
erasing_transform = T.RandomErasing(
    p=0.5,                  # Probability of applying
    scale=(0.02, 0.33),     # Erased area as fraction of image
    ratio=(0.3, 3.3),       # Aspect ratio range
    value=0,                # Fill with zeros (or 'random')
)
```

**Why it works**: By occluding random patches, the model cannot rely on any single local feature. This is especially effective for fine-grained classification where models might overfit to specific textures or parts.

### 4.2 GridMask

Erase regular grid patterns instead of a single rectangle, preserving more spatial structure while still forcing distributed feature learning:

```python
class GridMask:
    """Erase pixels on a regular grid pattern."""

    def __init__(self, d_range=(96, 224), ratio=0.6, p=0.5):
        self.d_range = d_range    # Grid cell size range
        self.ratio = ratio        # Ratio of cell to keep
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img

        _, h, w = img.shape
        d = torch.randint(self.d_range[0], self.d_range[1], (1,)).item()
        keep = int(d * self.ratio)

        mask = torch.ones(h, w)
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:i + d - keep, j:j + d - keep] = 0

        return img * mask.unsqueeze(0)
```

---

## 5. Mixing Augmentations

Mixing methods create new training samples by combining two or more existing samples, providing stronger regularization by smoothing the decision boundary between classes.

### 5.1 Mixup

Linearly interpolate both inputs and labels between two samples:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$ and typically $\alpha \in [0.1, 0.4]$.

```python
def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """
    Apply Mixup augmentation to a batch.

    Parameters
    ----------
    images : Tensor of shape (B, C, H, W)
    labels : Tensor of shape (B,) integer class labels
    alpha : float
        Beta distribution parameter controlling interpolation strength.

    Returns
    -------
    mixed_images : Tensor of shape (B, C, H, W)
    labels_a, labels_b : original label pairs
    lam : interpolation coefficient
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss as weighted combination."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### 5.2 CutMix

Replace a rectangular patch from one image with the corresponding patch from another, adjusting labels by area proportion:

$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j, \quad \lambda = 1 - \frac{r_w \cdot r_h}{W \cdot H}$$

where $(r_w, r_h)$ is the size of the cut region.

```python
def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """
    Apply CutMix augmentation to a batch.

    Parameters
    ----------
    images : Tensor of shape (B, C, H, W)
    labels : Tensor of shape (B,) integer class labels
    alpha : float
        Beta distribution parameter for area ratio.

    Returns
    -------
    mixed_images : Tensor of shape (B, C, H, W)
    labels_a, labels_b : original label pairs
    lam : effective interpolation coefficient (based on actual cut area)
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0

    batch_size, _, h, w = images.shape
    index = torch.randperm(batch_size, device=images.device)

    # Sample cut region
    cut_ratio = (1 - lam) ** 0.5
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = torch.randint(0, h, (1,)).item()
    cx = torch.randint(0, w, (1,)).item()

    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda to actual cut area
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return mixed_images, labels, labels[index], lam
```

### 5.3 Comparison of Mixing Methods

| Method | Mixing Mechanism | Label Adjustment | Effect |
|--------|-----------------|------------------|--------|
| **Mixup** | Pixel-wise blending | Proportional to $\lambda$ | Smoother decision boundaries, better calibration |
| **CutMix** | Rectangular patch swap | Proportional to cut area | Preserves local structure, localizes features |
| **Mosaic** | 4 images tiled together | Equal or area-weighted | Context diversity, used in YOLO |

---

## 6. Learned Augmentation Policies

Rather than hand-tuning augmentation parameters, these methods search for optimal augmentation strategies automatically.

### 6.1 AutoAugment

Uses reinforcement learning to search over a space of augmentation policies. Each policy is a sequence of sub-policies, each containing two operations with associated probability and magnitude:

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Use ImageNet-optimized policy
auto_augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

# Use CIFAR-10-optimized policy
auto_augment_cifar = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
```

### 6.2 RandAugment

Simplifies AutoAugment to just two hyperparameters: $N$ (number of transforms) and $M$ (magnitude):

```python
from torchvision.transforms import RandAugment

# N=2 transforms, magnitude M=9 (scale 0-30)
rand_augment = RandAugment(num_ops=2, magnitude=9)
```

**Practical advantage**: RandAugment eliminates the expensive search phase of AutoAugment while achieving comparable performance. It is the default augmentation strategy in many modern training recipes.

### 6.3 TrivialAugment

Applies a single random augmentation with random magnitude per image. Even simpler than RandAugment with zero hyperparameters to tune:

```python
from torchvision.transforms import TrivialAugmentWide

trivial_augment = TrivialAugmentWide()
```

### 6.4 Comparison of Learned Policies

| Method | Search Cost | Hyperparameters | Performance |
|--------|------------|-----------------|-------------|
| AutoAugment | Very high (RL search) | Policy-specific | Strong |
| RandAugment | None | $N$, $M$ | Comparable to AutoAugment |
| TrivialAugment | None | None | Competitive |

---

## 7. Complete Training Pipeline

### 7.1 Standard ImageNet Recipe

```python
import torch
import torchvision.transforms.v2 as T


def get_imagenet_transforms(image_size: int = 224):
    """Standard ImageNet augmentation pipeline."""
    train_transform = T.Compose([
        # Geometric
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),

        # Learned augmentation policy
        T.RandAugment(num_ops=2, magnitude=9),

        # To tensor
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),

        # Normalize
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # Erasing (applied after normalization)
        T.RandomErasing(p=0.25, scale=(0.02, 0.33)),
    ])

    val_transform = T.Compose([
        T.Resize(int(image_size / 0.875)),   # 256 for 224
        T.CenterCrop(image_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform
```

### 7.2 Training Loop with Mixing Augmentations

Mixing augmentations (Mixup/CutMix) are applied at the batch level inside the training loop, not in the dataset transform:

```python
import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.5,
):
    """Training loop with Mixup/CutMix augmentation."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Randomly choose Mixup or CutMix
        use_mix = torch.rand(1).item() < mix_prob
        if use_mix:
            if torch.rand(1).item() < 0.5:
                images, labels_a, labels_b, lam = mixup(images, labels, mixup_alpha)
            else:
                images, labels_a, labels_b, lam = cutmix(images, labels, cutmix_alpha)

        logits = model(images)

        if use_mix:
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
```

---

## 8. Test-Time Augmentation (TTA)

TTA applies augmentations at inference time and averages predictions across augmented versions. This provides a free accuracy boost at the cost of increased inference time.

### 8.1 Implementation

```python
@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    normalize: T.Normalize,
    n_augments: int = 5,
) -> torch.Tensor:
    """
    Test-time augmentation: average predictions over augmented views.

    Parameters
    ----------
    model : nn.Module
        Trained classifier in eval mode.
    images : Tensor of shape (B, C, H, W)
        Normalized input images.
    normalize : Transform
        Normalization transform (applied after augmentation).
    n_augments : int
        Number of augmented views per image.

    Returns
    -------
    avg_probs : Tensor of shape (B, num_classes)
        Averaged softmax probabilities.
    """
    model.eval()

    tta_transforms = [
        T.RandomHorizontalFlip(p=1.0),
        T.RandomResizedCrop(images.shape[-1], scale=(0.9, 1.0)),
        T.RandomRotation(degrees=5),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ]

    all_probs = []

    # Original prediction
    probs = torch.softmax(model(images), dim=1)
    all_probs.append(probs)

    # Augmented predictions
    for i in range(min(n_augments, len(tta_transforms))):
        aug_images = tta_transforms[i](images)
        probs = torch.softmax(model(aug_images), dim=1)
        all_probs.append(probs)

    return torch.stack(all_probs).mean(dim=0)
```

### 8.2 When to Use TTA

| Scenario | TTA Recommended? | Notes |
|----------|-----------------|-------|
| Competition submissions | Yes | Free accuracy, no retraining |
| Batch processing pipelines | Yes | Latency less critical |
| Real-time inference | Usually no | Multiplies inference cost by $n$ |
| Uncertain / ambiguous inputs | Yes | Reduces prediction variance |
| Production with SLA | Depends | Trade latency for accuracy |

---

## 9. Augmentation Strength and Scheduling

### 9.1 Choosing Augmentation Strength

Augmentation strength should match dataset size and model capacity:

| Dataset Size | Model Size | Recommended Augmentation |
|-------------|-----------|-------------------------|
| Small (<10K) | Small | Heavy augmentation + Mixup |
| Medium (10K–100K) | Medium | Standard pipeline + RandAugment |
| Large (100K+) | Large | Moderate; heavy augmentation can hurt |

**Under-augmenting** leads to overfitting. **Over-augmenting** introduces too much noise, harming convergence—the model wastes capacity learning to undo unrealistic distortions.

### 9.2 Progressive Augmentation

Increase augmentation strength during training to avoid overwhelming the model early:

```python
class ProgressiveAugment:
    """Linearly increase augmentation magnitude over training."""

    def __init__(self, max_magnitude: int = 15, total_epochs: int = 100):
        self.max_magnitude = max_magnitude
        self.total_epochs = total_epochs

    def get_transform(self, epoch: int) -> T.Compose:
        magnitude = int(self.max_magnitude * min(epoch / self.total_epochs, 1.0))
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=magnitude),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
```

---

## 10. Augmentation for Specific Architectures

Different architectures have different augmentation best practices:

| Architecture | Recommended Augmentation | Notes |
|-------------|-------------------------|-------|
| ResNet | RandomResizedCrop + Flip + ColorJitter | Standard recipe |
| EfficientNet | RandAugment($N$=2, $M$=9) + Cutout | From original paper |
| Vision Transformer (ViT) | RandAugment + Mixup + CutMix + RandomErasing | Heavier augmentation needed due to less inductive bias |
| ConvNeXt | RandAugment + Mixup + CutMix | Follows ViT training recipe |
| MobileNet | AutoAugment + lightweight transforms | Smaller models benefit from augmentation more |

ViTs and other transformer-based architectures generally require **heavier augmentation** than CNNs because they lack the built-in translation equivariance of convolutions. Without strong augmentation, ViTs are prone to overfitting on datasets smaller than ImageNet scale.

---

## 11. Key Takeaways

1. **Augmentation is the single most cost-effective regularizer** for image classification—it requires no additional data collection and composes with other regularization methods.

2. **Match augmentations to task invariances**: horizontal flips for natural images, all orientations for medical/satellite, minimal distortion for text recognition.

3. **Mixing augmentations** (Mixup, CutMix) operate at the batch level and provide complementary regularization to per-image transforms by smoothing decision boundaries.

4. **RandAugment** provides a strong default with only two hyperparameters ($N$, $M$), eliminating the expensive search of AutoAugment.

5. **Augmentation strength should scale** with dataset size (more augmentation for smaller datasets) and model architecture (heavier for ViTs than CNNs).

6. **TTA provides inference-time accuracy gains** by averaging predictions over augmented views, useful when latency constraints permit.

7. **Order matters**: geometric transforms → photometric transforms → tensor conversion → normalization → erasing.

---

## Exercises

### Exercise 1: Augmentation Ablation Study

Train ResNet-18 on CIFAR-10 with five settings: (a) no augmentation, (b) flip + crop only, (c) full geometric + photometric, (d) adding RandAugment, (e) adding Mixup + CutMix. Plot validation accuracy curves and report final accuracy for each.

### Exercise 2: Augmentation Visualization

Write a function that takes a single training image and displays a grid of 16 augmented versions. Use this to visually verify that your augmentation pipeline produces realistic-looking images.

### Exercise 3: CutMix vs Mixup

Implement both CutMix and Mixup for CIFAR-10 classification. Compare (a) final accuracy, (b) calibration (ECE), and (c) robustness to corrupted inputs. Which method is better calibrated?

### Exercise 4: TTA Analysis

Train a model with standard augmentation, then evaluate with TTA using 1, 3, 5, and 10 augmented views. Plot accuracy vs. number of views and measure the wall-clock time overhead.

### Exercise 5: Custom Augmentation for Finance

Design an augmentation strategy for a CNN-based model that classifies candlestick chart images into bullish/bearish patterns. Which geometric and photometric transforms are appropriate? Which would destroy the signal?

---

## References

1. Shorten, C. & Khoshgoftaar, T. M. (2019). A Survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, 6(60).
2. DeVries, T. & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. *arXiv preprint arXiv:1708.04552*.
3. Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). Mixup: Beyond Empirical Risk Minimization. *Proceedings of ICLR*.
4. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *Proceedings of ICCV*.
5. Cubuk, E. D., Zoph, B., Mané, D., Vasudevan, V., & Le, Q. V. (2019). AutoAugment: Learning Augmentation Strategies from Data. *Proceedings of CVPR*.
6. Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. *Proceedings of NeurIPS*.
7. Müller, S. G. & Hutter, F. (2021). TrivialAugment: Tuning-Free Yet State-of-the-Art Data Augmentation. *Proceedings of ICCV*.
8. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training Data-Efficient Image Transformers & Distillation Through Attention. *Proceedings of ICML*.
