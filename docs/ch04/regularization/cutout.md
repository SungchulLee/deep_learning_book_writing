# Cutout

## Overview

Cutout (also known as Random Erasing) is a data augmentation technique that randomly masks out square or rectangular regions of input images during training. By occluding portions of the image, Cutout forces the model to rely on a wider range of spatial features rather than fixating on a single discriminative patch, improving robustness and generalization.

## Mathematical Formulation

### Core Operation

Given an image $x \in \mathbb{R}^{C \times H \times W}$, Cutout generates a binary mask $\mathbf{M} \in \{0, 1\}^{H \times W}$ that is zero inside a randomly placed rectangular region and one elsewhere:

$$
\tilde{x} = \mathbf{M} \odot x
$$

where $\odot$ denotes element-wise multiplication broadcast across channels. The label $y$ remains unchanged — this distinguishes Cutout from CutMix, which also modifies the label.

### Mask Generation

1. Sample the center of the mask uniformly: $(c_x, c_y) \sim \text{Uniform}([0, W] \times [0, H])$
2. Define the mask size $s$ (fixed side length, or sampled from a range)
3. Compute the bounding box, clipped to image boundaries:

$$
x_1 = \max(0,\; c_x - \lfloor s/2 \rfloor), \quad x_2 = \min(W,\; c_x + \lfloor s/2 \rfloor)
$$

$$
y_1 = \max(0,\; c_y - \lfloor s/2 \rfloor), \quad y_2 = \min(H,\; c_y + \lfloor s/2 \rfloor)
$$

4. Set $\mathbf{M}[y_1:y_2, x_1:x_2] = 0$; all other entries remain 1

Allowing the center to be sampled anywhere — including near or outside image boundaries — means the effective mask size varies, providing natural variation in occlusion strength.

### Fill Value

The original Cutout paper fills the masked region with zeros (mean pixel value after normalization). Random Erasing generalizes this to allow random pixel values, the per-channel mean, or a constant fill:

$$
\tilde{x}_{c, y_1:y_2, x_1:x_2} = \begin{cases}
0 & \text{(zero fill — original Cutout)} \\
\mu_c & \text{(per-channel mean fill)} \\
\text{Uniform}(0, 1) & \text{(random fill — Random Erasing)}
\end{cases}
$$

### Comparison with Related Methods

| Method | Mask Target | Fill Value | Label Changed? | Key Benefit |
|--------|------------|------------|----------------|-------------|
| Cutout | Rectangle to zero/constant | Zeros or mean | No | Forces spatial robustness |
| Random Erasing | Rectangle to random values | Random pixels | No | Prevents memorization of patches |
| CutMix | Rectangle from another image | Content from another sample | Yes (proportional) | No wasted pixels |
| Mixup | Global blend | Weighted sum of two images | Yes (proportional) | Smooth decision boundaries |

## Why Cutout Works

### Preventing Reliance on Local Patches

Neural networks can achieve high training accuracy by relying on a small set of highly discriminative local patches. Cutout randomly removes such patches, forcing the model to develop redundant representations that use the entire spatial extent of the input.

### Regularization Effect

Cutout can be viewed as a form of input noise injection. For a model $f$ trained with expected loss:

$$
\mathcal{L}_{\text{cutout}} = \mathbb{E}_{\mathbf{M}}\left[\ell(f(\mathbf{M} \odot x), y)\right]
$$

This encourages the model to minimize loss under all possible occlusion patterns, penalizing over-reliance on any single spatial region.

### Connection to Dropout

Cutout is sometimes described as "spatial dropout on the input." While standard dropout randomly zeros individual elements, Cutout zeros contiguous rectangular regions, which better matches the spatial structure of images. This is related to but distinct from `Dropout2d` (spatial dropout), which drops entire feature maps rather than spatial patches.

### Improved Object Localization

By training with partial views of objects, the model learns to recognize objects from their parts. This has been shown to improve weakly supervised object localization, as the model cannot rely on a single most-discriminative region.

## PyTorch Implementation

### Custom Cutout Transform

```python
import torch
import numpy as np


class Cutout:
    """
    Randomly mask out one or more square patches from an image tensor.
    
    Reference: DeVries & Taylor, "Improved Regularization of CNNs with Cutout"
    
    Args:
        n_holes: Number of patches to cut out
        length: Side length of each square patch
        fill_value: Value to fill the masked region (default: 0.0)
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16, 
                 fill_value: float = 0.0):
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout to a tensor image.
        
        Args:
            img: Tensor image of shape (C, H, W)
            
        Returns:
            Image with cutout regions applied
        """
        h, w = img.shape[-2:]
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            # Sample center
            cy = np.random.randint(h)
            cx = np.random.randint(w)
            
            # Compute box (clipped)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            
            mask[..., y1:y2, x1:x2] = 0
        
        if self.fill_value == 0.0:
            return img * mask
        else:
            return img * mask + self.fill_value * (1 - mask)
```

### Using PyTorch's Built-in RandomErasing

PyTorch provides `transforms.RandomErasing` which generalizes Cutout with configurable fill and aspect ratios:

```python
import torchvision.transforms as T


# Basic Cutout (zero fill)
cutout_transform = T.RandomErasing(
    p=0.5,           # Probability of applying
    scale=(0.02, 0.33),  # Fraction of image area to erase
    ratio=(0.3, 3.3),    # Aspect ratio range
    value=0,              # Fill value (0 = zero fill)
    inplace=False
)

# Random Erasing (random pixel fill)
random_erasing_transform = T.RandomErasing(
    p=0.5,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3),
    value='random'   # Fill with random pixel values
)

# Complete training pipeline with Cutout
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0),  # Applied after ToTensor
])
```

### Configurable Cutout with Multiple Options

```python
class FlexibleCutout:
    """
    Cutout with configurable shape, fill, and application probability.
    
    Args:
        p: Probability of applying cutout
        n_holes: Number of holes to cut
        min_length: Minimum side length of each hole
        max_length: Maximum side length of each hole
        fill_mode: 'zero', 'mean', 'random', or a float value
    """
    
    def __init__(self, p: float = 0.5, n_holes: int = 1,
                 min_length: int = 8, max_length: int = 24,
                 fill_mode: str = 'zero'):
        self.p = p
        self.n_holes = n_holes
        self.min_length = min_length
        self.max_length = max_length
        self.fill_mode = fill_mode
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return img
        
        C, H, W = img.shape
        result = img.clone()
        
        for _ in range(self.n_holes):
            length = np.random.randint(self.min_length, self.max_length + 1)
            
            cy = np.random.randint(H)
            cx = np.random.randint(W)
            
            y1 = max(0, cy - length // 2)
            y2 = min(H, cy + length // 2)
            x1 = max(0, cx - length // 2)
            x2 = min(W, cx + length // 2)
            
            if self.fill_mode == 'zero':
                result[:, y1:y2, x1:x2] = 0
            elif self.fill_mode == 'mean':
                for c in range(C):
                    result[c, y1:y2, x1:x2] = img[c].mean()
            elif self.fill_mode == 'random':
                result[:, y1:y2, x1:x2] = torch.rand(C, y2-y1, x2-x1)
            elif isinstance(self.fill_mode, (int, float)):
                result[:, y1:y2, x1:x2] = self.fill_mode
        
        return result
```

### Batch-Level Cutout

Apply cutout at the batch level for efficiency:

```python
class BatchCutout:
    """
    Apply cutout to an entire batch at once (GPU-friendly).
    
    Args:
        n_holes: Number of holes per image
        length: Side length of each hole
        p: Application probability per image
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16, p: float = 0.5):
        self.n_holes = n_holes
        self.length = length
        self.p = p
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch: (B, C, H, W) tensor
            
        Returns:
            Batch with cutout applied
        """
        B, C, H, W = batch.shape
        
        # Decide which images get cutout
        apply_mask = torch.rand(B, device=batch.device) < self.p
        
        result = batch.clone()
        
        for _ in range(self.n_holes):
            # Random centers for the whole batch
            cy = torch.randint(0, H, (B,), device=batch.device)
            cx = torch.randint(0, W, (B,), device=batch.device)
            
            y1 = torch.clamp(cy - self.length // 2, 0, H)
            y2 = torch.clamp(cy + self.length // 2, 0, H)
            x1 = torch.clamp(cx - self.length // 2, 0, W)
            x2 = torch.clamp(cx + self.length // 2, 0, W)
            
            # Create per-image masks
            mask = torch.ones(B, 1, H, W, device=batch.device)
            for b in range(B):
                if apply_mask[b]:
                    mask[b, :, y1[b]:y2[b], x1[b]:x2[b]] = 0
            
            result = result * mask
        
        return result
```

## Training Example

```python
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def train_with_cutout(
    model: nn.Module,
    cutout_length: int = 16,
    cutout_n_holes: int = 1,
    epochs: int = 200,
    lr: float = 0.1
) -> dict:
    """
    Train a CNN on CIFAR-10 with Cutout augmentation.
    
    Args:
        model: CNN model
        cutout_length: Side length of cutout patches
        cutout_n_holes: Number of patches per image
        epochs: Training epochs
        lr: Initial learning rate
    """
    # Data pipeline with Cutout
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
        Cutout(n_holes=cutout_n_holes, length=cutout_length),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
    ])
    
    train_set = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=train_transform)
    val_set = datasets.CIFAR10('./data', train=False, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, 
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(correct / total)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={correct/total:.4f}")
    
    return history
```

## Cutout vs. Random Erasing

| Feature | Cutout (DeVries & Taylor) | Random Erasing (Zhong et al.) |
|---------|--------------------------|-------------------------------|
| Shape | Fixed-size square | Variable size and aspect ratio |
| Fill | Zero (constant) | Random pixel values |
| Size control | Side length $s$ | Area ratio $[s_l, s_h]$ |
| Aspect ratio | 1:1 (square) | Configurable range |
| PyTorch built-in | No (custom transform) | `transforms.RandomErasing` |

Random Erasing is generally preferred in modern pipelines because the random fill prevents the model from learning to detect the erased region (which is easy when the fill is a constant like zero after normalization).

## Hyperparameter Selection

### Cutout Size

The mask size is the most important hyperparameter. Guidelines from the original paper:

| Dataset | Image Size | Recommended Cutout Length |
|---------|-----------|--------------------------|
| CIFAR-10 | 32×32 | 16 (50% of image width) |
| CIFAR-100 | 32×32 | 8 (25% of image width) |
| SVHN | 32×32 | 20 (62.5% of image width) |
| ImageNet | 224×224 | Use `RandomErasing` with `scale=(0.02, 0.33)` |

### Number of Holes

- **1 hole**: Standard setting, sufficient for most tasks
- **2–3 holes**: Can provide additional regularization for very large or complex images
- **Too many holes**: Destroys too much information, hurts training

### Application Probability

- $p = 0.5$: Standard for `RandomErasing`
- $p = 1.0$: Used in original Cutout paper (always applied)
- Lower $p$ when combined with other strong augmentations

## Combining with Other Techniques

Cutout is complementary to most other regularization methods:

```python
# Cutout + standard augmentation + weight decay + dropout
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0),
])

model = SomeCNN(dropout_rate=0.3)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

When using CutMix, Cutout is typically not needed because CutMix subsumes the occlusion effect while additionally providing mixed labels and informative fill content. See **[CutMix](cutmix.md)** for details.

## Practical Guidelines

### When to Use Cutout

1. **Image classification**: Strong baseline augmentation for CNNs
2. **Small datasets**: Significant improvement when data is limited
3. **Models relying on local features**: When you suspect the model overfits to specific patches
4. **As part of a standard pipeline**: Low risk, easy to add

### When to Avoid Cutout

1. **Already using CutMix**: CutMix provides a superset of Cutout's benefits
2. **Very small images**: If the image is already small, cutout may remove too much information
3. **Tasks requiring full spatial information**: Where every pixel matters (e.g., dense prediction without careful handling)

### Evaluation

Cutout is **never applied** during validation or testing. Always evaluate on clean, unmodified images.

## References

1. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. *arXiv:1708.04552*.
2. Zhong, Z., et al. (2020). Random Erasing Data Augmentation. *AAAI*.
3. Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
4. Singh, K. K., & Lee, Y. J. (2017). Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization. *ICCV*.
