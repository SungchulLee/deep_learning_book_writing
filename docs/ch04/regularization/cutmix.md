# CutMix

## Overview

CutMix is a data augmentation and regularization technique that creates training samples by cutting a rectangular patch from one image and pasting it onto another, with labels mixed proportionally to the area of the patch. CutMix combines the spatial occlusion benefits of Cutout with the label-mixing benefits of Mixup, encouraging the model to attend to the full extent of objects while learning from blended targets.

## Mathematical Formulation

### Core Operation

Given two training examples $(x_A, y_A)$ and $(x_B, y_B)$ where $x \in \mathbb{R}^{C \times H \times W}$, CutMix constructs a new sample:

$$
\tilde{x} = \mathbf{M} \odot x_A + (\mathbf{1} - \mathbf{M}) \odot x_B
$$

$$
\tilde{y} = \lambda \, y_A + (1 - \lambda) \, y_B
$$

where $\mathbf{M} \in \{0, 1\}^{H \times W}$ is a binary mask indicating the region to cut, $\odot$ is element-wise multiplication (broadcast over channels), and $\lambda$ is determined by the ratio of unmasked area:

$$
\lambda = 1 - \frac{(x_2 - x_1)(y_2 - y_1)}{W \cdot H}
$$

Here $(x_1, y_1, x_2, y_2)$ are the coordinates of the rectangular cut region.

### Bounding Box Sampling

The cut region is sampled uniformly. Given a mixing ratio $\lambda_0 \sim \text{Beta}(\alpha, \alpha)$:

1. Compute the cut ratio: $r = \sqrt{1 - \lambda_0}$
2. Sample the cut size: $r_w = r \cdot W$, $r_h = r \cdot H$
3. Sample the center: $c_x \sim \text{Uniform}(0, W)$, $c_y \sim \text{Uniform}(0, H)$
4. Compute the bounding box (clipped to image boundaries):

$$
x_1 = \max(0, \lfloor c_x - r_w/2 \rfloor), \quad x_2 = \min(W, \lfloor c_x + r_w/2 \rfloor)
$$

$$
y_1 = \max(0, \lfloor c_y - r_h/2 \rfloor), \quad y_2 = \min(H, \lfloor c_y + r_h/2 \rfloor)
$$

5. Recompute $\lambda$ from actual box area (after clipping):

$$
\lambda = 1 - \frac{(x_2 - x_1)(y_2 - y_1)}{W \cdot H}
$$

The recomputation is important because boundary clipping changes the actual area ratio.

### Comparison with Related Methods

| Method | Input Modification | Label Modification |
|--------|-------------------|-------------------|
| Cutout | Mask region with zeros | Unchanged |
| Mixup | Global linear blend | Linear interpolation |
| CutMix | Paste patch from another image | Proportional to area |

CutMix addresses limitations of both predecessors. Cutout wastes pixels by replacing them with uninformative zeros. Mixup creates globally blended images that may not reflect natural visual inputs. CutMix keeps all pixels informative while producing locally realistic images.

## Why CutMix Works

### Full Utilization of Training Pixels

Unlike Cutout, which replaces patches with zeros (or a constant fill value), CutMix replaces patches with content from another training image. This means every pixel in the augmented image carries semantic information, making more efficient use of the training data.

### Localization Encouragement

Because the label is proportional to the area of each source image, the model must learn to recognize objects from partial views. This encourages attention to multiple discriminative regions rather than relying on a single most-salient patch.

### Regularization Through Partial Information

By showing the model images where only a fraction of the original content is visible, CutMix prevents the model from depending on any single spatial region and encourages more holistic feature learning.

## PyTorch Implementation

### Basic CutMix

```python
import torch
import torch.nn as nn
import numpy as np


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.
    
    Args:
        size: (batch_size, channels, height, width)
        lam: Mixing ratio from Beta distribution
        
    Returns:
        Bounding box coordinates (x1, y1, x2, y2)
    """
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Clip to image boundaries
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix_data(x: torch.Tensor, y: torch.Tensor,
                alpha: float = 1.0) -> tuple:
    """
    Apply CutMix to a batch of images.
    
    Args:
        x: Input batch, shape (batch_size, C, H, W)
        y: Labels (class indices), shape (batch_size,)
        alpha: Beta distribution parameter
        
    Returns:
        x_cutmix: Augmented images
        y_a: Original labels
        y_b: Permuted labels
        lam: Adjusted mixing coefficient (after clipping)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Generate bounding box
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)

    # Cut and paste
    x_cutmix = x.clone()
    x_cutmix[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Recompute lambda based on actual box area
    _, _, H, W = x.shape
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)

    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam


def cutmix_criterion(criterion: nn.Module, pred: torch.Tensor,
                     y_a: torch.Tensor, y_b: torch.Tensor,
                     lam: float) -> torch.Tensor:
    """Compute CutMix loss as weighted combination."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### Complete Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader


def train_with_cutmix(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 1.0,
    cutmix_prob: float = 0.5,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    Train model with CutMix augmentation.
    
    Args:
        model: CNN model
        train_loader: Training data
        val_loader: Validation data
        alpha: Beta distribution parameter for CutMix
        cutmix_prob: Probability of applying CutMix per batch
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Apply CutMix with probability cutmix_prob
            if np.random.random() < cutmix_prob:
                X_mixed, y_a, y_b, lam = cutmix_data(X_batch, y_batch, alpha)
                outputs = model(X_mixed)
                loss = cutmix_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation (no CutMix)
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_correct/val_total:.4f}")
    
    return history
```

## CutMix Variants

### Class-Aware CutMix

Select patches from images of different classes to maximize the diversity of the resulting composite:

```python
def class_aware_cutmix(x, y, alpha=1.0):
    """
    CutMix that preferentially pairs samples from different classes.
    
    This ensures the model must distinguish between regions of
    genuinely different categories.
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)
    
    # Create cross-class permutation
    index = torch.randperm(batch_size, device=x.device)
    
    # Try to ensure different classes (best effort)
    for i in range(batch_size):
        if y[i] == y[index[i]]:
            for j in range(batch_size):
                if y[i] != y[index[j]] and y[j] != y[index[i]]:
                    index[i], index[j] = index[j].clone(), index[i].clone()
                    break
    
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    
    x_cutmix = x.clone()
    x_cutmix[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    _, _, H, W = x.shape
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)
    
    return x_cutmix, y, y[index], lam
```

### Multi-Image CutMix

Mix patches from more than two images:

```python
def multi_cutmix(x, y, n_patches=4, alpha=1.0):
    """
    CutMix using patches from multiple images in a grid layout.
    
    Args:
        x: Input batch (B, C, H, W)
        y: Labels (B,)
        n_patches: Number of grid cells (must be a perfect square)
        alpha: Mixing parameter
        
    Returns:
        Augmented batch and soft label vectors
    """
    B, C, H, W = x.shape
    num_classes = y.max().item() + 1
    grid_size = int(np.sqrt(n_patches))
    
    patch_h = H // grid_size
    patch_w = W // grid_size
    
    x_mixed = x.clone()
    soft_labels = torch.zeros(B, num_classes, device=x.device)
    
    total_area = H * W
    
    for gi in range(grid_size):
        for gj in range(grid_size):
            # Random source for this patch
            index = torch.randperm(B, device=x.device)
            
            h_start = gi * patch_h
            h_end = (gi + 1) * patch_h if gi < grid_size - 1 else H
            w_start = gj * patch_w
            w_end = (gj + 1) * patch_w if gj < grid_size - 1 else W
            
            x_mixed[:, :, h_start:h_end, w_start:w_end] = \
                x[index, :, h_start:h_end, w_start:w_end]
            
            patch_area = (h_end - h_start) * (w_end - w_start)
            weight = patch_area / total_area
            
            # Accumulate soft labels
            patch_labels = torch.zeros(B, num_classes, device=x.device)
            patch_labels.scatter_(1, y[index].unsqueeze(1), 1.0)
            soft_labels += weight * patch_labels
    
    return x_mixed, soft_labels
```

### Saliency-Guided CutMix

Use gradient-based saliency to place the cut region where it matters most:

```python
def saliency_cutmix(model, x, y, alpha=1.0):
    """
    CutMix guided by saliency maps to ensure patches overlap
    with informative regions.
    
    Reference: Uddin et al., "SaliencyMix" (2020)
    """
    model.eval()
    x_sal = x.clone().requires_grad_(True)
    
    # Compute saliency
    outputs = model(x_sal)
    loss = nn.CrossEntropyLoss()(outputs, y)
    loss.backward()
    
    saliency = x_sal.grad.abs().mean(dim=1)  # (B, H, W)
    
    model.train()
    
    # Sample lambda
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    index = torch.randperm(B, device=x.device)
    
    # Place box at most salient location of the source image
    x1_list, y1_list, x2_list, y2_list = [], [], [], []
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))
    
    for b in range(B):
        sal = saliency[index[b]]
        # Find the most salient center point
        flat_idx = sal.argmax().item()
        cy, cx = flat_idx // W, flat_idx % W
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        x1_list.append(x1); y1_list.append(y1)
        x2_list.append(x2); y2_list.append(y2)
    
    # Apply per-sample CutMix
    x_cutmix = x.clone()
    lam_actual = torch.zeros(B, device=x.device)
    
    for b in range(B):
        x_cutmix[b, :, y1_list[b]:y2_list[b], x1_list[b]:x2_list[b]] = \
            x[index[b], :, y1_list[b]:y2_list[b], x1_list[b]:x2_list[b]]
        area = (x2_list[b] - x1_list[b]) * (y2_list[b] - y1_list[b])
        lam_actual[b] = 1 - area / (H * W)
    
    return x_cutmix, y, y[index], lam_actual
```

## Combining with Other Augmentations

### CutMix + Standard Augmentation Pipeline

```python
import torchvision.transforms as T

def get_cutmix_training_pipeline(image_size=32):
    """
    Standard augmentation pipeline. CutMix is applied at the
    batch level (in the training loop), not in the transform pipeline.
    """
    train_transform = T.Compose([
        T.RandomCrop(image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    return train_transform, val_transform
```

### CutMix or Mixup (Random Selection)

A common practice is to randomly choose between CutMix and Mixup per batch:

```python
def cutmix_or_mixup(x, y, cutmix_alpha=1.0, mixup_alpha=0.2, 
                     cutmix_prob=0.5):
    """
    Randomly apply CutMix or Mixup per batch.
    
    This is the strategy used in many modern training recipes
    (e.g., DeiT, EfficientNetV2).
    """
    if np.random.random() < cutmix_prob:
        return cutmix_data(x, y, alpha=cutmix_alpha)
    else:
        from mixup import mixup_data  # See mixup.md
        return mixup_data(x, y, alpha=mixup_alpha)
```

## Practical Guidelines

### Hyperparameter Selection

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| $\alpha$ | 1.0 | Standard for CutMix; $\text{Beta}(1,1)$ is Uniform[0,1] |
| `cutmix_prob` | 0.5 – 1.0 | When combined with Mixup, use 0.5 |
| Image size | Any | Works well across resolutions |

### When to Use CutMix

1. **Image classification**: Strong regularizer for CNNs and Vision Transformers
2. **Object detection**: Better than Mixup because spatial structure is preserved
3. **Limited training data**: Significant improvement when data is scarce
4. **Fine-grained recognition**: Encourages attention to multiple discriminative parts

### When to Avoid CutMix

1. **Non-image data**: Spatial cutting is not meaningful for tabular, text, or 1D signals
2. **Pixel-level tasks**: Where exact spatial labels are needed (use with care in segmentation)
3. **Very small images**: If images are already small, the cut region may be too small to be effective

### Evaluation

CutMix is **never applied at evaluation time**. Use clean, unmodified images for validation and testing.

## References

1. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
2. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
3. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of CNNs with Cutout. *arXiv*.
4. Uddin, A. F. M. S., et al. (2020). SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization. *ICLR Workshop*.
