# Loss Functions for Semantic Segmentation

## Learning Objectives

By the end of this section, you will be able to:

- Understand pixel-wise cross-entropy loss and its limitations
- Implement Dice loss and Tversky loss for handling class imbalance
- Apply focal loss for hard example mining in segmentation
- Design boundary-aware loss functions for precise edge delineation
- Combine multiple loss functions for optimal segmentation performance
- Select appropriate loss functions based on dataset characteristics

## Introduction

Selecting the right loss function is critical for segmentation performance. Unlike classification where a single prediction is made, segmentation requires optimizing millions of pixel predictions simultaneously. This creates unique challenges including severe class imbalance, boundary precision requirements, and multi-scale object handling.

## Cross-Entropy Loss: The Foundation

### Standard Pixel-wise Cross-Entropy

The most common loss function treats each pixel as an independent classification problem:

$$\mathcal{L}_{CE} = -\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\sum_{k=1}^{K} y_{ijk} \log(\hat{y}_{ijk})$$

where:
- $y_{ijk}$: One-hot encoded ground truth for pixel $(i,j)$ and class $k$
- $\hat{y}_{ijk}$: Predicted probability (after softmax)
- $K$: Number of classes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def pixel_wise_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                              ignore_index: int = 255) -> torch.Tensor:
    """
    Standard cross-entropy loss for semantic segmentation.
    
    Args:
        logits: Raw predictions (B, K, H, W) where K is number of classes
        targets: Ground truth class indices (B, H, W)
        ignore_index: Label index to ignore (e.g., unlabeled pixels)
    
    Returns:
        Scalar loss value
    """
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


# Binary case: Binary Cross-Entropy with Logits
def binary_segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy for binary (foreground/background) segmentation.
    
    Args:
        logits: Raw predictions (B, 1, H, W)
        targets: Binary ground truth (B, 1, H, W), values in {0, 1}
    """
    return F.binary_cross_entropy_with_logits(logits, targets.float())
```

### Class-Weighted Cross-Entropy

For imbalanced datasets, weight the loss by inverse class frequency:

$$\mathcal{L}_{WCE} = -\frac{1}{HW}\sum_{i,j,k} w_k \cdot y_{ijk} \log(\hat{y}_{ijk})$$

```python
def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                            class_weights: torch.Tensor = None,
                            ignore_index: int = 255) -> torch.Tensor:
    """
    Weighted cross-entropy to handle class imbalance.
    
    Args:
        logits: Predictions (B, K, H, W)
        targets: Ground truth (B, H, W)
        class_weights: Weight for each class (K,), typically inverse frequency
        ignore_index: Label to ignore
    """
    return F.cross_entropy(logits, targets, weight=class_weights, 
                           ignore_index=ignore_index)


def compute_class_weights(dataset, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency.
    
    Args:
        dataset: Segmentation dataset with (image, mask) pairs
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    class_counts = torch.zeros(num_classes)
    
    for _, mask in dataset:
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights
```

### Limitations of Cross-Entropy

1. **Class imbalance blindness**: Background dominates, small objects get ignored
2. **No direct IoU optimization**: Optimizes pixel accuracy, not IoU
3. **Boundary ignorance**: Treats all pixels equally regardless of position
4. **Hard example neglect**: Easy pixels can dominate the gradient

## Dice Loss: Region-Based Optimization

### Dice Coefficient Background

The Dice coefficient (F1 score) measures overlap between prediction and ground truth:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Dice Loss directly optimizes this metric:

$$\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

where $p_i$ is the predicted probability and $g_i$ is the ground truth for pixel $i$.

```python
class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Directly optimizes the Dice coefficient, which is closely related to IoU.
    Effective for imbalanced datasets as it focuses on the foreground region.
    
    Args:
        smooth: Smoothing factor to prevent division by zero
        reduction: 'mean' or 'sum' over batch
    """
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions (B, 1, H, W) for binary segmentation
            targets: Ground truth (B, 1, H, W)
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # Compute Dice per sample
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Dice Loss using one-vs-all approach.
    
    Computes Dice loss for each class separately and averages.
    """
    def __init__(self, num_classes: int, smooth: float = 1e-6, 
                 ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predictions (B, K, H, W)
            targets: Ground truth class indices (B, H, W)
        """
        probs = F.softmax(logits, dim=1)
        
        # Create one-hot encoding
        targets_one_hot = F.one_hot(
            targets.clamp(0, self.num_classes - 1),  # Clamp to valid range
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()  # (B, K, H, W)
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        
        # Apply mask
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Compute Dice for each class
        dice_per_class = []
        for cls in range(self.num_classes):
            pred_cls = probs[:, cls]
            target_cls = targets_one_hot[:, cls]
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            if union > 0:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_per_class.append(dice)
        
        if len(dice_per_class) > 0:
            mean_dice = torch.stack(dice_per_class).mean()
            return 1.0 - mean_dice
        return torch.tensor(0.0, device=logits.device)
```

### Dice Loss Advantages

1. **Handles imbalance**: Naturally weighs by region size
2. **Direct metric optimization**: Optimizes overlap, not pixel accuracy
3. **Scale invariant**: Works well for both large and small objects

### Dice Loss Limitations

1. **Gradient instability**: Near-zero predictions can cause large gradients
2. **Per-image optimization**: Doesn't aggregate well across batch
3. **Boundary insensitivity**: Still treats all pixels equally

## Tversky Loss: Controlling FP/FN Trade-off

Tversky loss generalizes Dice by introducing separate weights for false positives and false negatives:

$$\mathcal{L}_{Tversky} = 1 - \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}$$

When $\alpha = \beta = 0.5$, this equals Dice loss.

```python
class TverskyLoss(nn.Module):
    """
    Tversky Loss for controlling precision/recall trade-off.
    
    Allows weighting false positives and false negatives differently.
    Useful when missing objects (FN) is worse than false alarms (FP),
    or vice versa.
    
    Args:
        alpha: Weight for false positives (higher = penalize FP more)
        beta: Weight for false negatives (higher = penalize FN more)
        smooth: Smoothing factor
    
    Note:
        alpha = beta = 0.5 is equivalent to Dice loss
        alpha = 0.3, beta = 0.7 emphasizes recall (fewer missed detections)
        alpha = 0.7, beta = 0.3 emphasizes precision (fewer false alarms)
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, 
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        """
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()
        
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: Combines Tversky with focal mechanism.
    
    Adds a focusing parameter gamma that down-weights easy examples
    and focuses on hard examples (poor predictions).
    
    Loss = (1 - Tversky)^gamma
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 gamma: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()
        
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        # Focal modifier
        focal_tversky = (1.0 - tversky) ** self.gamma
        
        return focal_tversky
```

## Focal Loss: Hard Example Mining

Focal loss down-weights easy examples to focus training on hard negatives:

$$\mathcal{L}_{Focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the probability of the correct class and $\gamma$ is the focusing parameter.

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    
    Down-weights easy examples (high confidence predictions) and focuses
    learning on hard examples (low confidence predictions).
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
               gamma=0 is equivalent to cross-entropy
               gamma=2 is a common choice
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Binary focal loss.
        
        Args:
            logits: Raw predictions (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss.mean()


class MultiClassFocalLoss(nn.Module):
    """Multi-class focal loss for semantic segmentation."""
    
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predictions (B, K, H, W)
            targets: Ground truth (B, H, W)
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                   ignore_index=self.ignore_index)
        
        # Get predicted probabilities for correct class
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1).clamp(0, logits.size(1) - 1))
        pt = pt.squeeze(1)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1).clamp(0, len(self.alpha) - 1))
            alpha_t = alpha_t.view(targets.shape)
            focal_weight = alpha_t * focal_weight
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index).float()
        
        focal_loss = focal_weight * ce_loss * valid_mask
        
        return focal_loss.sum() / (valid_mask.sum() + 1e-6)
```

## Boundary-Aware Loss Functions

Boundaries are critical for accurate segmentation but are often poorly predicted. Boundary-aware losses explicitly penalize boundary errors.

```python
import scipy.ndimage as ndimage
import numpy as np

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes pixels near object boundaries.
    
    Computes boundary maps from ground truth and weights the loss
    higher for pixels near boundaries.
    
    Args:
        theta: Boundary thickness parameter (higher = thicker boundary region)
        weight_factor: Multiplicative weight for boundary pixels
    """
    def __init__(self, theta: int = 3, weight_factor: float = 5.0):
        super().__init__()
        self.theta = theta
        self.weight_factor = weight_factor
    
    def _compute_boundary_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel weights based on distance to boundary."""
        batch_size = targets.size(0)
        weights = torch.ones_like(targets, dtype=torch.float32)
        
        targets_np = targets.cpu().numpy()
        
        for b in range(batch_size):
            mask = targets_np[b].squeeze()
            
            # Find boundary via erosion and dilation
            eroded = ndimage.binary_erosion(mask > 0.5, iterations=1)
            dilated = ndimage.binary_dilation(mask > 0.5, iterations=1)
            boundary = dilated.astype(float) - eroded.astype(float)
            
            # Dilate boundary for thicker region
            boundary_region = ndimage.binary_dilation(boundary > 0, iterations=self.theta)
            
            # Set weights
            weights[b, 0][torch.from_numpy(boundary_region)] = self.weight_factor
        
        return weights.to(targets.device)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        """
        boundary_weights = self._compute_boundary_weights(targets)
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce = bce * boundary_weights
        
        return weighted_bce.mean()


class DistanceMapLoss(nn.Module):
    """
    Distance transform-based boundary loss.
    
    Uses signed distance transform to weight pixels based on their
    distance to the nearest boundary. Pixels closer to boundaries
    receive higher weights.
    """
    def __init__(self, sigma: float = 5.0):
        super().__init__()
        self.sigma = sigma
    
    def _compute_distance_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute weights based on distance transform."""
        batch_size = targets.size(0)
        weights = torch.zeros_like(targets, dtype=torch.float32)
        
        targets_np = targets.cpu().numpy()
        
        for b in range(batch_size):
            mask = targets_np[b].squeeze() > 0.5
            
            # Distance transform from foreground
            dist_fg = ndimage.distance_transform_edt(mask)
            # Distance transform from background
            dist_bg = ndimage.distance_transform_edt(~mask)
            
            # Signed distance (positive inside, negative outside)
            signed_dist = dist_fg - dist_bg
            
            # Weight: higher near boundary (where dist is small)
            weight = np.exp(-np.abs(signed_dist) / self.sigma)
            
            weights[b, 0] = torch.from_numpy(weight)
        
        return weights.to(targets.device)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dist_weights = self._compute_distance_weights(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (bce * dist_weights).mean()
```

## Combined Loss Functions

Combining multiple losses often yields the best results by leveraging complementary strengths.

```python
class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for semantic segmentation.
    
    Combines multiple loss functions with configurable weights.
    Common combinations:
    - BCE + Dice: Pixel accuracy + region overlap
    - Focal + Dice: Hard examples + region overlap
    - CE + Dice + Boundary: All objectives
    
    Args:
        ce_weight: Weight for cross-entropy loss
        dice_weight: Weight for Dice loss
        focal_weight: Weight for focal loss
        boundary_weight: Weight for boundary loss
    """
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 focal_weight: float = 0.0, boundary_weight: float = 0.0):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice = DiceLoss()
        self.focal = FocalLoss() if focal_weight > 0 else None
        self.boundary = BoundaryLoss() if boundary_weight > 0 else None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        """
        total_loss = 0.0
        
        # Cross-entropy component
        if self.ce_weight > 0:
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets)
            total_loss += self.ce_weight * ce_loss
        
        # Dice component
        if self.dice_weight > 0:
            dice_loss = self.dice(logits, targets)
            total_loss += self.dice_weight * dice_loss
        
        # Focal component
        if self.focal_weight > 0 and self.focal is not None:
            focal_loss = self.focal(logits, targets)
            total_loss += self.focal_weight * focal_loss
        
        # Boundary component
        if self.boundary_weight > 0 and self.boundary is not None:
            boundary_loss = self.boundary(logits, targets)
            total_loss += self.boundary_weight * boundary_loss
        
        return total_loss


# Factory function for common configurations
def create_segmentation_loss(task_type: str = 'balanced') -> nn.Module:
    """
    Create appropriate loss function based on task characteristics.
    
    Args:
        task_type: 
            'balanced' - balanced classes, use CE + Dice
            'imbalanced' - severe class imbalance, use Focal + Dice
            'boundary' - boundary precision important, add boundary loss
            'medical' - medical imaging, use Dice + Tversky
    """
    if task_type == 'balanced':
        return CombinedSegmentationLoss(ce_weight=0.5, dice_weight=0.5)
    
    elif task_type == 'imbalanced':
        return CombinedSegmentationLoss(ce_weight=0.0, dice_weight=0.5, 
                                         focal_weight=0.5)
    
    elif task_type == 'boundary':
        return CombinedSegmentationLoss(ce_weight=0.3, dice_weight=0.4,
                                         boundary_weight=0.3)
    
    elif task_type == 'medical':
        # For medical imaging: combine Dice and Tversky
        # Tversky with beta > alpha emphasizes recall (don't miss lesions)
        class MedicalLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice = DiceLoss()
                self.tversky = TverskyLoss(alpha=0.3, beta=0.7)  # Emphasize recall
            
            def forward(self, logits, targets):
                return 0.5 * self.dice(logits, targets) + 0.5 * self.tversky(logits, targets)
        
        return MedicalLoss()
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
```

## Loss Function Selection Guide

| Scenario | Recommended Loss | Rationale |
|----------|------------------|-----------|
| Balanced classes | CE + Dice | Standard combination |
| Severe imbalance | Focal + Dice | Focus on minority class |
| Small objects | Dice alone | Scale-invariant |
| Boundary precision | CE + Boundary | Explicit boundary penalty |
| Medical imaging | Dice + Tversky | High recall, overlap |
| Real-time/simple | CE only | Fast computation |

## Summary

Loss function selection significantly impacts segmentation performance:

1. **Cross-entropy** is the baseline but struggles with imbalance
2. **Dice loss** directly optimizes overlap metrics
3. **Focal loss** addresses hard example mining
4. **Boundary losses** improve edge precision
5. **Combined losses** leverage complementary strengths

Experimentation is key—the optimal loss depends on your specific dataset characteristics, class distribution, and performance requirements.

## References

1. Milletari, F., et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.
2. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
3. Salehi, S. S. M., et al. (2017). Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. MLMI.
4. Kervadec, H., et al. (2019). Boundary Loss for Highly Unbalanced Segmentation. MIDL.
