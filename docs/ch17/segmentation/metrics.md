# Segmentation Metrics

## Learning Objectives

By the end of this section, you will be able to:

- Compute and interpret Intersection over Union (IoU / Jaccard Index) and mean IoU
- Implement Dice coefficient and understand its relationship to IoU
- Recognize when pixel accuracy is misleading and why IoU is preferred
- Apply clinically relevant metrics (sensitivity, specificity, precision) for medical imaging
- Evaluate panoptic segmentation using Panoptic Quality (PQ)
- Implement a comprehensive metric suite for production segmentation systems

## Intersection over Union (IoU / Jaccard Index)

IoU is the standard metric for segmentation evaluation. It measures the overlap between predicted and ground truth regions:

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

where $TP$, $FP$, and $FN$ denote true positive, false positive, and false negative pixels respectively.

### Properties of IoU

- **Range**: $[0, 1]$ where 1 indicates perfect overlap
- **Symmetric**: $\text{IoU}(A, B) = \text{IoU}(B, A)$
- **Penalizes both over- and under-segmentation**
- **Scale-sensitive**: Small objects with few pixels can have volatile IoU

### Per-Class and Mean IoU

```python
import torch
import numpy as np

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, 
                  num_classes: int, ignore_index: int = 255) -> dict:
    """
    Calculate IoU for each class and mean IoU.
    
    Args:
        pred: Predicted class labels (B, H, W)
        target: Ground truth labels (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore (e.g., boundary pixels)
    
    Returns:
        Dictionary with per-class IoU and mIoU
    """
    ious = {}
    valid_mask = (target != ignore_index)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid_mask
        target_cls = (target == cls) & valid_mask
        
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        
        if union > 0:
            ious[cls] = (intersection / union).item()
        else:
            ious[cls] = float('nan')  # Class not present
    
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0
    
    return ious
```

## Dice Coefficient (F1 Score)

The Dice coefficient is closely related to IoU and particularly popular in medical imaging:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### Relationship to IoU

$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

Dice is always $\geq$ IoU for the same prediction. Both metrics rank predictions identically (they are monotonically related), but Dice values are numerically higher, which can give a misleadingly optimistic impression.

```python
def calculate_dice(pred: torch.Tensor, target: torch.Tensor, 
                   smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient for binary segmentation.
    
    Args:
        pred: Predicted probabilities after sigmoid (B, 1, H, W)
        target: Ground truth binary mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()
```

## Pixel Accuracy

While intuitive, pixel accuracy can be severely misleading with imbalanced classes:

$$\text{Pixel Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Why IoU Over Pixel Accuracy?

Consider a medical image where a lesion covers only 5% of pixels:

```
Prediction A (predicts everything as background):
- Pixel Accuracy: 95%  ← misleadingly high!
- IoU for lesion: 0%   ← correctly reflects failure

Prediction B (correctly segments lesion):
- Pixel Accuracy: 98%
- IoU for lesion: 85%  ← reflects actual quality
```

IoU penalizes missing small objects that pixel accuracy ignores. Always report IoU or Dice alongside pixel accuracy.

```python
def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor,
                             ignore_index: int = 255) -> float:
    """Calculate pixel-wise accuracy."""
    valid_mask = (target != ignore_index)
    correct = ((pred == target) & valid_mask).float().sum()
    total = valid_mask.float().sum()
    return (correct / total).item() if total > 0 else 0.0
```

## Clinical Metrics for Medical Imaging

Medical applications require metrics that reflect clinical significance. Sensitivity (recall) is often paramount—missing a lesion is typically worse than a false alarm.

```python
def calculate_medical_metrics(pred: torch.Tensor, target: torch.Tensor,
                               threshold: float = 0.5) -> dict:
    """
    Comprehensive clinically relevant segmentation metrics.
    
    Returns:
        Dictionary with Dice, Sensitivity, Specificity, Precision
    """
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        
        eps = 1e-6
        
        return {
            'dice': (2 * tp + eps) / (2 * tp + fp + fn + eps),
            'sensitivity': (tp + eps) / (tp + fn + eps),    # Recall / TPR
            'specificity': (tn + eps) / (tn + fp + eps),    # TNR
            'precision': (tp + eps) / (tp + fp + eps),
        }
```

### Threshold Selection for Clinical Applications

In medical imaging, the operating threshold should be tuned to achieve a target sensitivity rather than maximizing overall accuracy:

```python
def find_optimal_threshold(model, val_loader, target_sensitivity=0.95):
    """Find threshold achieving target sensitivity."""
    model.eval()
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for images, masks in val_loader:
            probs = torch.sigmoid(model(images))
            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())
    
    all_probs = torch.cat(all_probs).view(-1)
    all_targets = torch.cat(all_targets).view(-1)
    
    for threshold in torch.linspace(0.01, 0.99, 100):
        pred = (all_probs > threshold).float()
        tp = (pred * all_targets).sum()
        fn = ((1 - pred) * all_targets).sum()
        sensitivity = tp / (tp + fn + 1e-6)
        
        if sensitivity >= target_sensitivity:
            return threshold.item()
    
    return 0.5
```

## Panoptic Quality (PQ)

For panoptic segmentation (combining semantic and instance segmentation), Panoptic Quality evaluates both segmentation and recognition:

$$PQ = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}}_{SQ\text{ (Segmentation Quality)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{RQ\text{ (Recognition Quality)}}$$

PQ decomposes neatly: SQ measures how well matched segments are segmented, while RQ measures how well segments are detected (an F1 score over segment matches).

## Data Validation for Medical Imaging

### Patient-Level Splits

When evaluating medical segmentation, splitting by image rather than by patient causes data leakage—slices from the same patient's scan are highly correlated:

```python
from sklearn.model_selection import GroupKFold

def create_patient_level_splits(patient_ids, n_splits=5):
    """
    Create train/val splits at patient level to prevent data leakage.
    
    Multiple images from the same patient must stay in the same fold.
    """
    group_kfold = GroupKFold(n_splits=n_splits)
    dummy_X = range(len(patient_ids))
    
    splits = []
    for train_idx, val_idx in group_kfold.split(dummy_X, groups=patient_ids):
        splits.append({'train': train_idx, 'val': val_idx})
    
    return splits
```

## Metric Selection Guide

| Scenario | Primary Metric | Secondary Metrics |
|----------|---------------|-------------------|
| General segmentation | mIoU | Pixel accuracy, per-class IoU |
| Binary segmentation | Dice | IoU, sensitivity, specificity |
| Medical imaging | Dice + Sensitivity | Specificity, Hausdorff distance |
| Instance segmentation | Mask AP (COCO-style) | AP@50, AP@75 |
| Panoptic segmentation | PQ | SQ, RQ, per-class PQ |
| Detection-oriented | mAP | AP at various IoU thresholds |

## Summary

Proper metric selection is essential for meaningful evaluation:

1. **IoU/mIoU** is the standard for semantic segmentation—always prefer over pixel accuracy
2. **Dice** is equivalent to IoU in ranking but numerically higher; dominant in medical imaging
3. **Sensitivity** is critical in medical applications where missing pathology has severe consequences
4. **PQ** provides unified evaluation for panoptic segmentation
5. **Patient-level splitting** is mandatory for medical imaging to prevent data leakage

## References

1. Everingham, M., et al. (2010). The Pascal Visual Object Classes (VOC) Challenge. IJCV.
2. Kirillov, A., et al. (2019). Panoptic Segmentation. CVPR.
3. Isensee, F., et al. (2021). nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation. Nature Methods.
4. Müller, D., et al. (2022). Towards a Guideline for Evaluation Metrics in Medical Image Segmentation. BMC Research Notes.
