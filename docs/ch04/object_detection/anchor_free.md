# Anchor-Free Object Detection

## Learning Objectives

By the end of this section, you will be able to:

- Understand the limitations of anchor-based methods
- Explain keypoint-based detection (CornerNet, CenterNet)
- Implement center-based detection with FCOS
- Compare anchor-free and anchor-based approaches
- Apply modern anchor-free detectors for practical applications

## Motivation: Limitations of Anchors

Traditional object detectors rely on **anchor boxes**—predefined bounding box templates at various scales and aspect ratios. While effective, anchors introduce several challenges:

### Problems with Anchor-Based Detection

1. **Hyperparameter Sensitivity**: Performance depends heavily on anchor configurations (sizes, ratios, number)

2. **Scale Imbalance**: Most anchors are negative, requiring careful sampling or loss weighting

3. **IoU Threshold Ambiguity**: Setting thresholds for positive/negative anchors is arbitrary

4. **Computational Overhead**: Generating and processing thousands of anchors per image

5. **Domain Transfer**: Anchor designs tuned for one dataset may not generalize

```
Anchor-Based Approach:
┌─────────────────────────────────────────────────────────────────┐
│  For each feature location:                                     │
│    → Place K anchor boxes (different scales/ratios)            │
│    → Classify each anchor (object vs background)               │
│    → Regress offset from anchor to object                      │
│                                                                 │
│  Problems:                                                      │
│    • Need to design anchor configurations per dataset          │
│    • Most anchors are negative (imbalanced)                    │
│    • IoU threshold is arbitrary                                │
└─────────────────────────────────────────────────────────────────┘

Anchor-Free Approach:
┌─────────────────────────────────────────────────────────────────┐
│  For each feature location:                                     │
│    → Directly predict if this location is an object center     │
│    → Directly regress box dimensions                           │
│                                                                 │
│  Advantages:                                                    │
│    • No anchor hyperparameters                                 │
│    • Simpler training                                          │
│    • Better generalization                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Keypoint-Based Detection

### CornerNet (2018)

CornerNet detects objects as pairs of corners (top-left, bottom-right):

```
┌─────────────────────────────────────────────────────────────────┐
│                      CornerNet Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────────────────────────────┐                    │
│  │     Hourglass Network (Backbone)         │                    │
│  │     Multi-scale feature extraction       │                    │
│  └────────────────────┬────────────────────┘                    │
│                       │                                          │
│         ┌─────────────┴─────────────┐                           │
│         │                           │                           │
│         ▼                           ▼                           │
│  ┌─────────────────┐       ┌─────────────────┐                  │
│  │   Top-Left      │       │  Bottom-Right   │                  │
│  │   Corner Pool   │       │  Corner Pool    │                  │
│  └────────┬────────┘       └────────┬────────┘                  │
│           │                         │                           │
│           ▼                         ▼                           │
│  • Heatmap (K classes)      • Heatmap (K classes)               │
│  • Embeddings               • Embeddings                        │
│  • Offsets                  • Offsets                           │
│                                                                  │
│  Matching: Group corners with similar embeddings                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Corner Pooling**: Aggregates features along edges to locate corners
2. **Embedding Vectors**: Learn to associate corners of the same object
3. **Offset Regression**: Precise corner localization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CornerPool(nn.Module):
    """
    Corner pooling operation for CornerNet.
    
    For top-left corner: max pool from right and from bottom
    For bottom-right corner: max pool from left and from top
    """
    def __init__(self, corner_type: str = 'top_left'):
        super().__init__()
        self.corner_type = corner_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W) feature map
            
        Returns:
            Corner pooled features
        """
        if self.corner_type == 'top_left':
            # Pool from right (horizontal) and from bottom (vertical)
            horizontal = self._pool_horizontal_right(x)
            vertical = self._pool_vertical_bottom(x)
        else:  # bottom_right
            # Pool from left (horizontal) and from top (vertical)
            horizontal = self._pool_horizontal_left(x)
            vertical = self._pool_vertical_top(x)
        
        return horizontal + vertical
    
    def _pool_horizontal_right(self, x: torch.Tensor) -> torch.Tensor:
        """Max pool from right to left."""
        batch, ch, h, w = x.shape
        output = torch.zeros_like(x)
        output[..., -1] = x[..., -1]
        
        for i in range(w - 2, -1, -1):
            output[..., i] = torch.max(x[..., i], output[..., i + 1])
        
        return output
    
    def _pool_horizontal_left(self, x: torch.Tensor) -> torch.Tensor:
        """Max pool from left to right."""
        batch, ch, h, w = x.shape
        output = torch.zeros_like(x)
        output[..., 0] = x[..., 0]
        
        for i in range(1, w):
            output[..., i] = torch.max(x[..., i], output[..., i - 1])
        
        return output
    
    def _pool_vertical_bottom(self, x: torch.Tensor) -> torch.Tensor:
        """Max pool from bottom to top."""
        batch, ch, h, w = x.shape
        output = torch.zeros_like(x)
        output[..., -1, :] = x[..., -1, :]
        
        for i in range(h - 2, -1, -1):
            output[..., i, :] = torch.max(x[..., i, :], output[..., i + 1, :])
        
        return output
    
    def _pool_vertical_top(self, x: torch.Tensor) -> torch.Tensor:
        """Max pool from top to bottom."""
        batch, ch, h, w = x.shape
        output = torch.zeros_like(x)
        output[..., 0, :] = x[..., 0, :]
        
        for i in range(1, h):
            output[..., i, :] = torch.max(x[..., i, :], output[..., i - 1, :])
        
        return output
```

### CenterNet (2019)

CenterNet simplifies detection by representing objects as single center points:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CenterNet: Objects as Points                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Key Insight: An object = its center point                      │
│                                                                  │
│       Ground Truth               Heatmap Prediction              │
│  ┌─────────────────┐            ┌─────────────────┐             │
│  │  ┌─────────┐    │            │       ○         │  (Gaussian) │
│  │  │    ●    │    │     →      │      ○●○        │             │
│  │  │  (car)  │    │            │       ○         │             │
│  │  └─────────┘    │            │                 │             │
│  └─────────────────┘            └─────────────────┘             │
│                                                                  │
│  Outputs per center:                                             │
│    • Center heatmap: Where are object centers?                  │
│    • Size regression: (width, height) of bounding box           │
│    • Offset: Sub-pixel center location refinement               │
└─────────────────────────────────────────────────────────────────┘
```

**Advantages**:
- Simpler than CornerNet (no corner matching needed)
- Single point per object (no grouping)
- Can extend to other tasks (pose, 3D)

```python
class CenterNetHead(nn.Module):
    """
    CenterNet detection head.
    
    Predicts center heatmaps, box sizes, and center offsets.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        head_channels: int = 64
    ):
        super().__init__()
        
        # Center heatmap head (K classes)
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, 1),
            nn.Sigmoid()  # Focal loss expects probabilities
        )
        
        # Size regression head (width, height)
        self.size = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, 1)
        )
        
        # Offset head (sub-pixel refinement)
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, 1)
        )
        
        # Initialize heatmap bias for focal loss
        self.heatmap[-2].bias.data.fill_(-2.19)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, in_channels, H, W) backbone features
            
        Returns:
            Dictionary with 'heatmap', 'size', 'offset'
        """
        return {
            'heatmap': self.heatmap(x),
            'size': self.size(x),
            'offset': self.offset(x)
        }


def generate_heatmap(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    output_size: tuple,
    num_classes: int,
    min_overlap: float = 0.7
) -> torch.Tensor:
    """
    Generate ground truth heatmaps with Gaussian peaks at object centers.
    
    Args:
        gt_boxes: (N, 4) boxes in xyxy format
        gt_labels: (N,) class labels
        output_size: (H, W) of output heatmap
        num_classes: Number of classes
        min_overlap: Minimum IoU with Gaussian radius
        
    Returns:
        (num_classes, H, W) heatmap
    """
    H, W = output_size
    heatmap = torch.zeros(num_classes, H, W, dtype=torch.float32)
    
    for box, label in zip(gt_boxes, gt_labels):
        # Compute center
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # Compute Gaussian radius based on box size
        radius = gaussian_radius((h, w), min_overlap)
        radius = max(0, int(radius))
        
        # Draw Gaussian
        cx_int, cy_int = int(cx), int(cy)
        draw_gaussian(heatmap[label], (cx_int, cy_int), radius)
    
    return heatmap


def gaussian_radius(size: tuple, min_overlap: float = 0.7) -> float:
    """
    Compute Gaussian radius for a given box size.
    
    Radius is chosen such that a box with center at radius distance
    from the true center still has IoU > min_overlap with ground truth.
    """
    height, width = size
    
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    
    return min(r1, r2, r3)


def draw_gaussian(
    heatmap: torch.Tensor,
    center: tuple,
    radius: int,
    k: float = 1.0
):
    """Draw 2D Gaussian on heatmap."""
    diameter = 2 * radius + 1
    gaussian = torch.exp(
        -torch.arange(diameter).float().sub(radius).pow(2).unsqueeze(1) / (2 * radius ** 2)
    ) * torch.exp(
        -torch.arange(diameter).float().sub(radius).pow(2).unsqueeze(0) / (2 * radius ** 2)
    )
    
    x, y = center
    H, W = heatmap.shape
    
    left = min(x, radius)
    right = min(W - x, radius + 1)
    top = min(y, radius)
    bottom = min(H - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if masked_gaussian.shape[0] > 0 and masked_gaussian.shape[1] > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
```

### CenterNet Loss

```python
def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0
) -> torch.Tensor:
    """
    Focal loss for heatmap prediction.
    
    Modified focal loss from CornerNet paper that handles
    Gaussian-smoothed ground truth.
    
    Args:
        pred: (batch, C, H, W) predicted heatmaps
        target: (batch, C, H, W) target heatmaps with Gaussian peaks
        alpha, beta: Focal loss hyperparameters
    """
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    
    neg_weights = (1 - target).pow(beta)
    
    pos_loss = -((1 - pred).pow(alpha) * pred.log() * pos_mask).sum()
    neg_loss = -((pred.pow(alpha)) * (1 - pred).log() * neg_weights * neg_mask).sum()
    
    num_pos = pos_mask.sum()
    
    if num_pos == 0:
        return neg_loss
    else:
        return (pos_loss + neg_loss) / num_pos


def centernet_loss(
    pred_heatmap: torch.Tensor,
    pred_size: torch.Tensor,
    pred_offset: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_size: torch.Tensor,
    target_offset: torch.Tensor,
    target_mask: torch.Tensor
) -> dict:
    """
    CenterNet multi-task loss.
    
    Args:
        pred_heatmap: (B, C, H, W) predicted center heatmaps
        pred_size: (B, 2, H, W) predicted sizes
        pred_offset: (B, 2, H, W) predicted offsets
        target_heatmap: (B, C, H, W) target heatmaps
        target_size: (B, max_objects, 2) target sizes
        target_offset: (B, max_objects, 2) target offsets
        target_mask: (B, max_objects) valid object mask
    """
    # Heatmap loss (focal loss)
    hm_loss = focal_loss(pred_heatmap, target_heatmap)
    
    # Size and offset losses (only at object centers)
    # ... gather predictions at center locations
    
    size_loss = F.l1_loss(pred_size_at_centers, target_size, reduction='sum')
    offset_loss = F.l1_loss(pred_offset_at_centers, target_offset, reduction='sum')
    
    num_objects = target_mask.sum()
    
    return {
        'hm_loss': hm_loss,
        'size_loss': size_loss / (num_objects + 1e-4),
        'offset_loss': offset_loss / (num_objects + 1e-4),
        'total': hm_loss + 0.1 * size_loss + offset_loss
    }
```

## FCOS: Fully Convolutional One-Stage Detection

FCOS predicts boxes by regressing distances from each location to box edges:

```
┌─────────────────────────────────────────────────────────────────┐
│                      FCOS Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each pixel location (x, y):                                │
│                                                                  │
│       l ←───┬───→ r                                             │
│             │                                                   │
│       t     ●     (x, y)  feature map location                 │
│       ↑     │                                                   │
│       │     ↓                                                   │
│             b                                                   │
│                                                                  │
│  Predict:                                                        │
│    • (l, t, r, b): Distances to box edges                       │
│    • class score: What object (if any)?                         │
│    • centerness: How close to object center?                    │
│                                                                  │
│  Box = (x - l, y - t, x + r, y + b)                             │
└─────────────────────────────────────────────────────────────────┘
```

### FCOS Implementation

```python
class FCOSHead(nn.Module):
    """
    FCOS detection head with centerness prediction.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4,
        prior_prob: float = 0.01
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Classification branch
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        
        # Regression branch  
        reg_tower = []
        for _ in range(num_convs):
            reg_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)
        
        # Output heads
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        
        # Learnable scale for each FPN level
        self.scales = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(5)])
        
        # Initialize bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
    
    def forward(self, features: list) -> tuple:
        """
        Args:
            features: List of feature maps from FPN
            
        Returns:
            cls_scores: List of (B, num_classes, H, W) per level
            bbox_preds: List of (B, 4, H, W) per level
            centernesses: List of (B, 1, H, W) per level
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        
        for i, feature in enumerate(features):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)
            
            # Classification
            cls_score = self.cls_logits(cls_feat)
            cls_scores.append(cls_score)
            
            # Box regression with exp for positive values
            bbox_pred = self.bbox_pred(reg_feat)
            bbox_pred = F.relu(bbox_pred) * self.scales[i](torch.ones(1, 1, 1, 1, device=feature.device))
            bbox_preds.append(bbox_pred)
            
            # Centerness (predicted from regression features)
            centerness = self.centerness(reg_feat)
            centernesses.append(centerness)
        
        return cls_scores, bbox_preds, centernesses


def compute_centerness(
    left: torch.Tensor,
    top: torch.Tensor,
    right: torch.Tensor,
    bottom: torch.Tensor
) -> torch.Tensor:
    """
    Compute centerness targets.
    
    Centerness measures how close a location is to the object center.
    Ranges from 0 (corner) to 1 (center).
    
    centerness = sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
    """
    lr_min = torch.min(left, right)
    lr_max = torch.max(left, right)
    tb_min = torch.min(top, bottom)
    tb_max = torch.max(top, bottom)
    
    centerness = torch.sqrt(
        (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
    )
    
    return centerness
```

### FCOS Training

**Positive Sample Assignment**:
- A location is positive if it's inside a ground truth box
- Multi-scale assignment: Different FPN levels handle different object sizes

**Loss Function**:
$$L = L_{cls} + \lambda_1 L_{reg} + \lambda_2 L_{centerness}$$

```python
class FCOSLoss(nn.Module):
    """FCOS loss function."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(
        self,
        cls_scores: list,
        bbox_preds: list,
        centernesses: list,
        targets: dict
    ) -> dict:
        """Compute FCOS losses."""
        
        # Flatten predictions
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for s in cls_scores
        ])
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(-1, 4)
            for b in bbox_preds
        ])
        all_centernesses = torch.cat([
            c.permute(0, 2, 3, 1).reshape(-1)
            for c in centernesses
        ])
        
        # Get targets
        labels = targets['labels']
        bbox_targets = targets['bbox_targets']
        centerness_targets = targets['centerness_targets']
        
        # Positive mask
        pos_mask = labels > 0
        num_pos = pos_mask.sum().float()
        
        # Classification loss (Focal Loss)
        cls_loss = sigmoid_focal_loss(
            all_cls_scores,
            labels,
            reduction='sum'
        ) / num_pos
        
        if pos_mask.sum() > 0:
            # Regression loss (IoU Loss)
            reg_loss = iou_loss(
                all_bbox_preds[pos_mask],
                bbox_targets[pos_mask],
                reduction='sum'
            ) / num_pos
            
            # Centerness loss (BCE)
            centerness_loss = F.binary_cross_entropy_with_logits(
                all_centernesses[pos_mask],
                centerness_targets[pos_mask],
                reduction='sum'
            ) / num_pos
        else:
            reg_loss = all_bbox_preds.sum() * 0
            centerness_loss = all_centernesses.sum() * 0
        
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'centerness_loss': centerness_loss,
            'total': cls_loss + reg_loss + centerness_loss
        }
```

## Comparison: Anchor-Based vs Anchor-Free

| Aspect | Anchor-Based | Anchor-Free |
|--------|--------------|-------------|
| **Hyperparameters** | Many (sizes, ratios, IoU thresholds) | Few |
| **Design Effort** | Dataset-specific tuning | General purpose |
| **Training** | Complex assignment | Simpler |
| **Memory** | Higher (anchor storage) | Lower |
| **Small Objects** | Depends on anchor design | Can be challenging |
| **Inference Speed** | Similar | Similar |
| **State-of-the-art** | YOLOv5-v7 | YOLOv8, DETR |

### When to Use Each

**Anchor-Based**:
- Well-studied datasets with known object distributions
- When you can tune anchors for your domain
- Legacy systems with anchor-based models

**Anchor-Free**:
- New datasets without prior knowledge
- General-purpose detection
- When simplicity is preferred
- Modern architectures (YOLOv8, DETR)

## Modern Anchor-Free Detectors

### DETR (2020)

Detection Transformer: End-to-end object detection with transformers

```python
# Using torchvision DETR
from torchvision.models.detection import detr_resnet50

model = detr_resnet50(pretrained=True)
model.eval()

# DETR uses set-based loss with Hungarian matching
# No NMS required!
```

### YOLOv8 (Anchor-Free)

Latest YOLO uses anchor-free detection:

```python
from ultralytics import YOLO

# YOLOv8 is anchor-free
model = YOLO('yolov8n.pt')

# Predictions directly without anchor templates
results = model('image.jpg')
```

## Summary

Anchor-free detection simplifies object detection by eliminating predefined anchor boxes:

1. **Keypoint-based** (CornerNet, CenterNet): Detect objects via keypoints
2. **Dense prediction** (FCOS): Predict boxes from every feature location
3. **Centerness**: Down-weight predictions far from object centers
4. **Modern trend**: YOLOv8, DETR are anchor-free

Key advantages:
- Fewer hyperparameters to tune
- Better generalization across datasets
- Simpler training pipelines
- Competitive or superior accuracy

Anchor-free methods are now the standard for new architectures.

## References

1. Law, H., & Deng, J. (2018). CornerNet: Detecting Objects as Paired Keypoints. *ECCV*.
2. Zhou, X., et al. (2019). Objects as Points. *arXiv*.
3. Tian, Z., et al. (2019). FCOS: Fully Convolutional One-Stage Object Detection. *ICCV*.
4. Carion, N., et al. (2020). End-to-End Object Detection with Transformers. *ECCV*.
5. Jocher, G. (2023). YOLOv8 by Ultralytics. *GitHub*.
