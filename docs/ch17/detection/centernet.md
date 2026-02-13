# CenterNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand keypoint-based object detection as an alternative to anchor-based methods
- Explain the CenterNet approach of detecting objects as center points
- Implement the center heatmap, offset, and size prediction heads
- Describe the advantages of anchor-free, NMS-free detection

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


## References

1. Law, H., & Deng, J. (2018). CornerNet: Detecting Objects as Paired Keypoints. ECCV.
2. Zhou, X., Wang, D., & Krähenbühl, P. (2019). Objects as Points. arXiv.
3. Duan, K., et al. (2019). CenterNet: Keypoint Triplets for Object Detection. ICCV.
