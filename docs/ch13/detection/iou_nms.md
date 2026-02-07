# Intersection over Union (IoU) and Non-Maximum Suppression (NMS)

## Learning Objectives

By the end of this section, you will be able to:

- Derive and implement Intersection over Union (IoU) from first principles
- Understand IoU variants (GIoU, DIoU, CIoU) and their advantages
- Implement Non-Maximum Suppression (NMS) and understand its role in detection
- Apply NMS variants (Soft-NMS, DIoU-NMS) for improved results
- Analyze the computational complexity and optimization strategies for these operations

## Intersection over Union (IoU)

### Definition and Intuition

Intersection over Union (IoU), also called the **Jaccard Index**, measures the overlap between two regions. For bounding boxes $A$ and $B$:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

Geometrically, this ratio captures how well two boxes align:

```
         Box A              Box B           Intersection        Union
        ┌──────┐          ┌──────┐          ┌────┐         ┌──────────┐
        │      │          │      │          │████│         │          │
        │  ┌───┼──────────┼───┐  │    →     │████│    /    │          │
        │  │   │          │   │  │          └────┘         │          │
        └──┼───┘          └───┼──┘                         └──────────┘
           └──────────────────┘
```

### Properties of IoU

1. **Bounded**: $0 \leq \text{IoU} \leq 1$
2. **Symmetric**: $\text{IoU}(A, B) = \text{IoU}(B, A)$
3. **Scale-invariant**: Result depends only on relative overlap, not absolute size
4. **IoU = 0**: No overlap between boxes
5. **IoU = 1**: Perfect overlap (identical boxes)

### Interpretation Guidelines

| IoU Range | Interpretation | Typical Use |
|-----------|----------------|-------------|
| 0.00 - 0.20 | Poor overlap | Likely different objects |
| 0.20 - 0.50 | Partial overlap | Ambiguous cases |
| 0.50 - 0.75 | Good overlap | Standard detection threshold |
| 0.75 - 0.90 | Strong overlap | Strict evaluation (AP@75) |
| 0.90 - 1.00 | Excellent overlap | High-precision applications |

### Mathematical Derivation

For two boxes in $(x_{min}, y_{min}, x_{max}, y_{max})$ format:

**Box A**: $(x_1^A, y_1^A, x_2^A, y_2^A)$

**Box B**: $(x_1^B, y_1^B, x_2^B, y_2^B)$

**Intersection coordinates**:
$$x_1^I = \max(x_1^A, x_1^B), \quad y_1^I = \max(y_1^A, y_1^B)$$
$$x_2^I = \min(x_2^A, x_2^B), \quad y_2^I = \min(y_2^A, y_2^B)$$

**Intersection area** (zero if no overlap):
$$\text{Area}_I = \max(0, x_2^I - x_1^I) \times \max(0, y_2^I - y_1^I)$$

**Individual areas**:
$$\text{Area}_A = (x_2^A - x_1^A) \times (y_2^A - y_1^A)$$
$$\text{Area}_B = (x_2^B - x_1^B) \times (y_2^B - y_1^B)$$

**Union area** (inclusion-exclusion principle):
$$\text{Area}_U = \text{Area}_A + \text{Area}_B - \text{Area}_I$$

**Final IoU**:
$$\text{IoU} = \frac{\text{Area}_I}{\text{Area}_U}$$

### PyTorch Implementation

```python
import torch
from typing import Union


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    
    This implementation supports batched computation for efficiency.
    
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Tensor of shape (N, M) containing pairwise IoU values
        
    Example:
        >>> boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
        >>> boxes2 = torch.tensor([[5, 5, 15, 15], [20, 20, 30, 30]], dtype=torch.float32)
        >>> iou = box_iou(boxes1, boxes2)
        >>> print(iou)
        tensor([[0.1429, 0.0000],
                [1.0000, 0.0000]])
    """
    # Extract coordinates
    # boxes1: (N, 4) -> (N, 1, 4) for broadcasting
    # boxes2: (M, 4) -> (1, M, 4) for broadcasting
    x1_1, y1_1, x2_1, y2_1 = boxes1.unsqueeze(1).unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = boxes2.unsqueeze(0).unbind(-1)
    
    # Compute intersection coordinates
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    
    # Compute intersection area (clamp to 0 for non-overlapping boxes)
    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_width * inter_height
    
    # Compute individual areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Compute union area
    union_area = area1 + area2 - inter_area
    
    # Compute IoU
    iou = inter_area / (union_area + eps)
    
    return iou


def box_iou_single(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two individual boxes.
    
    Args:
        box1: Tensor of shape (4,) in xyxy format
        box2: Tensor of shape (4,) in xyxy format
        
    Returns:
        IoU value as float
    """
    return box_iou(box1.unsqueeze(0), box2.unsqueeze(0)).item()
```

### Vectorized Implementation for Training

During training, we often need IoU between prediction and ground truth tensors:

```python
def batch_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU between corresponding boxes (not pairwise).
    
    Useful during training when predictions are already matched to targets.
    
    Args:
        pred_boxes: (N, 4) predicted boxes in xyxy format
        target_boxes: (N, 4) target boxes in xyxy format
        
    Returns:
        (N,) IoU values for each prediction-target pair
    """
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    # Areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])
    
    # Union
    union_area = pred_area + target_area - inter_area
    
    return inter_area / (union_area + 1e-7)
```

## IoU as a Loss Function

Standard IoU has limitations when used directly as a loss function:

1. **Zero gradient when no overlap**: If $\text{IoU} = 0$, the loss provides no learning signal
2. **Doesn't distinguish how boxes don't overlap**: Two non-overlapping boxes far apart have the same IoU=0 as boxes that are close but don't overlap

### Generalized IoU (GIoU)

GIoU addresses the zero-gradient problem by considering the smallest enclosing box:

$$\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}$$

where $C$ is the smallest box enclosing both $A$ and $B$.

**Properties**:
- Range: $[-1, 1]$ (can be negative when boxes don't overlap)
- Equals IoU when boxes overlap perfectly
- Provides gradient signal even for non-overlapping boxes

```python
def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute Generalized IoU between corresponding boxes.
    
    GIoU = IoU - (Area of enclosing box - Union) / Area of enclosing box
    
    Args:
        boxes1: (N, 4) boxes in xyxy format
        boxes2: (N, 4) boxes in xyxy format
        
    Returns:
        (N,) GIoU values
        
    Reference:
        Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019
    """
    # Standard IoU computation
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + eps)
    
    # Enclosing box
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + eps)
    
    return giou
```

### Distance IoU (DIoU)

DIoU adds a penalty based on the distance between box centers:

$$\text{DIoU}(A, B) = \text{IoU}(A, B) - \frac{\rho^2(A, B)}{c^2}$$

where:
- $\rho(A, B)$ is the Euclidean distance between box centers
- $c$ is the diagonal length of the smallest enclosing box

**Advantages**:
- Faster convergence than GIoU
- Directly minimizes center distance
- Better for box regression

### Complete IoU (CIoU)

CIoU adds an aspect ratio consistency term:

$$\text{CIoU}(A, B) = \text{IoU}(A, B) - \frac{\rho^2(A, B)}{c^2} - \alpha v$$

where:
$$v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$$
$$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

```python
def complete_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute Complete IoU (CIoU) between corresponding boxes.
    
    CIoU considers overlap, center distance, and aspect ratio.
    
    Args:
        boxes1: (N, 4) predicted boxes in xyxy format
        boxes2: (N, 4) target boxes in xyxy format
        
    Returns:
        (N,) CIoU values
        
    Reference:
        Zheng et al., "Distance-IoU Loss", AAAI 2020
    """
    # IoU computation
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + eps)
    
    # Center distance
    center1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
    center1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
    center2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
    center2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    center_dist_sq = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
    
    # Enclosing box diagonal
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # Aspect ratio term
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]
    
    v = (4 / (torch.pi ** 2)) * \
        (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    # CIoU
    ciou = iou - center_dist_sq / (enclose_diag_sq + eps) - alpha * v
    
    return ciou
```

### IoU Variant Comparison

| Variant | Formula | Range | Key Benefit |
|---------|---------|-------|-------------|
| **IoU** | $\frac{I}{U}$ | [0, 1] | Simple, intuitive |
| **GIoU** | $\text{IoU} - \frac{C - U}{C}$ | [-1, 1] | Gradient for non-overlap |
| **DIoU** | $\text{IoU} - \frac{\rho^2}{c^2}$ | [-1, 1] | Faster convergence |
| **CIoU** | $\text{DIoU} - \alpha v$ | [-1, 1] | Aspect ratio awareness |

## Non-Maximum Suppression (NMS)

### The Duplicate Detection Problem

Object detectors generate dense predictions across the image. For each actual object, multiple overlapping boxes with similar confidence scores are predicted:

```
Before NMS:                     After NMS:
┌─────┐                         
│ 0.9 │ ← High confidence       ┌─────┐
└─────┘                         │ 0.9 │ ← Keep best
  ┌─────┐                       └─────┘
  │ 0.8 │ ← Also high           
  └─────┘                       Removed: 0.8, 0.7 (overlapping)
    ┌─────┐
    │ 0.7 │ 
    └─────┘
```

NMS selects the best detection and removes redundant overlapping boxes.

### Standard NMS Algorithm

```
Algorithm: Non-Maximum Suppression
Input: Boxes B, Scores S, IoU threshold τ
Output: Kept indices K

1. Sort boxes by score in descending order
2. Initialize K = []
3. While B is not empty:
   a. Select box with highest score, add to K
   b. Compute IoU of this box with all remaining boxes
   c. Remove boxes with IoU > τ from B
4. Return K
```

### PyTorch Implementation

```python
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Perform Non-Maximum Suppression on detection boxes.
    
    This is a pure Python implementation for educational purposes.
    For production, use torchvision.ops.nms which is optimized in C++.
    
    Args:
        boxes: (N, 4) bounding boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
        
    Complexity:
        Time: O(N² × 4) for N boxes
        Space: O(N) for indices
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # Sort by score (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    
    while sorted_indices.numel() > 0:
        # Keep the highest scoring box
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if sorted_indices.numel() == 1:
            break
        
        # Get remaining indices
        remaining_indices = sorted_indices[1:]
        
        # Compute IoU between current box and remaining boxes
        current_box = boxes[current_idx].unsqueeze(0)
        remaining_boxes = boxes[remaining_indices]
        
        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU below threshold
        mask = ious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Perform NMS separately for each class.
    
    Boxes from different classes don't suppress each other.
    
    Args:
        boxes: (N, 4) bounding boxes
        scores: (N,) confidence scores
        labels: (N,) class labels
        iou_threshold: IoU threshold
        
    Returns:
        Indices of boxes to keep
    """
    # Offset boxes by class to prevent cross-class suppression
    max_coord = boxes.max()
    offsets = labels.float() * (max_coord + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    return nms(boxes_for_nms, scores, iou_threshold)
```

### Using torchvision NMS

For production code, use the optimized C++ implementation:

```python
import torchvision.ops as ops

# Standard NMS
keep_indices = ops.nms(boxes, scores, iou_threshold=0.5)

# Batched NMS (per-class)
keep_indices = ops.batched_nms(boxes, scores, labels, iou_threshold=0.5)

# Filter results
final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
final_labels = labels[keep_indices]
```

## NMS Variants

### Soft-NMS

Standard NMS uses a hard cutoff: boxes above the IoU threshold are completely removed. Soft-NMS instead reduces scores of overlapping boxes proportionally:

**Gaussian weighting**:
$$s_i = s_i \cdot e^{-\frac{\text{IoU}(M, b_i)^2}{\sigma}}$$

**Linear weighting**:
$$s_i = \begin{cases} s_i & \text{if IoU} < N_t \\ s_i(1 - \text{IoU}) & \text{if IoU} \geq N_t \end{cases}$$

```python
def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
    sigma: float = 0.5,
    method: str = 'gaussian'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Soft Non-Maximum Suppression.
    
    Instead of removing overlapping boxes, reduces their scores.
    Better for crowded scenes with overlapping objects.
    
    Args:
        boxes: (N, 4) bounding boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for linear method
        score_threshold: Minimum score to keep
        sigma: Gaussian decay parameter
        method: 'linear' or 'gaussian'
        
    Returns:
        Tuple of (kept_boxes, new_scores)
        
    Reference:
        Bodla et al., "Soft-NMS", ICCV 2017
    """
    boxes = boxes.clone()
    scores = scores.clone()
    
    indices = torch.arange(boxes.shape[0], device=boxes.device)
    keep = []
    
    while scores.numel() > 0:
        # Get highest scoring box
        max_idx = scores.argmax()
        keep.append(indices[max_idx])
        
        if scores.numel() == 1:
            break
        
        # Get current box and compute IoU with rest
        current_box = boxes[max_idx:max_idx+1]
        
        # Remove current box from consideration
        mask = torch.ones(scores.numel(), dtype=torch.bool, device=boxes.device)
        mask[max_idx] = False
        boxes = boxes[mask]
        scores = scores[mask]
        indices = indices[mask]
        
        # Compute IoU
        ious = box_iou(current_box, boxes).squeeze(0)
        
        # Decay scores based on IoU
        if method == 'gaussian':
            decay = torch.exp(-(ious ** 2) / sigma)
        else:  # linear
            decay = torch.where(ious >= iou_threshold, 1 - ious, torch.ones_like(ious))
        
        scores = scores * decay
        
        # Remove low-scoring boxes
        keep_mask = scores >= score_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        indices = indices[keep_mask]
    
    keep = torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)
    return keep, scores
```

### DIoU-NMS

Uses DIoU instead of IoU for suppression, considering center distance:

$$R_{DIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2}$$

This helps distinguish boxes that have similar IoU but different center positions, reducing false suppressions in crowded scenes.

```python
def diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    beta: float = 0.6
) -> torch.Tensor:
    """
    DIoU-based Non-Maximum Suppression.
    
    Uses DIoU instead of IoU for better handling of overlapping objects.
    
    Args:
        boxes: (N, 4) bounding boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: DIoU threshold for suppression
        beta: Exponent for DIoU (controls center distance weight)
        
    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while sorted_indices.numel() > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if sorted_indices.numel() == 1:
            break
        
        remaining_indices = sorted_indices[1:]
        
        # Compute DIoU
        current_box = boxes[current_idx].unsqueeze(0).expand(len(remaining_indices), -1)
        remaining_boxes = boxes[remaining_indices]
        
        dious = compute_diou(current_box, remaining_boxes)
        
        # Suppress based on DIoU
        mask = dious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def compute_diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute DIoU between corresponding boxes."""
    # IoU
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    iou = inter_area / (area1 + area2 - inter_area + 1e-7)
    
    # Center distance
    c1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
    c1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
    c2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
    c2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    center_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
    
    # Enclosing box diagonal
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    
    diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
    
    return iou - center_dist_sq / (diag_sq + 1e-7)
```

### NMS Variant Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Hard NMS** | Simple, fast | Misses overlapping objects | Sparse scenes |
| **Soft-NMS** | Better recall | Slower, needs tuning | Crowded scenes |
| **DIoU-NMS** | Center-aware | More computation | Varying object sizes |

## Complete Detection Post-Processing Pipeline

```python
def detection_postprocess(
    predictions: dict,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    max_detections: int = 100
) -> dict:
    """
    Complete post-processing pipeline for object detection.
    
    Args:
        predictions: Dict with 'boxes', 'scores', 'labels' tensors
        conf_threshold: Confidence threshold for filtering
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to return
        
    Returns:
        Dict with filtered 'boxes', 'scores', 'labels'
    """
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']
    
    # Step 1: Confidence thresholding
    conf_mask = scores >= conf_threshold
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    labels = labels[conf_mask]
    
    if boxes.numel() == 0:
        return {
            'boxes': torch.empty(0, 4, device=boxes.device),
            'scores': torch.empty(0, device=scores.device),
            'labels': torch.empty(0, dtype=torch.long, device=labels.device)
        }
    
    # Step 2: Class-wise NMS
    keep = ops.batched_nms(boxes, scores, labels, nms_threshold)
    
    # Step 3: Limit detections
    if len(keep) > max_detections:
        # Keep top-k by score
        _, top_k_indices = scores[keep].topk(max_detections)
        keep = keep[top_k_indices]
    
    return {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }
```

## Performance Optimization

### CUDA-Optimized NMS

For GPU acceleration, ensure boxes are on CUDA:

```python
# Ensure tensors are on GPU
boxes = boxes.cuda()
scores = scores.cuda()

# torchvision NMS automatically uses CUDA kernels
keep = ops.nms(boxes, scores, iou_threshold=0.5)
```

### Batched Processing

Process multiple images efficiently:

```python
def batched_detection_postprocess(
    batch_predictions: list[dict],
    **kwargs
) -> list[dict]:
    """Process batch of predictions."""
    return [detection_postprocess(pred, **kwargs) for pred in batch_predictions]
```

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| IoU (pairwise) | O(N × M) | O(N × M) |
| NMS (naive) | O(N²) | O(N) |
| NMS (optimized) | O(N log N + K × N) | O(N) |
| Soft-NMS | O(N²) | O(N) |

Where N is number of boxes, M is number of ground truth, K is number of kept boxes.

## Summary

IoU and NMS are fundamental building blocks in object detection:

**Intersection over Union (IoU)**:
- Measures overlap quality between bounding boxes
- Range [0, 1] with 0.5 being a common threshold
- Variants (GIoU, DIoU, CIoU) improve training by addressing edge cases

**Non-Maximum Suppression (NMS)**:
- Removes duplicate detections of the same object
- Greedy algorithm: keep best, remove overlapping
- Variants (Soft-NMS, DIoU-NMS) handle crowded scenes better

**Key Implementation Points**:
- Vectorize IoU computation for efficiency
- Use `torchvision.ops.nms` for production code
- Apply NMS per-class to avoid cross-class suppression
- Tune thresholds based on your use case

## References

1. Rezatofighi, H., et al. (2019). Generalized Intersection over Union: A Metric and a Loss for Bounding Box Regression. *CVPR*.
2. Zheng, Z., et al. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. *AAAI*.
3. Bodla, N., et al. (2017). Soft-NMS: Improving Object Detection with One Line of Code. *ICCV*.
4. Neubeck, A., & Van Gool, L. (2006). Efficient Non-Maximum Suppression. *ICPR*.
