# Region Proposal Networks

## Learning Objectives

By the end of this section, you will be able to:

- Understand how RPNs generate class-agnostic region proposals
- Implement anchor generation with multiple scales and aspect ratios
- Explain the anchor assignment strategy and RPN loss function
- Describe the proposal filtering and NMS pipeline

## Motivation

Before RPNs, object detectors relied on external proposal methods like Selective Search (~2,000 proposals per image, taking ~2 seconds). RPNs replace this with a lightweight neural network that shares the backbone's feature map, generating proposals in ~10ms.

## Architecture

The RPN is a small fully convolutional network that slides over the backbone's output feature map. At each spatial position, it evaluates $k$ anchor boxes and predicts:

1. **Objectness score**: probability that the anchor contains any object (2 classes: object vs. background)
2. **Box regression**: four offsets $(t_x, t_y, t_w, t_h)$ to refine the anchor's position

$$\text{RPN output at each position: } k \text{ objectness scores} + k \times 4 \text{ box offsets}$$

For a feature map of size $H \times W$ with $k$ anchors per position, the RPN produces $H \times W \times k$ candidate proposals.

## Anchor Generation

Anchors are predefined reference boxes centered at each feature map position:

```python
import torch

def generate_anchors(feature_size, stride, 
                     scales=(32, 64, 128, 256, 512),
                     ratios=(0.5, 1.0, 2.0)):
    """
    Generate anchors for a single feature map level.
    
    Args:
        feature_size: (H, W) of the feature map
        stride: Downsampling factor from input to feature map
        scales: Anchor sizes in pixels (relative to input image)
        ratios: Width/height aspect ratios
    
    Returns:
        Tensor of shape (H*W*k, 4) in xyxy format
    """
    H, W = feature_size
    k = len(scales) * len(ratios)
    
    # Base anchors centered at origin
    base = []
    for s in scales:
        for r in ratios:
            w = s * (r ** 0.5)
            h = s / (r ** 0.5)
            base.append([-w/2, -h/2, w/2, h/2])
    base = torch.tensor(base)  # (k, 4)
    
    # Grid of centers
    cx = (torch.arange(W) + 0.5) * stride
    cy = (torch.arange(H) + 0.5) * stride
    cy, cx = torch.meshgrid(cy, cx, indexing='ij')
    centers = torch.stack([cx, cy, cx, cy], dim=-1).reshape(-1, 4)  # (H*W, 4)
    
    # Broadcast: (H*W, 1, 4) + (1, k, 4) → (H*W, k, 4)
    anchors = centers.unsqueeze(1) + base.unsqueeze(0)
    return anchors.reshape(-1, 4)
```

### Multi-Scale Anchors with FPN

When used with Feature Pyramid Networks (FPN), each pyramid level uses anchors of a single scale:

| FPN Level | Stride | Anchor Scale | Feature Map |
|-----------|--------|-------------|-------------|
| P2 | 4 | 32 | Large (high-res) |
| P3 | 8 | 64 | — |
| P4 | 16 | 128 | — |
| P5 | 32 | 256 | — |
| P6 | 64 | 512 | Small (low-res) |

Each level still uses multiple aspect ratios (typically 0.5, 1.0, 2.0).

## Anchor Assignment

During training, each anchor is assigned a label:

- **Positive** ($p^* = 1$): IoU > 0.7 with any ground truth box, or the highest-IoU anchor for each ground truth
- **Negative** ($p^* = 0$): IoU < 0.3 with all ground truth boxes
- **Ignored** ($p^* = -1$): IoU between 0.3 and 0.7 (excluded from loss)

A mini-batch of 256 anchors is sampled per image, with a 1:1 positive-to-negative ratio (if fewer than 128 positives exist, extra negatives fill the batch).

## RPN Loss

$$\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}} \sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* \cdot \text{smooth}_{L_1}(t_i - t_i^*)$$

- $\mathcal{L}_{\text{cls}}$: binary cross-entropy for objectness
- $\text{smooth}_{L_1}$: regression loss (only for positive anchors)
- $\lambda = 10$ balances the two terms

```python
import torch.nn.functional as F

def rpn_loss(cls_logits, box_deltas, labels, reg_targets, lambda_reg=10.0):
    """
    Compute RPN loss.
    
    Args:
        cls_logits: (N, num_anchors, 2) objectness logits
        box_deltas: (N, num_anchors, 4) predicted box offsets
        labels: (N, num_anchors) anchor labels (1=pos, 0=neg, -1=ignore)
        reg_targets: (N, num_anchors, 4) target offsets for positive anchors
    """
    # Classification: exclude ignored anchors
    valid = labels >= 0
    cls_loss = F.cross_entropy(cls_logits[valid], labels[valid].long(), reduction='mean')
    
    # Regression: only positive anchors
    pos = labels == 1
    if pos.sum() > 0:
        reg_loss = F.smooth_l1_loss(box_deltas[pos], reg_targets[pos], reduction='mean')
    else:
        reg_loss = torch.tensor(0.0, device=box_deltas.device)
    
    return cls_loss + lambda_reg * reg_loss
```

## Proposal Filtering

After the RPN forward pass, proposals are filtered before entering the second stage:

1. **Score thresholding**: Remove proposals with low objectness
2. **Top-K selection**: Keep top ~2,000 proposals by score (training) or ~1,000 (inference)
3. **NMS**: Apply non-maximum suppression with IoU threshold 0.7 to remove duplicates
4. **Final selection**: Keep top ~300 proposals after NMS

```python
from torchvision.ops import nms

def filter_proposals(boxes, scores, pre_nms_topk=2000, post_nms_topk=300, 
                     nms_threshold=0.7, score_threshold=0.0):
    """Filter RPN proposals."""
    # Score threshold
    keep = scores > score_threshold
    boxes, scores = boxes[keep], scores[keep]
    
    # Top-K before NMS
    if len(scores) > pre_nms_topk:
        topk = scores.topk(pre_nms_topk).indices
        boxes, scores = boxes[topk], scores[topk]
    
    # NMS
    keep = nms(boxes, scores, nms_threshold)
    
    # Top-K after NMS
    keep = keep[:post_nms_topk]
    
    return boxes[keep], scores[keep]
```

## Summary

The Region Proposal Network is the component that made Faster R-CNN fully end-to-end:

1. **Anchors** provide multi-scale, multi-ratio reference boxes
2. **Shared features** eliminate the proposal generation bottleneck
3. **Assignment strategy** handles the massive positive/negative imbalance
4. **NMS filtering** produces a manageable set of high-quality proposals

RPNs generalize beyond Faster R-CNN—the concept of predicting objectness and box offsets at anchor locations is the basis for many one-stage detectors as well.

## References

1. Ren, S., et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS.
2. Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR.
