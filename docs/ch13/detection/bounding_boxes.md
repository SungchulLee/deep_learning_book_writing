# Bounding Box Representations

## Learning Objectives

By the end of this section, you will be able to:

- Convert between the three standard bounding box formats (xyxy, xywh, cxcywh)
- Understand normalized vs. absolute coordinate systems
- Implement efficient box format conversions in PyTorch
- Apply bounding box parameterization for regression targets

## Bounding Box Formats

Object detection uses three standard bounding box representations. Different frameworks and models adopt different conventions, so reliable conversion is essential.

### Format 1: Corner Coordinates (xyxy)

Specifies the top-left and bottom-right corners:

$$\mathbf{b} = (x_1, y_1, x_2, y_2)$$

where $(x_1, y_1)$ is the top-left corner and $(x_2, y_2)$ is the bottom-right corner. Used by torchvision, COCO evaluation, and most inference APIs.

### Format 2: Position and Size (xywh)

Specifies the top-left corner plus width and height:

$$\mathbf{b} = (x, y, w, h)$$

Used by COCO annotations and many dataset formats.

### Format 3: Center and Size (cxcywh)

Specifies the center point plus width and height:

$$\mathbf{b} = (c_x, c_y, w, h)$$

Used internally by YOLO, SSD, and most anchor-based detectors. Natural for regression since predicted offsets are relative to anchor centers.

### Conversion Relationships

$$\text{xyxy} \leftrightarrow \text{xywh}: \quad x_1 = x,\ y_1 = y,\ x_2 = x + w,\ y_2 = y + h$$

$$\text{xyxy} \leftrightarrow \text{cxcywh}: \quad c_x = \frac{x_1 + x_2}{2},\ c_y = \frac{y_1 + y_2}{2},\ w = x_2 - x_1,\ h = y_2 - y_1$$

### Normalized Coordinates

Some frameworks normalize coordinates to $[0, 1]$ by dividing by image dimensions:

$$x_{\text{norm}} = \frac{x}{W}, \quad y_{\text{norm}} = \frac{y}{H}$$

This makes predictions resolution-independent, which simplifies multi-scale training.

## PyTorch Implementation

```python
import torch

class BoundingBoxConverter:
    """Convert between bounding box formats efficiently."""
    
    @staticmethod
    def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
        """(x1, y1, x2, y2) → (x, y, w, h)"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
    
    @staticmethod
    def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """(x, y, w, h) → (x1, y1, x2, y2)"""
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x, y, x + w, y + h], dim=-1)
    
    @staticmethod
    def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """(x1, y1, x2, y2) → (cx, cy, w, h)"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([
            (x1 + x2) / 2, (y1 + y2) / 2,
            x2 - x1, y2 - y1
        ], dim=-1)
    
    @staticmethod
    def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """(cx, cy, w, h) → (x1, y1, x2, y2)"""
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2
        ], dim=-1)
    
    @staticmethod
    def normalize(boxes: torch.Tensor, image_size: tuple) -> torch.Tensor:
        """Normalize absolute coordinates to [0, 1]."""
        h, w = image_size
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        return boxes / scale
    
    @staticmethod
    def denormalize(boxes: torch.Tensor, image_size: tuple) -> torch.Tensor:
        """Convert normalized coordinates back to absolute."""
        h, w = image_size
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        return boxes * scale
```

## Bounding Box Parameterization for Regression

Rather than regressing absolute coordinates, detectors predict offsets relative to anchors or reference points. The standard parameterization (from R-CNN):

$$t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a}$$

where $(x_a, y_a, w_a, h_a)$ is the anchor box. The log transform ensures width/height predictions are scale-invariant and always positive after exponentiation.

```python
def encode_boxes(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Encode ground-truth boxes as offsets relative to anchors.
    
    Both inputs in cxcywh format: (cx, cy, w, h)
    """
    tx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    ty = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
    tw = torch.log(gt_boxes[:, 2] / anchors[:, 2])
    th = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    return torch.stack([tx, ty, tw, th], dim=1)

def decode_boxes(offsets: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Decode predicted offsets back to absolute boxes.
    """
    cx = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
    w = torch.exp(offsets[:, 2]) * anchors[:, 2]
    h = torch.exp(offsets[:, 3]) * anchors[:, 3]
    return torch.stack([cx, cy, w, h], dim=1)
```

## Summary

Bounding box representation is foundational to all detection systems:

1. **Three formats**: xyxy (corner), xywh (position+size), cxcywh (center+size) serve different purposes
2. **Normalization** enables resolution-independent predictions
3. **Parameterized regression** predicts relative offsets rather than absolute coordinates, improving training stability
4. **Consistent conversion** between formats is critical when combining different frameworks

## References

1. Girshick, R. (2015). Fast R-CNN. ICCV.
2. Ren, S., et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS.
