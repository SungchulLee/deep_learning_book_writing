# Fast R-CNN

## Learning Objectives

By the end of this section, you will be able to:

- Explain the shared computation paradigm that makes Fast R-CNN efficient
- Understand RoI Pooling and its role in variable-size region processing
- Implement the multi-task loss for joint classification and box regression
- Describe bounding box parameterization for regression targets

## Key Innovation: Shared Computation

Fast R-CNN (Girshick, 2015) addresses R-CNN's primary bottleneck: processing each proposal independently through the CNN. Instead, Fast R-CNN processes the **entire image once** through the backbone, then extracts features for each region proposal from the shared feature map.

This simple change reduces inference time from ~47 seconds to ~2 seconds per image.

## RoI Pooling

RoI (Region of Interest) Pooling converts variable-size regions into fixed-size feature maps, enabling batch processing of proposals with different dimensions:

```python
import torch
import torch.nn as nn
from torchvision.ops import roi_pool

class RoIPoolingLayer(nn.Module):
    """
    RoI Pooling converts arbitrary-size regions to fixed output.
    
    Args:
        output_size: Fixed output spatial size (H, W)
        spatial_scale: Scale factor from input image to feature map
                       (e.g., 1/16 for VGG-16 with 4 pooling layers)
    """
    def __init__(self, output_size=(7, 7), spatial_scale=1/16):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, features, rois):
        """
        Args:
            features: (N, C, H, W) feature maps from backbone
            rois: (K, 5) where each row is [batch_idx, x1, y1, x2, y2]
        
        Returns:
            (K, C, output_H, output_W) pooled features
        """
        return roi_pool(features, rois, 
                        output_size=self.output_size,
                        spatial_scale=self.spatial_scale)
```

**Limitation**: RoI Pooling quantizes floating-point coordinates to integer grid positions, introducing spatial misalignment. This was later addressed by RoI Align in Faster R-CNN / Mask R-CNN.

## Multi-Task Loss

Fast R-CNN introduced end-to-end training with a combined classification and localization loss:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{loc}}$$

**Classification Loss** (softmax cross-entropy):

$$\mathcal{L}_{\text{cls}} = -\log p_u$$

where $p_u$ is the predicted probability for the true class $u$.

**Localization Loss** (smooth L1):

$$\mathcal{L}_{\text{loc}} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t_i^u - v_i)$$

where the smooth L1 loss is defined as:

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

Smooth L1 is less sensitive to outliers than L2 loss (preventing gradient explosion for poorly matched proposals) while being differentiable everywhere (unlike L1).

## Bounding Box Parameterization

The network predicts transformations relative to the proposal box rather than absolute coordinates:

$$t_x = \frac{x - x_p}{w_p}, \quad t_y = \frac{y - y_p}{h_p}, \quad t_w = \log\frac{w}{w_p}, \quad t_h = \log\frac{h}{h_p}$$

where $(x_p, y_p, w_p, h_p)$ defines the proposal box. This parameterization is scale-invariant—the same offset magnitude applies regardless of object size.

```python
def encode_boxes(gt_boxes, proposals):
    """Encode ground-truth boxes as offsets relative to proposals."""
    eps = 1e-6
    tx = (gt_boxes[:, 0] - proposals[:, 0]) / (proposals[:, 2] + eps)
    ty = (gt_boxes[:, 1] - proposals[:, 1]) / (proposals[:, 3] + eps)
    tw = torch.log(gt_boxes[:, 2] / (proposals[:, 2] + eps))
    th = torch.log(gt_boxes[:, 3] / (proposals[:, 3] + eps))
    return torch.stack([tx, ty, tw, th], dim=-1)

def decode_boxes(deltas, proposals):
    """Decode predicted deltas back to absolute boxes."""
    x = deltas[:, 0] * proposals[:, 2] + proposals[:, 0]
    y = deltas[:, 1] * proposals[:, 3] + proposals[:, 1]
    w = torch.exp(deltas[:, 2]) * proposals[:, 2]
    h = torch.exp(deltas[:, 3]) * proposals[:, 3]
    return torch.stack([x, y, w, h], dim=-1)
```

## Summary

Fast R-CNN's key contributions:

1. **Shared backbone computation** eliminates redundant per-proposal CNN forward passes
2. **RoI Pooling** enables batch processing of variable-size regions
3. **Multi-task loss** jointly trains classification and regression end-to-end
4. **Smooth L1 loss** provides robust box regression

The remaining bottleneck—external Selective Search for proposals (~2 seconds per image)—was addressed by Faster R-CNN's Region Proposal Network.

## References

1. Girshick, R. (2015). Fast R-CNN. ICCV.
