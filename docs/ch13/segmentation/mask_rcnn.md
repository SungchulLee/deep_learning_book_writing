# Mask R-CNN

## Learning Objectives

By the end of this section, you will be able to:

- Understand Mask R-CNN's architecture as an extension of Faster R-CNN
- Explain the role of RoI Align in producing accurate pixel-level masks
- Implement mask prediction heads and mask-specific loss functions
- Use pre-trained Mask R-CNN for instance segmentation inference
- Distinguish Mask R-CNN from one-stage instance segmentation alternatives

## From Faster R-CNN to Mask R-CNN

Mask R-CNN (He et al., 2017) extends Faster R-CNN by adding a parallel **mask prediction branch** alongside the existing bounding box and classification heads. The key insight is that instance segmentation can be decomposed into detection (bounding box + class) plus per-instance binary mask prediction.

```
Input Image
     │
     ▼
┌─────────────────────────┐
│  Backbone (ResNet + FPN) │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Region Proposal Network │
│         (RPN)            │
└──────────┬──────────────┘
           │
           ▼ (Regions of Interest)
┌─────────────────────────┐
│      RoI Align           │  ← Key improvement over RoI Pooling
└──────────┬──────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌─────────┐
│Box Head │  │Mask Head│
│(class + │  │(binary  │
│ bbox)   │  │ mask)   │
└─────────┘  └─────────┘
```

### RoI Align: The Critical Innovation

Standard RoI Pooling (from Fast R-CNN) introduces spatial misalignment through quantization—rounding floating-point RoI coordinates to integer grid positions. For bounding box regression, this coarse alignment is acceptable. For pixel-level mask prediction, it causes significant degradation.

**RoI Align** eliminates quantization entirely by using bilinear interpolation to compute exact feature values at non-integer locations:

$$\text{RoI Pool}: \text{round}(x / \text{stride}) \rightarrow \text{integer grid}$$
$$\text{RoI Align}: \text{bilinear\_interpolate}(x / \text{stride}) \rightarrow \text{exact position}$$

This seemingly small change improves mask AP by 1–3 points on COCO.

## Mask Head Architecture

The mask head is a small FCN that predicts a binary mask for each detected instance. It operates on the RoI-aligned features and predicts a fixed-size mask (typically $28 \times 28$ or $14 \times 14$) per class:

```python
import torch
import torch.nn as nn

class MaskHead(nn.Module):
    """
    Mask prediction head for Mask R-CNN.
    
    Takes RoI-aligned features and predicts per-class binary masks.
    
    Args:
        in_channels: Input feature channels (from RoI Align)
        num_classes: Number of object classes
        mask_size: Output mask resolution (default: 28×28)
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 80, 
                 mask_size: int = 28):
        super().__init__()
        
        # Four 3×3 convolutions (standard Mask R-CNN design)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        
        # Upsample 2× via transposed convolution
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        # Per-class mask prediction (K binary masks, one per class)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RoI-aligned features (N, C, mask_size/2, mask_size/2)
        
        Returns:
            Per-class mask logits (N, num_classes, mask_size, mask_size)
        """
        x = self.conv_layers(x)
        x = self.relu(self.deconv(x))
        return self.mask_pred(x)
```

### Mask Loss

Mask R-CNN uses **per-pixel binary cross-entropy** loss, but only for the mask corresponding to the ground-truth class. This decouples mask prediction from classification—the mask head does not need to compete across classes:

$$\mathcal{L}_{\text{mask}} = -\frac{1}{m^2} \sum_{i,j} \left[ y_{ij} \log \hat{y}_{ij}^{(k)} + (1 - y_{ij}) \log(1 - \hat{y}_{ij}^{(k)}) \right]$$

where $k$ is the ground-truth class for the instance and $m$ is the mask resolution.

```python
def mask_rcnn_loss(mask_logits: torch.Tensor, gt_masks: torch.Tensor, 
                   gt_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute Mask R-CNN mask loss.
    
    Only the mask for the ground-truth class contributes to the loss.
    
    Args:
        mask_logits: Predicted masks (N, K, m, m)
        gt_masks: Ground-truth binary masks (N, m, m)
        gt_labels: Ground-truth class labels (N,)
    """
    # Select mask for ground-truth class
    N = mask_logits.shape[0]
    indices = torch.arange(N, device=mask_logits.device)
    selected_masks = mask_logits[indices, gt_labels]  # (N, m, m)
    
    return nn.functional.binary_cross_entropy_with_logits(
        selected_masks, gt_masks.float()
    )
```

### Multi-Task Loss

The complete Mask R-CNN loss combines three terms:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}$$

Each head operates independently on the same RoI features, and the mask loss only applies to positive (matched) proposals.

## Using Pre-trained Mask R-CNN

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

def load_and_run_maskrcnn(image: torch.Tensor, threshold: float = 0.5):
    """
    Run pre-trained Mask R-CNN on an input image.
    
    Args:
        image: Input tensor (3, H, W), values in [0, 1]
        threshold: Confidence threshold for detections
    
    Returns:
        Filtered predictions with boxes, labels, scores, and masks
    """
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    with torch.no_grad():
        predictions = model([image])[0]
    
    keep = predictions['scores'] > threshold
    
    return {
        'boxes': predictions['boxes'][keep],       # (N, 4) xyxy format
        'labels': predictions['labels'][keep],     # (N,) class indices
        'scores': predictions['scores'][keep],     # (N,) confidence
        'masks': predictions['masks'][keep] > 0.5  # (N, 1, H, W) binary masks
    }
```

### Fine-tuning for Custom Classes

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_custom_maskrcnn(num_classes: int):
    """Create Mask R-CNN with custom number of classes."""
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model
```

## Comparison with One-Stage Approaches

| Method | Type | Speed (FPS) | Mask AP (COCO) | Use Case |
|--------|------|-------------|----------------|----------|
| Mask R-CNN | Two-stage | ~5 | 37.1 | Accuracy-critical |
| YOLACT | One-stage | ~30 | 29.8 | Real-time |
| SOLOv2 | One-stage | ~15 | 37.8 | Balanced |
| PointRend | Refinement | ~5 | 38.3 | High-resolution masks |

Mask R-CNN remains the standard two-stage approach. One-stage methods like YOLACT trade accuracy for speed, while modern approaches like SOLOv2 achieve competitive accuracy without region proposals.

## Summary

Mask R-CNN's key contributions:

1. **Simple extension**: Adding a mask branch to Faster R-CNN with minimal overhead
2. **RoI Align**: Eliminating quantization for precise spatial alignment
3. **Decoupled prediction**: Per-class binary masks avoid inter-class competition
4. **Multi-task training**: Joint optimization of detection and segmentation

The architecture established instance segmentation as a tractable problem and remains the foundation for modern approaches including Cascade Mask R-CNN and PointRend.

## References

1. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
2. Bolya, D., et al. (2019). YOLACT: Real-time Instance Segmentation. ICCV.
3. Wang, X., et al. (2020). SOLOv2: Dynamic and Fast Instance Segmentation. NeurIPS.
4. Kirillov, A., et al. (2020). PointRend: Image Segmentation as Rendering. CVPR.
