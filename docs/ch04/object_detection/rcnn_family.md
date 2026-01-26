# R-CNN Family: Two-Stage Object Detection

## Learning Objectives

By the end of this section, you will be able to:

- Trace the evolution from R-CNN to Faster R-CNN and understand the innovations at each stage
- Explain the region proposal mechanism and its role in two-stage detection
- Implement and understand the Region Proposal Network (RPN)
- Describe how RoI pooling enables variable-size region processing
- Understand the multi-task loss formulation for joint classification and regression
- Extend the framework to instance segmentation with Mask R-CNN

## Overview of Two-Stage Detection

Two-stage detectors decompose object detection into two sequential problems:

**Stage 1 - Region Proposal**: Generate candidate regions likely to contain objects (class-agnostic)

**Stage 2 - Classification & Refinement**: Classify each proposal and refine its bounding box coordinates

This approach achieves high accuracy by focusing computational resources on promising regions rather than processing the entire image densely.

## R-CNN: Regions with CNN Features (2014)

R-CNN pioneered the use of deep CNNs for object detection:

### Architecture

1. **Selective Search**: Generates ~2000 region proposals using hierarchical segmentation
2. **Feature Extraction**: Each proposal is warped to 227×227 and processed through AlexNet
3. **Classification**: Linear SVMs classify each region's 4096-d feature vector
4. **Bounding Box Regression**: Linear regressor refines proposal coordinates

### Limitations

- **Slow**: Each region processed independently (~47 seconds/image)
- **Multi-stage training**: CNN, SVMs, and regressors trained separately
- **High storage**: Features must be cached for SVM training

## Fast R-CNN (2015)

### Key Innovation: Shared Computation

Fast R-CNN processes the entire image once through the CNN, then extracts features for each region from the shared feature map.

### RoI Pooling

RoI (Region of Interest) Pooling converts variable-size regions into fixed-size feature maps:

```python
import torch
import torch.nn as nn
from torchvision.ops import roi_pool, roi_align


class RoIPoolingDemo(nn.Module):
    """
    Demonstration of RoI Pooling operation.
    """
    def __init__(self, output_size=(7, 7), spatial_scale=1/16):
        """
        Args:
            output_size: Fixed output spatial size (H, W)
            spatial_scale: Scale factor from input image to feature map
        """
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
        return roi_pool(
            features, 
            rois, 
            output_size=self.output_size,
            spatial_scale=self.spatial_scale
        )
```

### Multi-Task Loss

Fast R-CNN introduced end-to-end training with a combined loss:

$$L = L_{cls} + \lambda L_{loc}$$

**Classification Loss** (softmax cross-entropy):
$$L_{cls} = -\log p_u$$

where $p_u$ is the predicted probability for the true class $u$.

**Localization Loss** (smooth L1):
$$L_{loc} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t_i^u - v_i)$$

where:
$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

### Bounding Box Parameterization

The network predicts transformations relative to the proposal box:

$$t_x = (x - x_p) / w_p, \quad t_y = (y - y_p) / h_p$$
$$t_w = \log(w / w_p), \quad t_h = \log(h / h_p)$$

```python
def encode_boxes(reference_boxes, proposals):
    """
    Encode target boxes relative to proposals.
    
    Args:
        reference_boxes: (N, 4) ground truth boxes [x, y, w, h]
        proposals: (N, 4) proposal boxes [x, y, w, h]
        
    Returns:
        (N, 4) encoded deltas [tx, ty, tw, th]
    """
    eps = 1e-6
    
    tx = (reference_boxes[:, 0] - proposals[:, 0]) / (proposals[:, 2] + eps)
    ty = (reference_boxes[:, 1] - proposals[:, 1]) / (proposals[:, 3] + eps)
    tw = torch.log(reference_boxes[:, 2] / (proposals[:, 2] + eps))
    th = torch.log(reference_boxes[:, 3] / (proposals[:, 3] + eps))
    
    return torch.stack([tx, ty, tw, th], dim=-1)


def decode_boxes(deltas, proposals):
    """
    Decode predicted deltas to boxes.
    
    Args:
        deltas: (N, 4) predicted deltas [tx, ty, tw, th]
        proposals: (N, 4) proposal boxes [x, y, w, h]
        
    Returns:
        (N, 4) decoded boxes [x, y, w, h]
    """
    x = deltas[:, 0] * proposals[:, 2] + proposals[:, 0]
    y = deltas[:, 1] * proposals[:, 3] + proposals[:, 1]
    w = torch.exp(deltas[:, 2]) * proposals[:, 2]
    h = torch.exp(deltas[:, 3]) * proposals[:, 3]
    
    return torch.stack([x, y, w, h], dim=-1)
```

## Faster R-CNN (2015)

### Key Innovation: Region Proposal Network

Faster R-CNN replaces Selective Search with a learnable **Region Proposal Network (RPN)**, enabling fully end-to-end training.

### Anchor Generator

```python
from typing import Tuple
import torch


class AnchorGenerator:
    """
    Generate anchor boxes at each spatial location.
    """
    def __init__(
        self,
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0)
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(sizes) * len(aspect_ratios)
    
    def generate_anchors(
        self,
        feature_map_size: Tuple[int, int],
        stride: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate anchors for a feature map.
        
        Args:
            feature_map_size: (H, W) of feature map
            stride: Spatial stride from input to feature map
            device: Target device
            
        Returns:
            (H*W*num_anchors, 4) anchors in [x1, y1, x2, y2] format
        """
        H, W = feature_map_size
        
        # Base anchors at origin
        base_anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                w = size * (ratio ** 0.5)
                h = size / (ratio ** 0.5)
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        base_anchors = torch.tensor(base_anchors, device=device)
        
        # Generate grid of anchor centers
        shifts_x = (torch.arange(W, device=device) + 0.5) * stride
        shifts_y = (torch.arange(H, device=device) + 0.5) * stride
        
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([
            shifts_x.flatten(),
            shifts_y.flatten(),
            shifts_x.flatten(),
            shifts_y.flatten()
        ], dim=1)
        
        # Broadcast anchors to all positions
        anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
        
        return anchors.reshape(-1, 4)
```

### Region Proposal Network

```python
import torch.nn as nn
import torch.nn.functional as F


class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network for Faster R-CNN.
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 9,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_anchors = num_anchors
        
        # Shared convolutional layer
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # Classification head: objectness score (object vs background)
        self.cls_head = nn.Conv2d(hidden_dim, num_anchors * 2, kernel_size=1)
        
        # Regression head: box deltas
        self.reg_head = nn.Conv2d(hidden_dim, num_anchors * 4, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.conv, self.cls_head, self.reg_head]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        """
        Forward pass of RPN.
        
        Args:
            features: (N, C, H, W) feature map from backbone
            
        Returns:
            objectness: (N, H*W*num_anchors, 2) classification logits
            box_deltas: (N, H*W*num_anchors, 4) regression deltas
        """
        N, C, H, W = features.shape
        
        # Shared features
        x = F.relu(self.conv(features))
        
        # Classification branch
        cls_logits = self.cls_head(x)
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(N, -1, 2)
        
        # Regression branch
        box_deltas = self.reg_head(x)
        box_deltas = box_deltas.permute(0, 2, 3, 1).reshape(N, -1, 4)
        
        return cls_logits, box_deltas
```

### RPN Training

**Anchor Assignment**:
- **Positive anchors**: IoU > 0.7 with any ground truth, OR highest IoU with a ground truth
- **Negative anchors**: IoU < 0.3 with all ground truths
- **Ignored anchors**: 0.3 ≤ IoU ≤ 0.7 (not used in training)

**RPN Loss**:
$$L_{RPN} = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* L_{reg}(t_i, t_i^*)$$

```python
def rpn_loss(objectness_logits, box_deltas, labels, regression_targets, lambda_reg=1.0):
    """
    Compute RPN loss.
    
    Args:
        objectness_logits: (N, num_anchors, 2) classification logits
        box_deltas: (N, num_anchors, 4) predicted deltas
        labels: (N, num_anchors) anchor labels (1=pos, 0=neg, -1=ignore)
        regression_targets: (N, num_anchors, 4) target deltas
        
    Returns:
        cls_loss, reg_loss
    """
    # Classification loss (only for labeled anchors)
    valid_mask = labels >= 0
    cls_loss = F.cross_entropy(
        objectness_logits[valid_mask],
        labels[valid_mask].long(),
        reduction='mean'
    )
    
    # Regression loss (only for positive anchors)
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(
            box_deltas[pos_mask],
            regression_targets[pos_mask],
            reduction='mean'
        )
    else:
        reg_loss = torch.tensor(0.0, device=box_deltas.device)
    
    return cls_loss, lambda_reg * reg_loss
```

### RoI Align

RoI Align uses bilinear interpolation instead of quantization, providing smoother gradients:

```python
from torchvision.ops import roi_align


class RoIAlignLayer(nn.Module):
    """
    RoI Align provides better spatial precision than RoI Pooling.
    
    Key difference: Uses bilinear interpolation instead of rounding
    coordinates.
    """
    def __init__(self, output_size=(7, 7), spatial_scale=1/16, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
    
    def forward(self, features, rois):
        return roi_align(
            features,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio
        )
```

## Mask R-CNN (2017)

Mask R-CNN extends Faster R-CNN by adding a parallel branch for instance segmentation:

### Mask Head

```python
class MaskHead(nn.Module):
    """
    Mask prediction head for Mask R-CNN.
    """
    def __init__(self, in_channels=256, num_classes=80, mask_size=28):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Upsample 2×
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        
        # Per-class mask prediction
        self.mask_pred = nn.Conv2d(256, num_classes, 1)
        
        self.mask_size = mask_size
    
    def forward(self, roi_features):
        """
        Args:
            roi_features: (N, C, H, W) features from RoI Align
            
        Returns:
            (N, num_classes, mask_size, mask_size) mask logits
        """
        x = F.relu(self.conv1(roi_features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))
        return self.mask_pred(x)
```

### Mask Loss

The mask loss is binary cross-entropy applied only to the ground truth class:

```python
def mask_loss(mask_logits, mask_targets, labels):
    """
    Compute mask loss.
    
    Args:
        mask_logits: (N, num_classes, H, W) predicted mask logits
        mask_targets: (N, H, W) binary ground truth masks
        labels: (N,) class labels for each RoI
    """
    N = mask_logits.shape[0]
    
    # Select masks for ground truth classes
    selected_masks = mask_logits[torch.arange(N), labels]
    
    return F.binary_cross_entropy_with_logits(
        selected_masks,
        mask_targets.float()
    )
```

## Using torchvision Implementations

```python
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn_v2
)


def create_faster_rcnn(num_classes, pretrained=True):
    """Create a Faster R-CNN model."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights='DEFAULT' if pretrained else None
    )
    
    # Replace classification head for custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    return model


def create_mask_rcnn(num_classes, pretrained=True):
    """Create a Mask R-CNN model."""
    model = maskrcnn_resnet50_fpn_v2(
        weights='DEFAULT' if pretrained else None
    )
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    
    return model


# Training example
def train_step(model, images, targets, optimizer):
    """Single training step."""
    model.train()
    
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    
    return loss_dict


# Inference example
def inference(model, image):
    """Run inference on a single image."""
    model.eval()
    
    with torch.no_grad():
        predictions = model([image])
    
    return predictions[0]
```

## Evolution Summary

| Model | Year | Region Proposals | Speed | Key Innovation |
|-------|------|------------------|-------|----------------|
| **R-CNN** | 2014 | Selective Search | ~47s/img | CNN features for detection |
| **Fast R-CNN** | 2015 | Selective Search | ~2s/img | Shared computation, multi-task |
| **Faster R-CNN** | 2015 | RPN (learned) | ~0.2s/img | End-to-end training |
| **Mask R-CNN** | 2017 | RPN | ~0.2s/img | Instance segmentation |

## Performance Comparison (COCO test-dev)

| Model | Backbone | AP | AP50 | AP75 |
|-------|----------|-----|------|------|
| Faster R-CNN | ResNet-50-FPN | 37.4 | 58.1 | 40.4 |
| Faster R-CNN | ResNet-101-FPN | 39.4 | 61.2 | 43.4 |
| Mask R-CNN | ResNet-50-FPN | 37.9 | 59.2 | 41.0 |
| Mask R-CNN | ResNet-101-FPN | 40.1 | 61.6 | 44.0 |

## Summary

The R-CNN family established the two-stage detection paradigm:

1. **R-CNN** pioneered using CNNs for detection but was slow
2. **Fast R-CNN** introduced shared computation and multi-task training
3. **Faster R-CNN** replaced external proposals with a learned RPN
4. **Mask R-CNN** extended to instance segmentation

Key components:
- **Region Proposal Network**: Generates class-agnostic proposals
- **RoI Pooling/Align**: Extracts fixed-size features from variable-size regions
- **Multi-task Loss**: Jointly optimizes classification and localization
- **Anchor Boxes**: Predefined reference boxes at multiple scales

Two-stage detectors remain the gold standard for accuracy, though one-stage detectors are preferred when speed is critical.

## References

1. Girshick, R., et al. (2014). Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. *CVPR*.
2. Girshick, R. (2015). Fast R-CNN. *ICCV*.
3. Ren, S., et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*.
4. He, K., et al. (2017). Mask R-CNN. *ICCV*.
5. Lin, T.Y., et al. (2017). Feature Pyramid Networks for Object Detection. *CVPR*.
