# Faster R-CNN

## Learning Objectives

By the end of this section, you will be able to:

- Understand how the Region Proposal Network replaces external proposal methods
- Explain anchor-based detection and the anchor assignment strategy
- Implement the complete Faster R-CNN training pipeline with torchvision
- Describe RoI Align as an improvement over RoI Pooling
- Extend Faster R-CNN to instance segmentation (Mask R-CNN)

## Key Innovation: Region Proposal Network

Faster R-CNN (Ren et al., 2015) eliminates the Selective Search bottleneck by introducing a **Region Proposal Network (RPN)**—a small neural network that shares the backbone features and generates proposals in a single forward pass. This enables fully end-to-end training and reduces proposal generation from ~2 seconds to ~10 milliseconds.

## Anchor Boxes

The RPN operates over a set of predefined reference boxes called **anchors**, placed at every spatial location in the feature map. Each anchor is defined by a scale and aspect ratio:

```python
import torch

class AnchorGenerator:
    """Generate anchor boxes at each spatial location."""
    
    def __init__(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(sizes) * len(aspect_ratios)
    
    def generate(self, feature_map_size, stride, device):
        """
        Args:
            feature_map_size: (H, W) of feature map
            stride: Spatial stride from input to feature map
        
        Returns:
            (H*W*num_anchors, 4) anchors in xyxy format
        """
        H, W = feature_map_size
        
        base_anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                w = size * (ratio ** 0.5)
                h = size / (ratio ** 0.5)
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        base_anchors = torch.tensor(base_anchors, device=device)
        
        shifts_x = (torch.arange(W, device=device) + 0.5) * stride
        shifts_y = (torch.arange(H, device=device) + 0.5) * stride
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=-1).reshape(-1, 4)
        
        anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
        return anchors.reshape(-1, 4)
```

## Region Proposal Network Architecture

The RPN slides a small network over the shared feature map. At each position, it predicts objectness scores and box offsets for each anchor:

```python
import torch.nn as nn
import torch.nn.functional as F

class RegionProposalNetwork(nn.Module):
    """
    RPN: predicts objectness and box deltas for each anchor.
    """
    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(256, num_anchors * 2, kernel_size=1)  # object vs background
        self.reg_head = nn.Conv2d(256, num_anchors * 4, kernel_size=1)  # box deltas
        
        for layer in [self.conv, self.cls_head, self.reg_head]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        N, C, H, W = features.shape
        x = F.relu(self.conv(features))
        
        cls_logits = self.cls_head(x).permute(0, 2, 3, 1).reshape(N, -1, 2)
        box_deltas = self.reg_head(x).permute(0, 2, 3, 1).reshape(N, -1, 4)
        
        return cls_logits, box_deltas
```

### Anchor Assignment for Training

- **Positive anchors**: IoU > 0.7 with any ground truth box, OR the anchor with highest IoU for each ground truth
- **Negative anchors**: IoU < 0.3 with all ground truth boxes
- **Ignored anchors**: IoU between 0.3 and 0.7 (excluded from loss computation)

Training samples 256 anchors per image (128 positive, 128 negative) for balanced mini-batches.

### RPN Loss

$$\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}}\sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}}\sum_i p_i^* \mathcal{L}_{\text{reg}}(t_i, t_i^*)$$

where $p_i^*$ is 1 for positive anchors (regression loss only applies to positives).

## RoI Align

RoI Align (introduced in Mask R-CNN) replaces RoI Pooling's quantization with bilinear interpolation, providing exact spatial alignment:

```python
from torchvision.ops import roi_align

class RoIAlignLayer(nn.Module):
    def __init__(self, output_size=(7, 7), spatial_scale=1/16, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
    
    def forward(self, features, rois):
        return roi_align(features, rois,
                         output_size=self.output_size,
                         spatial_scale=self.spatial_scale,
                         sampling_ratio=self.sampling_ratio)
```

## Using torchvision Implementations

### Training

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_faster_rcnn(num_classes, pretrained=True):
    """Create Faster R-CNN with custom number of classes."""
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT' if pretrained else None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Training loop
model = create_faster_rcnn(num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

model.train()
for images, targets in train_loader:
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
```

### Inference

```python
model.eval()
with torch.no_grad():
    predictions = model([image])[0]
    
# predictions contains: boxes, labels, scores
boxes = predictions['boxes']    # (N, 4) xyxy format
labels = predictions['labels']  # (N,)
scores = predictions['scores']  # (N,)
```

## Evolution Summary

| Model | Year | Proposals | Speed | Key Innovation |
|-------|------|-----------|-------|----------------|
| R-CNN | 2014 | Selective Search | ~47s/img | CNN features for detection |
| Fast R-CNN | 2015 | Selective Search | ~2s/img | Shared computation, multi-task loss |
| Faster R-CNN | 2015 | RPN (learned) | ~0.2s/img | End-to-end, learned proposals |
| Mask R-CNN | 2017 | RPN | ~0.2s/img | + Instance segmentation |

## Performance (COCO test-dev)

| Model | Backbone | AP | AP50 | AP75 |
|-------|----------|-----|------|------|
| Faster R-CNN | ResNet-50-FPN | 37.4 | 58.1 | 40.4 |
| Faster R-CNN | ResNet-101-FPN | 39.4 | 61.2 | 43.4 |

## Summary

Faster R-CNN unified proposal generation and detection into a single trainable network:

1. **RPN** generates proposals using shared backbone features
2. **Anchor boxes** at multiple scales/ratios provide initial references
3. **Two-stage design** enables high accuracy through focused computation
4. **RoI Align** provides precise spatial feature extraction

Faster R-CNN remains the standard two-stage detector and the foundation for Mask R-CNN (instance segmentation), Cascade R-CNN (iterative refinement), and numerous other extensions.

## References

1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS.
2. Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR.
3. He, K., et al. (2017). Mask R-CNN. ICCV.
