# RetinaNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand RetinaNet's architecture combining FPN with focal loss
- Explain why one-stage detectors historically lagged behind two-stage in accuracy
- Describe the parallel classification and regression subnet design

## Architecture

RetinaNet (Lin et al., 2017) is a one-stage detector that achieves two-stage accuracy by addressing class imbalance with **focal loss** rather than sampling heuristics.

### Components

1. **Backbone**: ResNet (typically ResNet-50 or ResNet-101)
2. **Neck**: Feature Pyramid Network (FPN) producing multi-scale feature maps at strides {8, 16, 32, 64, 128}
3. **Classification Subnet**: 4 × Conv3×3 + Conv for $K \times A$ class scores per position ($K$ classes, $A$ anchors)
4. **Box Regression Subnet**: 4 × Conv3×3 + Conv for $4 \times A$ box offsets per position

Both subnets are independent FCNs applied at each FPN level. Parameters are shared across levels but **not** between the two subnets.

### Subnet Design

```python
import torch.nn as nn

class RetinaNetSubnet(nn.Module):
    """Shared subnet design for classification and regression."""
    def __init__(self, in_channels=256, num_outputs_per_anchor=80, 
                 num_anchors=9, num_convs=4):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.convs = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_channels, num_outputs_per_anchor * num_anchors, 3, padding=1)
    
    def forward(self, features):
        return self.head(self.convs(features))
```

### Initialization

The classification subnet's final bias is initialized to $b = -\log((1-\pi)/\pi)$ where $\pi = 0.01$, ensuring initial predictions strongly favor background. This prevents the massive number of background anchors from generating large, destabilizing loss values early in training.

## Key Results

| Detector | Type | COCO AP | Speed |
|----------|------|---------|-------|
| Faster R-CNN (ResNet-101-FPN) | Two-stage | 36.2 | ~5 FPS |
| RetinaNet (ResNet-101-FPN) | One-stage | 39.1 | ~5 FPS |
| RetinaNet (ResNeXt-101-FPN) | One-stage | 40.8 | ~3 FPS |

RetinaNet was the first one-stage detector to surpass all two-stage methods, demonstrating that the accuracy gap was due to class imbalance, not architectural limitations.

## Summary

RetinaNet's contribution is more conceptual than architectural:

1. **Simple one-stage design** with FPN + parallel subnets
2. **Focal loss** eliminates the need for complex sampling strategies
3. **Proved** that class imbalance—not architecture—was the bottleneck for one-stage detectors

## References

1. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
2. Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR.
