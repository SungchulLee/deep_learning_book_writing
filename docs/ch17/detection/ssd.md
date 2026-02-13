# SSD: Single Shot MultiBox Detector

## Learning Objectives

By the end of this section, you will be able to:

- Explain the SSD architecture and its multi-scale detection strategy
- Understand default boxes (anchor priors) and their role in SSD
- Implement the SSD detection head with multiple feature maps
- Describe the hard negative mining strategy and its importance
- Compare SSD with YOLO and understand their tradeoffs

## Overview

SSD (Single Shot MultiBox Detector) achieves a balance between the speed of YOLO and the accuracy of region-based methods by:

1. Eliminating the proposal generation step (single-shot)
2. Using **multi-scale feature maps** for detection
3. Applying **default boxes** (anchor priors) at each location

## Multi-Scale Feature Maps

SSD's key innovation is detecting objects at multiple scales using progressively smaller feature maps:

| Feature Map | Size | Receptive Field | Typical Objects |
|-------------|------|-----------------|-----------------|
| Conv4_3 | 38×38 | Small | Small objects |
| Conv7 | 19×19 | Medium | Medium objects |
| Conv8_2 | 10×10 | Large | Large objects |
| Conv9_2 | 5×5 | Very Large | Very large objects |
| Conv10_2 | 3×3 | Huge | Largest objects |
| Conv11_2 | 1×1 | Global | Scene-level |

**Why Multi-Scale?**

- **Small feature maps** have large receptive fields → detect large objects
- **Large feature maps** preserve spatial detail → detect small objects
- Different scales share the same backbone → efficient computation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SSDExtraLayers(nn.Module):
    """
    Extra convolutional layers after VGG base for multi-scale detection.
    """
    def __init__(self, in_channels: int = 1024):
        super().__init__()
        
        # Conv8: 19×19 → 10×10
        self.conv8_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Conv9: 10×10 → 5×5
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Conv10: 5×5 → 3×3
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        
        # Conv11: 3×3 → 1×1
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        x = F.relu(self.conv8_1(x), inplace=True)
        x = F.relu(self.conv8_2(x), inplace=True)
        features.append(x)
        
        x = F.relu(self.conv9_1(x), inplace=True)
        x = F.relu(self.conv9_2(x), inplace=True)
        features.append(x)
        
        x = F.relu(self.conv10_1(x), inplace=True)
        x = F.relu(self.conv10_2(x), inplace=True)
        features.append(x)
        
        x = F.relu(self.conv11_1(x), inplace=True)
        x = F.relu(self.conv11_2(x), inplace=True)
        features.append(x)
        
        return features
```

## Default Boxes (Anchor Priors)

At each location in each feature map, SSD places a set of **default boxes** with different aspect ratios and scales.

### Default Box Configuration

For each feature map of size $f_k$:

**Scale**:
$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m - 1}(k - 1), \quad k \in [1, m]$$

where $s_{min} = 0.2$ and $s_{max} = 0.9$.

**Aspect Ratios**: ${1, 2, 3, 1/2, 1/3}$ plus an additional box with scale $\sqrt{s_k \cdot s_{k+1}}$.

```python
class DefaultBoxGenerator:
    """Generate SSD default boxes."""
    def __init__(
        self,
        image_size: int = 300,
        feature_maps: List[int] = [38, 19, 10, 5, 3, 1],
        aspect_ratios: List[List[float]] = None
    ):
        self.image_size = image_size
        self.feature_maps = feature_maps
        
        if aspect_ratios is None:
            self.aspect_ratios = [
                [1, 2, 1/2],
                [1, 2, 3, 1/2, 1/3],
                [1, 2, 3, 1/2, 1/3],
                [1, 2, 3, 1/2, 1/3],
                [1, 2, 1/2],
                [1, 2, 1/2],
            ]
        else:
            self.aspect_ratios = aspect_ratios
        
        # Calculate scales
        m = len(feature_maps)
        self.scales = [0.2 + (0.9 - 0.2) * k / (m - 1) for k in range(m)]
        self.scales.append(1.0)
    
    def generate(self, device: torch.device) -> torch.Tensor:
        default_boxes = []
        
        for k, f_k in enumerate(self.feature_maps):
            s_k = self.scales[k]
            s_k_prime = (self.scales[k] * self.scales[k + 1]) ** 0.5
            
            for i in range(f_k):
                for j in range(f_k):
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k
                    
                    default_boxes.append([cx, cy, s_k, s_k])
                    default_boxes.append([cx, cy, s_k_prime, s_k_prime])
                    
                    for ar in self.aspect_ratios[k]:
                        if ar != 1:
                            w = s_k * (ar ** 0.5)
                            h = s_k / (ar ** 0.5)
                            default_boxes.append([cx, cy, w, h])
        
        return torch.tensor(default_boxes, device=device).clamp(0, 1)
```

## Detection Head

```python
class SSDHead(nn.Module):
    """SSD detection head for a single feature map."""
    def __init__(self, in_channels: int, num_boxes: int, num_classes: int):
        super().__init__()
        
        self.loc_conv = nn.Conv2d(in_channels, num_boxes * 4, 3, padding=1)
        self.conf_conv = nn.Conv2d(in_channels, num_boxes * num_classes, 3, padding=1)
        self.num_classes = num_classes
        self.num_boxes = num_boxes
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        loc = self.loc_conv(x).permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        conf = self.conf_conv(x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
        
        return loc, conf
```

## SSD Loss Function

SSD uses a multi-task loss with hard negative mining:

$$L = \frac{1}{N}(L_{conf} + \alpha L_{loc})$$

### Hard Negative Mining

The ratio of negatives to positives is kept at 3:1 by selecting the hardest negatives (highest loss).

```python
class SSDLoss(nn.Module):
    def __init__(self, num_classes: int, neg_pos_ratio: float = 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
    
    def forward(self, loc_pred, conf_pred, loc_target, conf_target):
        pos_mask = conf_target > 0
        num_pos = pos_mask.sum(dim=1)
        
        # Localization loss
        loc_loss = F.smooth_l1_loss(
            loc_pred[pos_mask],
            loc_target[pos_mask],
            reduction='sum'
        )
        
        # Confidence loss with hard negative mining
        conf_loss_all = F.cross_entropy(
            conf_pred.view(-1, self.num_classes),
            conf_target.view(-1),
            reduction='none'
        ).view(conf_pred.size(0), -1)
        
        # Hard negative mining
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[pos_mask] = 0
        
        _, loss_idx = conf_loss_neg.sort(dim=1, descending=True)
        _, idx_rank = loss_idx.sort(dim=1)
        
        num_neg = torch.clamp(num_pos * self.neg_pos_ratio, max=conf_pred.size(1) - 1).long()
        neg_mask = idx_rank < num_neg.unsqueeze(1)
        
        conf_loss = (conf_loss_all * (pos_mask.float() + neg_mask.float())).sum()
        
        N = max(num_pos.sum().item(), 1)
        return loc_loss / N, conf_loss / N
```

## Using Pre-trained SSD

```python
import torchvision
from torchvision.models.detection import ssd300_vgg16

model = ssd300_vgg16(weights='DEFAULT')
model.eval()

with torch.no_grad():
    predictions = model([torch.rand(3, 300, 300)])

boxes = predictions[0]['boxes']
scores = predictions[0]['scores']
labels = predictions[0]['labels']
```

## SSD vs YOLO Comparison

| Aspect | SSD | YOLO |
|--------|-----|------|
| **Multi-scale** | Yes (6 scales) | v3+: Yes |
| **Speed** | ~46 FPS | ~45 FPS |
| **Small objects** | Better | Challenging |
| **Architecture** | VGG-16 | Darknet |

## Summary

SSD introduced key innovations for single-shot detection:

1. **Multi-scale feature maps**: Detect objects at different scales efficiently
2. **Default boxes**: Pre-defined anchor priors at each location
3. **Hard negative mining**: Handle class imbalance
4. **End-to-end training**: Single network for all predictions

SSD achieves a good balance between speed and accuracy, making it suitable for real-time applications.

## References

1. Liu, W., et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV*.
2. Fu, C.Y., et al. (2017). DSSD: Deconvolutional Single Shot Detector. *arXiv*.
