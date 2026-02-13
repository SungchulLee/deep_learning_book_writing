# ResNeXt

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by ResNeXt (2017)
- Identify how ResNeXt influenced subsequent architecture design

## Overview

**Year**: 2017 | **Parameters**: 25M (ResNeXt-50) | **Key Innovation**: Grouped convolutions with cardinality dimension

ResNeXt (Xie et al., 2017) extends ResNet by introducing a **cardinality** dimension—the number of parallel transformation paths within each block. This provides a more effective way to increase capacity than simply making networks deeper or wider.

## Aggregated Transformations

A ResNeXt block splits the input into $C$ groups, applies transformations independently, and aggregates:

$$\mathbf{y} = \mathbf{x} + \sum_{i=1}^{C} \mathcal{T}_i(\mathbf{x})$$

In practice, this is implemented efficiently using **grouped convolutions**:

```python
import torch.nn as nn

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_width, cardinality=32, stride=1):
        super().__init__()
        group_width = bottleneck_width * cardinality
        self.conv1 = nn.Conv2d(in_channels, group_width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, 3, stride=stride, 
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, in_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)
```

```python
import torchvision.models as models
model = models.resnext50_32x4d(weights='DEFAULT')  # 32 groups, 4d width
```

Increasing cardinality is more efficient than increasing depth or width for the same parameter budget.

## References

1. Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated Residual Transformations for Deep Neural Networks. CVPR.
