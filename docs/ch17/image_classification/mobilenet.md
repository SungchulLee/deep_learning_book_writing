# MobileNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by MobileNet (2017)
- Identify how MobileNet influenced subsequent architecture design

## Overview

**Year**: 2017 | **Parameters**: 3.4M (MobileNetV2) | **Key Innovation**: Depthwise separable convolutions for mobile deployment

MobileNet (Howard et al., 2017) introduced **depthwise separable convolutions** to dramatically reduce computation and parameters while maintaining accuracy, enabling CNN deployment on mobile and edge devices.

## Depthwise Separable Convolutions

A standard convolution with kernel size $K$, $C_{in}$ input channels, and $C_{out}$ output channels requires:

$$\text{Standard: } K^2 \cdot C_{in} \cdot C_{out} \text{ multiplications per position}$$

Depthwise separable factorizes this into two steps:

1. **Depthwise**: One $K \times K$ filter per input channel → $K^2 \cdot C_{in}$
2. **Pointwise**: 1×1 convolution to mix channels → $C_{in} \cdot C_{out}$

$$\text{Total: } K^2 \cdot C_{in} + C_{in} \cdot C_{out}$$

$$\text{Reduction ratio: } \frac{1}{C_{out}} + \frac{1}{K^2} \approx \frac{1}{K^2}$$

For $K=3$: approximately **8–9× fewer computations**.

## MobileNetV2: Inverted Residuals

MobileNetV2 introduced the **inverted residual block** with linear bottlenecks:

```python
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch
        
        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, 
                     groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            # Pointwise projection (linear, no activation)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
```

```python
import torchvision.models as models
model = models.mobilenet_v3_large(weights='DEFAULT')
```

## References

1. Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv.
2. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
3. Howard, A., et al. (2019). Searching for MobileNetV3. ICCV.
