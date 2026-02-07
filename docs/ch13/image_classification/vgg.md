# VGGNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by VGGNet (2014)
- Identify how VGGNet influenced subsequent architecture design

## Overview

**Year**: 2014 | **Parameters**: 138M | **Key Innovation**: Uniform 3×3 convolutions, very deep networks

VGGNet (Simonyan & Zisserman, 2014) demonstrated that network depth is critical for performance. By using exclusively small 3×3 convolution filters stacked in deep sequences, VGG achieved strong results with a simple, uniform architecture.

## Key Insight: Small Filters, More Depth

Two stacked 3×3 convolutions have the same receptive field as one 5×5 convolution, but with fewer parameters and more non-linearities:

$$\text{Two 3×3: } 2 \times (3^2 C^2) = 18C^2 \text{ params}$$
$$\text{One 5×5: } 5^2 C^2 = 25C^2 \text{ params}$$

Three stacked 3×3 convolutions equal one 7×7 with even greater savings.

## Architecture Variants

| Model | Depth | Parameters | Top-5 Error |
|-------|-------|-----------|-------------|
| VGG-11 | 11 layers | 133M | 10.4% |
| VGG-16 | 16 layers | 138M | 7.4% |
| VGG-19 | 19 layers | 144M | 7.3% |

```python
import torchvision.models as models

model = models.vgg16(weights='DEFAULT')
# Very commonly used as backbone for detection (Faster R-CNN)
# and segmentation (FCN) via transfer learning
```

VGG's uniform design made it the default backbone for many downstream tasks (FCN, Faster R-CNN). However, its 138M parameters and high memory usage led to the development of more efficient architectures.

## References

1. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
