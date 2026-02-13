# GoogLeNet/Inception

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by GoogLeNet/Inception (2014)
- Identify how GoogLeNet/Inception influenced subsequent architecture design

## Overview

**Year**: 2014 | **Parameters**: 6.8M | **Key Innovation**: Inception modules, multi-scale processing

GoogLeNet (Szegedy et al., 2015) introduced the Inception module—a block that processes input at multiple scales simultaneously and concatenates the results. This achieved strong accuracy with far fewer parameters than VGG.

## The Inception Module

The key idea is to apply multiple convolution sizes in parallel and let the network learn which scale is most informative:

```
Input
  ├── 1×1 Conv ──────────────┐
  ├── 1×1 Conv → 3×3 Conv ──┤
  ├── 1×1 Conv → 5×5 Conv ──┤  → Concatenate
  └── 3×3 MaxPool → 1×1 Conv┘
```

The 1×1 convolutions serve as bottleneck layers, reducing channel dimensionality before the expensive 3×3 and 5×5 operations.

## Evolution

| Version | Year | Key Addition |
|---------|------|-------------|
| Inception v1 (GoogLeNet) | 2014 | Inception module |
| Inception v2 | 2015 | Batch normalization |
| Inception v3 | 2015 | Factorized convolutions (e.g., 1×7 + 7×1) |
| Inception v4 / Inception-ResNet | 2016 | Residual connections |

```python
import torchvision.models as models
model = models.inception_v3(weights='DEFAULT')  # Requires 299×299 input
```

Inception demonstrated that carefully designed multi-scale processing could match deeper networks with far fewer parameters.

## References

1. Szegedy, C., et al. (2015). Going Deeper with Convolutions. CVPR.
2. Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR.
