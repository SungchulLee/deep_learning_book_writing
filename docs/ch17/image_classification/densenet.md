# DenseNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by DenseNet (2017)
- Identify how DenseNet influenced subsequent architecture design

## Overview

**Year**: 2017 | **Parameters**: 8M (DenseNet-121) | **Key Innovation**: Dense connections for maximum feature reuse

DenseNet (Huang et al., 2017) connects each layer to every subsequent layer within a dense block. This maximizes feature reuse, strengthens gradient flow, and substantially reduces parameters.

## Dense Connectivity

In a dense block with $L$ layers, layer $\ell$ receives feature maps from all preceding layers:

$$\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{\ell-1}])$$

where $[\cdot]$ denotes channel-wise concatenation.

### Growth Rate

Each layer produces $k$ feature maps (the growth rate). After $L$ layers, a dense block has $k_0 + L \times k$ channels. Typical growth rate: $k = 32$.

```python
import torchvision.models as models
model = models.densenet121(weights='DEFAULT')  # 121 layers, very parameter-efficient
```

DenseNet demonstrated that feature reuse through concatenation is more parameter-efficient than feature refinement through addition (ResNet).

## References

1. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. CVPR.
