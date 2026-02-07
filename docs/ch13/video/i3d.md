# I3D: Inflated 3D ConvNets

## Learning Objectives

By the end of this section, you will be able to:

- Understand the "inflation" technique for converting 2D ImageNet models to 3D video models
- Explain how I3D leverages ImageNet pre-training for video understanding
- Compare I3D with C3D and two-stream approaches

## Inflation: 2D to 3D

I3D (Carreira & Zisserman, 2017) converts successful 2D image classification architectures (Inception, ResNet) to 3D by "inflating" all 2D convolution filters into 3D:

$$\text{2D filter: } K \times K \times C_{in} \times C_{out} \quad \rightarrow \quad \text{3D filter: } T \times K \times K \times C_{in} \times C_{out}$$

### Weight Initialization

Pre-trained 2D weights are repeated along the temporal dimension and divided by $T$:

$$W_{3D}[t, :, :, :, :] = \frac{1}{T} W_{2D}[:, :, :, :]$$

This preserves the 2D model's response for static inputs while enabling temporal learning.

```python
import torch
import torch.nn as nn

def inflate_conv(conv2d, temporal_kernel_size=3):
    """Convert a 2D convolution to 3D by inflating weights."""
    conv3d = nn.Conv3d(
        conv2d.in_channels, conv2d.out_channels,
        kernel_size=(temporal_kernel_size, *conv2d.kernel_size),
        stride=(1, *conv2d.stride),
        padding=(temporal_kernel_size // 2, *conv2d.padding),
        bias=conv2d.bias is not None
    )
    
    # Repeat and normalize 2D weights along temporal dimension
    weight_2d = conv2d.weight.data
    weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, temporal_kernel_size, 1, 1)
    weight_3d /= temporal_kernel_size
    
    conv3d.weight.data = weight_3d
    if conv2d.bias is not None:
        conv3d.bias.data = conv2d.bias.data
    
    return conv3d
```

## Two-Stream I3D

I3D can operate on both RGB and optical flow inputs using separate streams, combining their predictions:

| Stream | Input | Captures |
|--------|-------|----------|
| RGB | Raw video frames | Appearance + implicit motion |
| Flow | Pre-computed optical flow | Explicit motion patterns |

The two-stream variant improves accuracy by ~3–5% over single-stream on Kinetics-400.

## Results

| Model | Kinetics-400 Top-1 | Pre-training |
|-------|-------------------|-------------|
| C3D | 61.1% | Sports-1M |
| I3D (RGB) | 71.1% | ImageNet → Kinetics |
| I3D (Two-stream) | 74.2% | ImageNet → Kinetics |

## Summary

I3D's key contribution is demonstrating that ImageNet pre-training can be effectively transferred to video models through weight inflation, providing a strong initialization that dramatically improves convergence and accuracy.

## References

1. Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. CVPR.
