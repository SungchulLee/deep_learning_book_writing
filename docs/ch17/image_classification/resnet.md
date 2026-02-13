# ResNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by ResNet (2015)
- Identify how ResNet influenced subsequent architecture design

## Overview

**Year**: 2015 | **Parameters**: 25.6M (ResNet-50) | **Key Innovation**: Residual connections enabling 100+ layer networks

ResNet (He et al., 2016) solved the degradation problem—the observation that simply adding more layers to a network eventually hurts accuracy due to optimization difficulty. The solution: **residual connections** that let layers learn residual functions.

## The Residual Learning Framework

Instead of learning a mapping $\mathcal{H}(\mathbf{x})$ directly, each block learns the residual:

$$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$$

The output is $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$, creating a skip (shortcut) connection.

### Why It Works

1. **Identity mapping is easy**: If extra layers are unnecessary, $\mathcal{F}(\mathbf{x}) = 0$ is easy to learn
2. **Gradient flow**: Gradients flow directly through skip connections, preventing vanishing gradients
3. **Feature reuse**: Earlier representations are preserved and combined with learned refinements

## Architecture

```python
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet basic block (for ResNet-18/34)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)
```

## Variants

| Model | Depth | Params | Top-5 Error | Notes |
|-------|-------|--------|-------------|-------|
| ResNet-18 | 18 | 11.7M | 10.9% | Lightweight |
| ResNet-50 | 50 | 25.6M | 6.7% | Most common backbone |
| ResNet-101 | 101 | 44.5M | 6.0% | Higher accuracy |
| ResNet-152 | 152 | 60.2M | 5.7% | Highest single-model |

```python
import torchvision.models as models
model = models.resnet50(weights='DEFAULT')  # Universal backbone
```

ResNet remains the most widely used backbone in computer vision. Its skip connection principle appears in virtually every modern architecture.

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. He, K., et al. (2016). Identity Mappings in Deep Residual Networks. ECCV.
