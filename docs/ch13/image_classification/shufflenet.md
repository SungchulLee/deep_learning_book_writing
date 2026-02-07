# ShuffleNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by ShuffleNet (2018)
- Identify how ShuffleNet influenced subsequent architecture design

## Overview

**Year**: 2018 | **Parameters**: ~1.4M | **Key Innovation**: Channel shuffle for cross-group information flow

ShuffleNet (Zhang et al., 2018) addresses the information isolation problem in grouped convolutions by introducing **channel shuffle**—a zero-parameter operation that mixes channels across groups.

## Channel Shuffle

Grouped convolutions are efficient but each group operates independently, limiting inter-group information flow. Channel shuffle rearranges channels between groups:

```python
def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)
```

```python
import torchvision.models as models
model = models.shufflenet_v2_x1_0(weights='DEFAULT')
```

ShuffleNetV2 refined the design with guidelines based on actual hardware latency (not just FLOPs), including equal channel widths and minimal fragmentation.

## References

1. Zhang, X., Zhou, X., Lin, M., & Sun, J. (2018). ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices. CVPR.
2. Ma, N., et al. (2018). ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design. ECCV.
