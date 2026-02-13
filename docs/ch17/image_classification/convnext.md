# ConvNeXt

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by ConvNeXt (2022)
- Identify how ConvNeXt influenced subsequent architecture design

## Overview

**Year**: 2022 | **Parameters**: 28.6M (Tiny) | **Key Innovation**: Modernized ConvNet matching Vision Transformer performance

ConvNeXt (Liu et al., 2022) systematically modernized the standard ResNet by incorporating design decisions from Vision Transformers, proving that pure ConvNets can match or exceed ViT performance.

## Modernization Steps

Starting from ResNet-50, ConvNeXt applies incremental changes:

1. **Macro design**: Adjusted stage compute ratios to (3, 3, 9, 3) matching Swin Transformer
2. **Patchify stem**: Replaced stride-4 conv + maxpool with non-overlapping 4×4 conv
3. **ResNeXt-ify**: Used depthwise convolutions (groups = channels)
4. **Inverted bottleneck**: Expanded hidden dim (like MobileNetV2)
5. **Large kernel**: 7×7 depthwise convolution
6. **Micro design**: LayerNorm, GELU, fewer activations/norms

Each change was validated independently; together they close the gap with Swin Transformer.

| Model | Params | FLOPs | ImageNet Top-1 |
|-------|--------|-------|---------------|
| ConvNeXt-T | 28.6M | 4.5G | 82.1% |
| ConvNeXt-B | 88.6M | 15.4G | 83.8% |
| ConvNeXt-L | 197.8M | 34.4G | 84.3% |

```python
import torchvision.models as models
model = models.convnext_tiny(weights='DEFAULT')
```

ConvNeXt demonstrated that the architectural innovations attributed to transformers (large receptive fields, modern normalization, training recipes) transfer effectively to pure ConvNets.

## References

1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. CVPR.
