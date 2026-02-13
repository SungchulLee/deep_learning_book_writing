# Classification Overview

## Learning Objectives

By the end of this section, you will be able to:

- Understand image classification as the foundational computer vision task
- Trace the evolution of CNN architectures from LeNet to modern designs
- Explain the standard classification pipeline: backbone → pooling → classifier
- Identify the key innovations that drove architecture improvements
- Select appropriate architectures for different deployment scenarios

## The Image Classification Task

Given an input image $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$, image classification assigns a label $y \in \{1, \ldots, K\}$ from $K$ predefined categories. The model outputs a probability distribution:

$$\hat{\mathbf{y}} = \text{softmax}(f_\theta(\mathbf{X})) \in [0, 1]^K, \quad \sum_k \hat{y}_k = 1$$

## Architecture Evolution

The history of image classification architectures maps a clear trajectory of increasing depth, efficiency, and design sophistication:

| Era | Architecture | Year | Top-5 Error (ImageNet) | Key Innovation |
|-----|-------------|------|----------------------|----------------|
| Pioneer | LeNet-5 | 1998 | — | CNN for digits |
| Breakthrough | AlexNet | 2012 | 16.4% | Deep CNN + GPU + ReLU + Dropout |
| Depth | VGGNet | 2014 | 7.3% | Small (3×3) filters, very deep |
| Width | GoogLeNet | 2014 | 6.7% | Inception modules, multi-scale |
| Residual | ResNet | 2015 | 3.6% | Skip connections, 152 layers |
| Dense | DenseNet | 2017 | — | Feature reuse via concatenation |
| Efficient | MobileNet | 2017 | — | Depthwise separable convolutions |
| Scaling | EfficientNet | 2019 | 2.9% | Compound scaling |
| Modern | ConvNeXt | 2022 | — | Modernized ConvNet matching ViT |

## Standard Pipeline

```
Input Image (224×224×3)
       │
       ▼
┌──────────────┐
│   Backbone   │  Hierarchical feature extraction
│  (ConvNet)   │  Progressively: edges → textures → parts → objects
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Global Pool  │  Spatial dimensions → single vector
│ (Avg or Max) │  (C, H, W) → (C,)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Classifier  │  Linear layer(s) + softmax
│   (FC)       │  (C,) → (K,) class probabilities
└──────────────┘
```

## Common Training Recipe

Modern classification training typically uses:

- **Optimizer**: AdamW or SGD with momentum
- **Learning rate**: Cosine annealing with warmup
- **Augmentation**: RandAugment, MixUp, CutMix, random erasing
- **Regularization**: Label smoothing (0.1), stochastic depth, weight decay
- **Resolution**: Train at lower resolution, fine-tune at higher

```python
import torch
import torchvision.models as models

# Load any pre-trained model
model = models.resnet50(weights='DEFAULT')

# Adapt for custom number of classes
num_classes = 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Standard training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
```

## Choosing an Architecture

| Priority | Recommended | Why |
|----------|------------|-----|
| Maximum accuracy | ConvNeXt-L, EfficientNet-B7 | Best accuracy/compute |
| Mobile deployment | MobileNetV3, ShuffleNetV2 | Low FLOPs, low latency |
| Transfer learning | ResNet-50, ConvNeXt-T | Strong pre-trained features |
| Medical/scientific | ResNet-50, EfficientNet-B4 | Well-validated, interpretable |
| Real-time edge | MobileNetV3-Small, ShuffleNet | Optimized for mobile hardware |
| Research baseline | ResNet-50 | Universal benchmark |

## Summary

Image classification architectures serve as backbones for virtually all vision tasks—detection, segmentation, video understanding, and multimodal models. Understanding their design principles (depth, width, skip connections, efficient operations) provides the foundation for the entire chapter.

## References

1. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
4. Liu, Z., et al. (2022). A ConvNet for the 2020s. CVPR.
