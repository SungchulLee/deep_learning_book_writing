# CNN Overview

## Introduction

**Convolutional Neural Networks** (CNNs) are a class of deep neural networks specifically designed to process data with spatial structure—most notably images, but also time series, audio signals, and other grid-like inputs. CNNs achieve this by replacing the dense matrix multiplications of fully connected layers with the **convolution operation**, which exploits three fundamental structural priors:

1. **Locality**: Nearby pixels are more correlated than distant ones
2. **Translation equivariance**: The same pattern can appear anywhere in the image
3. **Hierarchical composition**: Complex features are built from simpler ones

These priors enable CNNs to learn rich spatial representations with far fewer parameters than fully connected networks, making them the backbone of modern computer vision.

---

## Core Concepts at a Glance

### The Convolution Operation

At its heart, a CNN replaces dense layers with **local, shared-weight** operations. A small learnable filter (kernel) slides over the input, computing a dot product at each position to produce an output called a **feature map**:

$$Y[i, j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X[i+m, j+n] \cdot W[m, n] + b$$

where $W \in \mathbb{R}^{K \times K}$ is the kernel and $b$ is the bias. This is technically **cross-correlation**, but deep learning universally calls it "convolution."

### From Pixels to Predictions

A CNN typically consists of three stages:

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌──────────────┐
│   Input      │     │    Feature        │     │    Feature      │     │   Output      │
│   Image      │ ──▶ │    Extraction     │ ──▶ │    Aggregation  │ ──▶ │  (Classes)    │
│  (H×W×C)     │     │  (Conv + Pool)    │     │   (GAP / FC)    │     │  (softmax)    │
└─────────────┘     └──────────────────┘     └────────────────┘     └──────────────┘
```

1. **Feature Extraction**: Stacked convolutional and pooling layers progressively transform raw pixels into abstract feature representations
2. **Feature Aggregation**: Global average pooling or fully connected layers compress spatial information into a fixed-size vector
3. **Classification/Prediction**: A final linear layer maps features to task-specific outputs

### Key Structural Properties

| Property | Description | Consequence |
|----------|-------------|-------------|
| **Local connectivity** | Each output depends on a small local region | Captures local spatial patterns efficiently |
| **Parameter sharing** | Same kernel applied at every spatial position | Dramatic parameter reduction vs. FC layers |
| **Translation equivariance** | Shifting input shifts output by the same amount | Detects features regardless of position |
| **Hierarchical features** | Deeper layers compose simpler features | Edges → textures → parts → objects |

---

## Why Convolution Works for Images

### Parameter Efficiency

Consider processing a $224 \times 224 \times 3$ RGB image:

| Layer Type | Connections | Parameters |
|-----------|-------------|------------|
| Fully connected (→ 1000 outputs) | $224 \times 224 \times 3 \times 1000$ | ~150 million |
| Conv2d ($3 \times 3$, 64 filters) | Local only | $3 \times 64 \times 3 \times 3 + 64 = 1{,}792$ |

A single convolutional layer uses **five orders of magnitude fewer parameters** than a fully connected layer processing the same input.

### What CNNs Learn

Through training, convolutional kernels self-organize into a hierarchy of feature detectors:

- **Early layers**: Edge detectors (horizontal, vertical, diagonal), color gradients, simple textures
- **Middle layers**: Corners, contours, texture patterns, object parts (eyes, wheels)
- **Deep layers**: Object-level features, scene composition, semantic concepts

This hierarchy emerges naturally from the combination of local receptive fields, non-linear activations, and spatial downsampling.

---

## Building Blocks of a CNN

### Convolutional Layers

The core building block. Each layer applies multiple learnable kernels to produce a stack of feature maps. Key parameters include kernel size, stride, padding, dilation, and the number of output channels. See [Convolution Operation](convolution.md) for the mathematical treatment.

### Padding and Stride

**Padding** adds values around the input border to control spatial dimensions and preserve edge information. **Stride** controls how far the kernel moves between applications, enabling spatial downsampling. See [Padding and Stride](padding_stride.md).

### Pooling Layers

Non-learnable downsampling operations (max pooling, average pooling) that reduce spatial dimensions while building translation invariance. **Global Average Pooling** (GAP) has largely replaced fully connected classifier heads in modern architectures. See [Pooling Layers](pooling.md).

### Feature Maps

The multi-channel outputs of convolutional layers form the intermediate representations of the network. Understanding how feature maps encode spatial and channel information is essential for architecture design and debugging. See [Feature Maps](feature_maps.md).

### Receptive Field

The region of the original input that influences a given neuron's activation. Receptive field analysis guides architecture design—ensuring the network can "see" enough context for the task. See [Receptive Field](receptive_field.md).

---

## Specialized Convolution Variants

Beyond the standard 2D convolution, several important variants address specific challenges:

| Variant | Key Idea | Primary Use Case |
|---------|----------|-----------------|
| [1D Convolutions](conv1d.md) | Operate on sequences | Time series, audio, NLP |
| [Depthwise Separable](depthwise_separable.md) | Factor spatial and channel mixing | Mobile/efficient architectures |
| [Dilated Convolutions](dilated_convolutions.md) | Expand receptive field without downsampling | Semantic segmentation, audio |
| [Transposed Convolutions](transposed_conv.md) | Learnable upsampling | Decoders, GANs, segmentation |

---

## Historical Evolution

The development of CNN architectures marks key milestones in deep learning:

| Year | Architecture | Key Innovation | Parameters |
|------|-------------|----------------|------------|
| 1998 | **LeNet-5** | First successful CNN (digit recognition) | ~60K |
| 2012 | **AlexNet** | ReLU, dropout, GPU training | ~60M |
| 2014 | **VGGNet** | Small $3 \times 3$ kernels, depth matters | ~138M |
| 2014 | **GoogLeNet** | Inception modules, multi-scale | ~6.8M |
| 2015 | **ResNet** | Skip connections, 150+ layers | ~25M |
| 2017 | **MobileNet** | Depthwise separable convolutions | ~3.4M |
| 2019 | **EfficientNet** | Compound scaling | ~5.3M |

The trend is clear: architectures have become deeper and more parameter-efficient, enabled by innovations in skip connections, efficient convolution variants, and training techniques.

---

## CNN Design Principles

### Modern Architecture Patterns

Most modern CNNs follow a common template:

```python
import torch.nn as nn

class ModernCNN(nn.Module):
    """Canonical modern CNN structure."""
    def __init__(self, num_classes):
        super().__init__()

        # Stem: aggressive early downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Body: stacked residual blocks with progressive downsampling
        # Stage 1: 64 channels, no downsampling
        # Stage 2: 128 channels, stride-2 at entry
        # Stage 3: 256 channels, stride-2 at entry
        # Stage 4: 512 channels, stride-2 at entry

        # Head: global average pooling + linear classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )
```

### Guiding Principles

1. **Use small kernels** ($3 \times 3$): Two stacked $3 \times 3$ convolutions have the same receptive field as one $5 \times 5$ but with fewer parameters and more non-linearity
2. **Double channels when halving resolution**: Maintains computational balance across stages
3. **Batch normalization after every convolution**: Stabilizes training and acts as regularization
4. **Skip connections for depth**: Enable training networks with 100+ layers
5. **Global Average Pooling over FC layers**: Reduces parameters by ~100× and allows variable input sizes

---

## Quantitative Finance Applications

CNNs find natural applications in quantitative finance where spatial or temporal structure is present:

- **Technical chart analysis**: Treating candlestick charts as images for pattern recognition
- **Limit order book modeling**: 1D convolutions over price/volume depth snapshots
- **Volatility surface fitting**: 2D convolutions over strike × maturity grids
- **Satellite imagery for commodity trading**: Crop health, shipping activity, energy infrastructure monitoring
- **Alternative data processing**: Parsing structured financial documents and tables

The architectural principles—locality, hierarchy, and parameter sharing—transfer directly: price patterns are local in time, volatility structures exhibit spatial correlation across strikes and maturities, and the same pattern (e.g., a supply disruption signature) can appear at different scales.

---

## Section Roadmap

This section provides a comprehensive treatment of CNNs, organized as follows:

1. **[Convolution Operation](convolution.md)**: Mathematical foundations of 2D convolution, multi-channel operations, and key properties
2. **[Padding and Stride](padding_stride.md)**: Controlling spatial dimensions and downsampling
3. **[Pooling Layers](pooling.md)**: Spatial downsampling, translation invariance, and modern alternatives
4. **[Feature Maps](feature_maps.md)**: Understanding intermediate representations and parameter counting
5. **[Receptive Field](receptive_field.md)**: Mathematical analysis and architecture design implications
6. **[1D Convolutions](conv1d.md)**: Temporal and sequence convolutions with PyTorch
7. **[Depthwise Separable Convolutions](depthwise_separable.md)**: Efficient architectures for mobile deployment
8. **[Dilated Convolutions](dilated_convolutions.md)**: Expanding receptive fields without downsampling
9. **[Transposed Convolutions](transposed_conv.md)**: Learnable upsampling for decoder networks

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278–2324.

2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 9: Convolutional Networks.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.

5. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.
