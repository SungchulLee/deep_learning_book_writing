# A1 Classic CNNs

## Overview

This appendix provides complete, annotated PyTorch implementations of landmark convolutional neural network architectures that shaped deep learning for computer vision. Each implementation includes the mathematical foundations, architectural diagrams, and practical training recipes relevant to quantitative finance applications such as chart pattern recognition, satellite imagery analysis, and alternative data processing.

## Architectures

### Foundational Architectures

| Model | Year | Key Innovation | Parameters |
|-------|------|----------------|------------|
| [LeNet](lenet.py) | 1998 | First successful CNN for digit recognition | ~60K |
| [AlexNet](alex_net.py) | 2012 | ReLU, dropout, GPU training at scale | ~61M |
| [VGGNet](vgg_net.py) | 2014 | Uniform 3×3 convolutions, depth matters | ~138M |

### Multi-Branch and Inception Designs

| Model | Year | Key Innovation | Parameters |
|-------|------|----------------|------------|
| [GoogLeNet](google_net_inception.py) | 2014 | Inception module, multi-scale feature extraction | ~6.8M |
| [Inception v3](inception_v3.py) | 2015 | Factorized convolutions, batch normalization, label smoothing | ~23.8M |
| [Xception](xception.py) | 2017 | Extreme Inception via depthwise separable convolutions | ~22.9M |

### Residual and Dense Architectures

| Model | Year | Key Innovation | Parameters |
|-------|------|----------------|------------|
| [ResNet](resnet.py) | 2015 | Skip connections, identity mappings | ~25.6M (ResNet-50) |
| [ResNeXt](res_next.py) | 2017 | Grouped convolutions, cardinality as a new dimension | ~25M |
| [DenseNet](densenet.py) | 2017 | Dense connectivity, feature reuse, parameter efficiency | ~8M (DenseNet-121) |
| [PyramidNet](pyramidnet.py) | 2017 | Gradual channel increase across residual blocks | ~26M |
| [ResNeSt](res_nest.py) | 2020 | Split-attention within cardinal groups | ~28M |

### Attention-Based

| Model | Year | Key Innovation |
|-------|------|----------------|
| [SENet](se_net.py) | 2018 | Squeeze-and-excitation channel attention |
| [CBAM](cbam.py) | 2018 | Sequential channel + spatial attention |

### Efficient / Lightweight Architectures

| Model | Year | Key Innovation | Parameters |
|-------|------|----------------|------------|
| [SqueezeNet](squeeze_net.py) | 2016 | Fire modules, AlexNet accuracy at 50× fewer parameters | ~1.2M |
| [MobileNet v2](mobilenet_v2.py) | 2018 | Inverted residuals with linear bottlenecks | ~3.4M |
| [ShuffleNet](shuffle_net.py) | 2018 | Channel shuffle for efficient group convolutions | ~2.3M |
| [GhostNet](ghost_net.py) | 2020 | Ghost modules via cheap linear operations | ~5.2M |
| [MixNet](mixnet.py) | 2019 | Mixed depthwise convolution kernel sizes | ~5M |

### Scaling and Search-Based

| Model | Year | Key Innovation | Parameters |
|-------|------|----------------|------------|
| [NASNet](nas_net.py) | 2018 | Neural architecture search, transferable cell design | ~5.3M (Mobile) |
| [EfficientNet](efficientnet.py) | 2019 | Compound scaling (depth, width, resolution) | ~5.3M (B0) |
| [EfficientNet v2](efficient_net_v2.py) | 2021 | Fused-MBConv, progressive learning | ~21M (S) |
| [RegNet](regnet.py) | 2020 | Design space quantization, simple parameterized networks | Configurable |

### Modern and Specialized

| Model | Year | Key Innovation | Parameters |
|-------|------|----------------|------------|
| [ConvNeXt](convnext.py) | 2022 | Modernized ResNet rivaling Vision Transformers | ~28.6M (T) |
| [ConvNeXt v2](convnext_v2.py) | 2023 | Global response normalization, MAE-compatible | ~28.6M (T) |
| [HRNet](hrnet.py) | 2019 | Maintains high-resolution representations throughout | ~28.5M (W32) |
| [CoordConv](coordconv.py) | 2018 | Appends coordinate channels to convolution input | Plug-in |
| [CapsuleNet](capsule_net.py) | 2017 | Capsules with dynamic routing for equivariant representations | ~8.2M |

## Design Principles Across Eras

- **Depth enables abstraction**: Deeper networks learn increasingly abstract features, from edges to textures to semantic concepts
- **Skip connections are essential**: Residual and dense connections solve the degradation problem, enabling very deep networks
- **Efficiency through design**: Depthwise separable convolutions, channel shuffling, and ghost modules achieve strong accuracy with minimal compute
- **Attention matters**: Channel (SE, CBAM) and spatial attention mechanisms consistently improve performance with minimal overhead
- **Scaling laws**: Compound scaling (EfficientNet) and design space search (RegNet, NASNet) systematize architecture design

## Quantitative Finance Applications

- **Technical chart pattern recognition** using candlestick chart images
- **Satellite and aerial imagery analysis** for commodity supply estimation
- **Document and receipt OCR** for alternative data extraction
- **Anomaly detection** in financial time series converted to image representations (e.g., Gramian Angular Fields, recurrence plots)
- **Edge deployment**: Lightweight models (MobileNet, ShuffleNet, GhostNet) for real-time inference on trading infrastructure

## Prerequisites

- [Ch5: Convolutional Neural Networks](../../ch05/index.md) — convolution operations, pooling, receptive fields
- [Ch4: Training Deep Networks](../../ch04/index.md) — batch normalization, dropout, learning rate schedules
- [Ch3: PyTorch nn.Module](../../ch03/index.md) — model construction patterns
