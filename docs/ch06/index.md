# Chapter 6: Convolutional Neural Networks and Vision Transformers

## Overview

This chapter covers the foundational architectures for processing spatially structured data—primarily images, but with direct applications to financial time series, volatility surfaces, and other grid-like inputs encountered in quantitative finance. We progress from classical convolutional neural networks through residual architectures to modern vision transformers, tracing the evolution from hand-crafted inductive biases toward learned representations.

---

## Chapter Structure

### 6.1 Convolutional Neural Networks

The first section develops CNNs from first principles, beginning with the convolution operation and building toward complete architectures. CNNs exploit three structural priors—**locality**, **translation equivariance**, and **hierarchical composition**—to achieve remarkable parameter efficiency on spatially structured data.

We cover the core operations (convolution, padding, stride, pooling), analyze feature maps and receptive fields mathematically, and extend to specialized variants including 1D convolutions for time series, depthwise separable convolutions for efficiency, dilated convolutions for expanded receptive fields, and transposed convolutions for upsampling. Practical implementations on CIFAR-10 demonstrate the effects of batch normalization and dropout.

**Key topics**: Convolution operation · Padding and stride · Pooling layers · Feature maps · Receptive field analysis · 1D convolutions · Depthwise separable convolutions · Dilated convolutions · Transposed convolutions

### 6.2 Residual Connections

The second section addresses the degradation problem that limited early deep networks. Skip connections, introduced in ResNet, reformulate learning as residual estimation: instead of learning $H(x)$ directly, the network learns $F(x) = H(x) - x$, making identity mappings trivial.

We analyze the ResNet architecture in detail, examine identity mapping variants and their effect on gradient flow, and extend to DenseNet's dense connections and highway networks. The section provides rigorous gradient flow analysis explaining why these architectures enable training of networks with hundreds of layers.

**Key topics**: Skip connections · ResNet architecture · Identity mapping · Gradient flow analysis · DenseNet · Highway networks · Transfer learning

### 6.3 Vision Transformers

The third section covers the application of transformer architectures to vision tasks. Vision Transformers (ViT) challenge the assumption that convolution is necessary for image understanding by treating images as sequences of patches and applying standard transformer mechanisms.

We develop the ViT pipeline from patch embedding through position encoding and the CLS token mechanism to the full architecture. We then cover training-efficient variants (DeiT), hierarchical designs (Swin Transformer), and hybrid architectures that combine convolutional and attention-based processing.

**Key topics**: Patch embedding · Position embeddings for images · CLS token · ViT architecture · DeiT · Swin Transformer · Hybrid CNN-Transformer architectures

---

## Quantitative Finance Context

The architectures in this chapter have direct applications across quantitative finance:

| Architecture | Financial Application |
|---|---|
| **2D CNNs** | Volatility surface modeling, candlestick chart pattern recognition, satellite imagery analysis for commodity trading |
| **1D CNNs** | Limit order book feature extraction, return series pattern detection, multi-factor signal processing |
| **ResNet** | Deep multi-factor models with linear factor exposure preserved via skip connections |
| **DenseNet** | Multi-horizon forecasting with feature reuse across prediction horizons |
| **ViT** | Cross-asset attention patterns, global relationship modeling in large asset universes |
| **Swin Transformer** | Hierarchical market microstructure analysis at multiple time scales |
| **Hybrid architectures** | Local pattern extraction (CNN) combined with global dependency modeling (Transformer) for alpha generation |

The progression from CNNs to transformers mirrors a broader trend in quantitative finance: moving from models that impose strong structural assumptions (locality, stationarity) toward architectures that can learn relationships directly from data, trading inductive bias for flexibility when sufficient data is available.

---

## Prerequisites

This chapter assumes familiarity with:

- **Fully connected networks** (Chapter 4): Dense layers, backpropagation, activation functions
- **Optimization** (Chapter 5): SGD, Adam, learning rate scheduling, batch normalization, dropout
- **Linear algebra**: Matrix operations, eigendecomposition, tensor products
- **PyTorch fundamentals**: `nn.Module`, autograd, data loading pipelines

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
4. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.
