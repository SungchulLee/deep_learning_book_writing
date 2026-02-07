# A2 Vision Transformers

## Overview

This appendix provides complete PyTorch implementations of Vision Transformer architectures that brought the transformer paradigm from NLP to computer vision. Each model demonstrates how self-attention mechanisms can capture global spatial relationships in images, offering advantages over the local receptive fields of traditional CNNs. These architectures are particularly relevant to quantitative finance for processing heterogeneous visual data and learning cross-modal representations.

## Architectures

| Model | Year | Key Innovation | Approach |
|-------|------|----------------|----------|
| [ViT](vision_transformer_vit.py) | 2020 | Pure transformer on image patches | Supervised pre-training on large datasets |
| [DeiT](deit.py) | 2021 | Data-efficient training via distillation | Knowledge distillation from CNN teachers |
| [Swin Transformer](swin_transformer.py) | 2021 | Shifted windows for hierarchical features | Local attention with cross-window connections |
| [BEiT](beit.py) | 2021 | BERT-style pre-training for vision | Masked image modeling with discrete visual tokens |
| [MAE](mae.py) | 2022 | Masked autoencoder with high mask ratios | Self-supervised learning by reconstructing patches |
| [CLIP](clip.py) | 2021 | Contrastive language–image pre-training | Joint vision-language embedding space |

## Key Concepts

### From CNNs to Transformers

Vision Transformers replace convolutional inductive biases (locality, translation equivariance) with learned spatial relationships through self-attention. The core pipeline is:

1. **Patch embedding**: Split image into fixed-size patches and project each to a token embedding
2. **Positional encoding**: Add learnable or sinusoidal position information
3. **Transformer encoder**: Apply multi-head self-attention and feed-forward layers
4. **Task head**: Classification token or global average pooling for downstream tasks

### Training Paradigms

- **Supervised**: ViT requires large-scale labeled data (ImageNet-21k or JFT-300M)
- **Distillation**: DeiT achieves competitive results using only ImageNet-1k with CNN teacher guidance
- **Self-supervised**: BEiT and MAE learn representations without labels, critical for domains with scarce annotations

### Efficiency Considerations

- **ViT**: Quadratic attention complexity $O(n^2)$ limits resolution scaling
- **Swin**: Linear complexity via windowed attention, suitable for high-resolution inputs
- **MAE**: Efficient pre-training by encoding only visible patches (25% of input)

## Quantitative Finance Applications

- **CLIP**: Cross-modal retrieval linking financial charts to textual descriptions or news headlines
- **MAE/BEiT**: Self-supervised pre-training on unlabeled financial imagery (charts, heatmaps, satellite data)
- **Swin Transformer**: High-resolution document analysis and dense prediction tasks on financial documents
- **ViT/DeiT**: Feature extraction backbone for multi-modal financial models

## Prerequisites

- [A1: Classic CNNs](../cnn/index.md) — convolutional baselines and design principles
- [A10: Utility Modules — Attention Mechanisms](../utils/attention.py) — multi-head attention internals
- [A10: Utility Modules — Positional Encodings](../utils/positional.py) — position representation strategies
- [Ch5: Convolutional Neural Networks](../../ch05/index.md) — patch embedding as convolution
