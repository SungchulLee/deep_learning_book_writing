# A10 Utility Modules

## Overview

This appendix provides complete PyTorch implementations of reusable building blocks that appear across multiple deep learning architectures. Rather than re-implementing these components within each model, they are collected here as standalone, well-documented modules with consistent APIs. Each module includes the mathematical derivation, implementation, and usage examples showing how it integrates into larger architectures.

## Modules

| Module | Description | Used By |
|--------|-------------|---------|
| [Attention Mechanisms](attention.py) | Scaled dot-product, multi-head, cross-attention, additive (Bahdanau) attention | Transformers, Seq2Seq, GAT, DETR |
| [Normalization Layers](normalization.py) | BatchNorm, LayerNorm, GroupNorm, RMSNorm, InstanceNorm | All architectures |
| [Activation Functions](activations.py) | ReLU, GELU, SiLU/Swish, Mish, GLU variants, Softmax | All architectures |
| [Positional Encodings](positional.py) | Sinusoidal, learned, rotary (RoPE), ALiBi, relative position bias | Transformers, ViT, LLaMA |
| [Loss Functions](losses.py) | Cross-entropy, focal loss, contrastive, triplet, InfoNCE, reconstruction losses | Detection, CLIP, VAE, RL |
| [Learning Rate Schedulers](schedulers.py) | Warm-up, cosine annealing, step decay, one-cycle, polynomial decay | All training pipelines |
| [Data Augmentation](augmentation.py) | Random crop, flip, color jitter, mixup, cutout, RandAugment | CNN training, ViT, contrastive learning |

## Design Principles

### Composability

Each module is designed as a drop-in `nn.Module` (or utility function) with a standardized interface:

```python
# All attention modules: (query, key, value, mask=None) -> (output, attention_weights)
# All normalization modules: (input) -> (normalized_output)
# All activations: (input) -> (activated_output)
```

### Mathematical Rigor

Every module includes:

1. **Mathematical definition** with precise notation
2. **Gradient behavior** — how the module affects backpropagation
3. **Numerical considerations** — stability, precision, and edge cases
4. **Computational complexity** — time and memory scaling

### Practical Integration

Each module documents:

- Which architectures use it and how
- Configuration choices and their trade-offs
- Common pitfalls and debugging tips
- PyTorch built-in equivalents where available

## Module Dependencies

```
Attention ← Positional Encodings (RoPE, sinusoidal)
         ← Normalization (pre-norm vs post-norm)
         ← Activation Functions (softmax, GELU in FFN)

Loss Functions ← Activation Functions (log-softmax, sigmoid)

Data Augmentation ← (standalone, no internal dependencies)

LR Schedulers ← (standalone, wraps torch.optim)
```

## Quick Reference: Where Each Module Is Used

### Attention Mechanisms
- **Scaled dot-product**: Original Transformer, BERT, GPT, ViT, DETR
- **Multi-head**: All transformer variants
- **Cross-attention**: Seq2Seq, T5, Stable Diffusion, DETR
- **Additive (Bahdanau)**: Attention Seq2Seq
- **Graph attention**: GAT

### Normalization Layers
- **BatchNorm**: ResNet, DenseNet, DCGAN, EfficientNet
- **LayerNorm**: All transformers, ViT
- **RMSNorm**: LLaMA
- **GroupNorm**: U-Net, detection models
- **InstanceNorm**: StyleGAN, CycleGAN, Pix2Pix

### Positional Encodings
- **Sinusoidal**: Original Transformer, DDPM time embedding
- **Learned**: BERT, GPT, ViT
- **Rotary (RoPE)**: LLaMA, modern LLMs
- **Relative bias**: Swin Transformer, T5

## Prerequisites

- [Ch3: Neural Network Fundamentals](../../ch03/index.md) — `nn.Module`, parameter registration, forward pass
- [Ch4: Training Deep Networks](../../ch04/index.md) — optimizer integration, training loops
