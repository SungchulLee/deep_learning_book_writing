# CLS Token

## Overview

The CLS (classification) token is a learnable embedding prepended to the input sequence in Vision Transformers. It serves as a global representation that aggregates information from all image patches through self-attention.

## Mechanism

At the input, a learnable vector $z_0^{\text{cls}}$ is prepended to the sequence of patch embeddings:

$$z_0 = [z_0^{\text{cls}}; z_0^1; z_0^2; \ldots; z_0^N] + E_{\text{pos}}$$

Through $L$ layers of self-attention, the CLS token attends to all patches and accumulates global information. The final CLS representation $z_L^{\text{cls}}$ is fed to the classification head.

## CLS vs Global Average Pooling

An alternative to the CLS token is global average pooling (GAP) over all patch representations:

$$z_{\text{GAP}} = \frac{1}{N}\sum_{i=1}^{N} z_L^i$$

| Approach | Advantage | Used By |
|----------|-----------|---------|
| CLS token | Flexible, learned aggregation | ViT, BERT, DINO |
| GAP | No extra parameter, symmetric | DeiT (optional), Swin |

In practice, both approaches achieve similar performance. GAP is slightly more robust to training hyperparameters; CLS allows the model to learn a task-specific aggregation.

## CLS in Self-Supervised Learning

In DINO and similar methods, the CLS token plays a special role: it captures global scene information while patch tokens retain local details. This separation is useful for both classification (via CLS) and dense prediction (via patch tokens).
