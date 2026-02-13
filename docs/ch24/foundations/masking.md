# Masking Approaches

## Overview

Masking is the mechanism that enforces the autoregressive property: each output $x_i$ must depend only on preceding elements $x_{<i}$. Different masking strategies enable different architectures to maintain this constraint.

## Causal Masking in Attention

For transformer-based autoregressive models, causal masking sets future positions to $-\infty$ before softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M_{ij} = 0$ if $j \leq i$ and $M_{ij} = -\infty$ if $j > i$.

```python
def causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))
```

## Masked Convolutions

For CNN-based models (PixelCNN), the convolution kernel is masked to prevent access to future pixels:

### Type A Mask (first layer)
Excludes the current pixel — used only in the first layer:

```
1 1 1
1 0 0
0 0 0
```

### Type B Mask (subsequent layers)
Includes the current pixel — used in all layers after the first:

```
1 1 1
1 1 0
0 0 0
```

```python
def create_mask(kernel_size, mask_type='B'):
    mask = torch.ones(kernel_size, kernel_size)
    center = kernel_size // 2
    mask[center, center + (1 if mask_type == 'A' else 0):] = 0
    mask[center + 1:] = 0
    return mask
```

## MADE: Masked Autoencoder for Distribution Estimation

MADE assigns each hidden unit a random number $m_k \in \{1, \ldots, d-1\}$ and masks connections to ensure:

$$\text{output}_i \text{ depends only on inputs } \{x_j : j < i\}$$

The mask for weight matrix $W$ between layers: $M_{kj} = \mathbb{1}[m_k \geq j]$.

## Blind Spot Problem

In PixelCNN with raster scan masking, vertical stacks can only access pixels above the current row, creating a "blind spot" in the upper-right region. Gated PixelCNN resolves this using separate vertical and horizontal convolution stacks.
