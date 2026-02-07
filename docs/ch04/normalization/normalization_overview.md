# Normalization Layers Overview

Normalization layers stabilise and accelerate deep network training by controlling the distribution of intermediate activations. Every major architecture family — ResNets, Transformers, GANs — relies on some form of normalization, yet the methods differ in *which* dimensions they aggregate over and *when* statistics are computed. This page introduces the problem normalization solves, presents the general mathematical framework shared by all methods, and surveys the design space.

## Why Normalization Matters

### Internal Covariate Shift

Consider a deep network with layers $f_1, f_2, \ldots, f_L$. The output of layer $l$ feeds layer $l+1$:

$$h^{(l)} = f_l\!\bigl(h^{(l-1)};\, \theta^{(l)}\bigr)$$

During training the parameters $\theta^{(1)}, \ldots, \theta^{(l-1)}$ are updated at every step, so the distribution of $h^{(l-1)}$ shifts continuously. From layer $l$'s perspective its input distribution is a moving target — a phenomenon Ioffe and Szegedy (2015) called **internal covariate shift** (ICS).

The practical consequences are:

- **Gradient flow issues.** Shifting activations can push saturating nonlinearities (sigmoid, tanh) into flat regions, causing vanishing gradients. Even with ReLU, unbounded growth leads to exploding gradients.
- **Sensitivity to initialisation.** Without normalization, networks need careful weight initialisation schemes (Xavier, He) and conservative learning rates.
- **Slow convergence.** Each layer must adapt not only to the task but also to the changing statistics of its inputs, slowing optimisation.

### Loss Landscape Smoothing

Santurkar et al. (2018) offered a complementary explanation: normalization smooths the loss landscape by reducing the Lipschitz constant of the gradient:

$$\|\nabla_\theta \mathcal{L}(\theta_1) - \nabla_\theta \mathcal{L}(\theta_2)\| \leq L\,\|\theta_1 - \theta_2\|$$

A smaller $L$ means gradients change more gradually, permitting larger learning rates and more predictable optimisation.

### Regularisation Effect

When statistics are computed over a mini-batch (as in Batch Normalization), each sample's normalised value depends on which other samples share the mini-batch. This injects stochastic noise akin to dropout, providing implicit regularisation that diminishes as batch size grows.

## The General Normalization Framework

All widely-used normalization methods follow the same three-step pattern.

**Step 1 — Compute statistics** over some subset $\mathcal{S}$ of the input tensor:

$$\mu = \frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} x_k, \qquad \sigma^2 = \frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} (x_k - \mu)^2$$

**Step 2 — Normalise:**

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

where $\epsilon > 0$ (typically $10^{-5}$) prevents division by zero.

**Step 3 — Scale and shift** with learnable parameters:

$$y_i = \gamma\,\hat{x}_i + \beta$$

The parameters $\gamma$ (scale) and $\beta$ (shift) restore representational capacity: setting $\gamma = \sigma$ and $\beta = \mu$ recovers the un-normalised activation. In practice the network learns whatever affine transform benefits optimisation.

The methods differ solely in **which dimensions define $\mathcal{S}$**.

## Taxonomy of Normalization Methods

For a 4-D convolutional tensor with shape $(N, C, H, W)$:

| Method | Statistics over | \# Statistics | Batch dependent |
|--------|----------------|:------------:|:---------------:|
| **Batch Norm** | $(N, H, W)$ per channel | $C$ | Yes |
| **Layer Norm** | $(C, H, W)$ per sample | $N$ | No |
| **Instance Norm** | $(H, W)$ per sample-channel | $N \times C$ | No |
| **Group Norm** | $(C/G, H, W)$ per sample-group | $N \times G$ | No |
| **RMSNorm** | Features per sample (RMS only) | $N$ | No |

```
Input shape: (N, C, H, W)

Batch Norm      ▓▓▓ over N×H×W  →  one (μ, σ²) per C
Layer Norm      ▓▓▓ over C×H×W  →  one (μ, σ²) per N
Instance Norm   ▓▓▓ over H×W    →  one (μ, σ²) per (N, C)
Group Norm      ▓▓▓ over (C/G)×H×W  →  one (μ, σ²) per (N, G)
```

Group Normalization unifies the extremes: setting $G = 1$ recovers Layer Norm; setting $G = C$ recovers Instance Norm.

### Train / Inference Behaviour

| Method | Same at train and eval? | Running statistics? |
|--------|:-----------------------:|:-------------------:|
| Batch Norm | No | Yes |
| Layer Norm | Yes | No |
| Instance Norm | Yes | No |
| Group Norm | Yes | No |
| RMSNorm | Yes | No |

Batch Normalization is unique in requiring running statistics for inference and exhibiting different behaviour between `model.train()` and `model.eval()`. All other methods are stateless with respect to the data distribution.

## Choosing a Normalization Method

The choice depends primarily on the architecture and batch size:

```
Start
  │
  ├─► Transformer / sequence model?
  │     ├─ Standard ──► LayerNorm
  │     └─ Efficiency-focused LLM ──► RMSNorm
  │
  ├─► Style transfer / image-to-image GAN?
  │     └─► InstanceNorm
  │
  ├─► CNN for classification / detection / segmentation?
  │     ├─ Batch size ≥ 16 ──► BatchNorm
  │     └─ Batch size < 16 ──► GroupNorm
  │
  └─► Default
        ├─ Vision ──► GroupNorm (safest)
        └─ Sequences ──► LayerNorm
```

## Practical Considerations

### Layer Ordering

The dominant convention places normalization **after** the linear/convolutional operation and **before** the activation:

```python
nn.Conv2d(64, 128, 3, padding=1, bias=False)
nn.BatchNorm2d(128)
nn.ReLU(inplace=True)
```

Pre-activation variants (used in ResNet v2 and Pre-LN Transformers) normalise **before** the linear operation.

### Removing Redundant Bias

Because normalization subtracts the mean (or, for RMSNorm, re-scales), the bias in the preceding linear/conv layer is absorbed by $\beta$. Always set `bias=False` before a normalization layer.

### Combining with Dropout

When both are present, apply dropout **after** normalization and activation. BatchNorm already provides some regularisation, so the dropout rate can often be reduced.

## Summary

| Aspect | Key Insight |
|--------|-------------|
| **Core mechanism** | Normalise activations to stable statistics, then learn affine correction |
| **Methods differ by** | Which dimensions define the normalisation group $\mathcal{S}$ |
| **ICS vs smoothing** | Both explanations contribute; exact mechanism remains debated |
| **Learnable $\gamma, \beta$** | Preserve expressiveness while retaining optimisation benefits |
| **Batch dependence** | Only BatchNorm depends on batch composition |

## References

1. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML*.
2. Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). "How Does Batch Normalization Help Optimization?" *NeurIPS*.
3. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." *arXiv:1607.06450*.
4. Wu, Y., & He, K. (2018). "Group Normalization." *ECCV*.
5. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). "Instance Normalization: The Missing Ingredient for Fast Stylization." *arXiv:1607.08022*.
6. Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization." *NeurIPS*.
