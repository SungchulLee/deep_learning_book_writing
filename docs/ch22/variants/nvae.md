# NVAE: Nouveau VAE

A deep hierarchical VAE achieving state-of-the-art image generation quality.

---

## Learning Objectives

By the end of this section, you will be able to:

- Describe the key architectural innovations in NVAE
- Explain how depth-wise separable convolutions and squeeze-and-excitation improve efficiency
- Understand spectral regularization for stable deep VAE training
- Position NVAE relative to other generative models

---

## Overview

NVAE (Vahdat & Kautz, 2020) demonstrated that carefully designed deep hierarchical VAEs can achieve sample quality competitive with GANs while maintaining the benefits of likelihood-based training. The key insight is that VAE blurriness is not a fundamental limitation but rather a consequence of shallow architectures and training instability.

---

## Architecture

### Multi-Scale Hierarchy

NVAE uses a deep hierarchy (up to 30+ groups of latent variables) organized at multiple spatial scales:

```
Resolution:     2×2    4×4    8×8    16×16   32×32   64×64
                 │      │      │       │       │       │
Latent groups:  [z_L]  [z]   [z,z]  [z,z,z] [z,z,z] [z,z,z]
                 │      │      │       │       │       │
                 └──────┴──────┴───────┴───────┴───────┘
                              Decoder (top-down)
```

Each resolution scale contains multiple groups of latent variables, enabling fine-grained control over what information is captured at each spatial scale.

### Key Components

**Residual cells** use depth-wise separable convolutions for parameter efficiency, reducing the parameter count compared to standard convolutions while maintaining expressiveness:

```python
import torch
import torch.nn as nn

class NVAEResidualCell(nn.Module):
    """NVAE residual cell with depthwise separable convolutions."""
    
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        
        self.cell = nn.Sequential(
            nn.BatchNorm2d(channels),
            # Depthwise convolution
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, 
                     groups=channels),
            # Pointwise convolution
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(),
            # Second depthwise + pointwise
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2,
                     groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(),
        )
        
        # Squeeze-and-Excitation
        self.se = SEBlock(channels)
    
    def forward(self, x):
        h = self.cell(x)
        h = self.se(h)
        return x + h


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w
```

---

## Training Innovations

### Spectral Regularization

Deep VAEs are prone to training instability. NVAE applies **spectral regularization** to all weight matrices, constraining the Lipschitz constant of each layer:

$$\mathcal{L}_{\text{SR}} = \sum_l \left(\sigma_{\max}(W_l) - 1\right)^2$$

where $\sigma_{\max}(W_l)$ is the largest singular value of weight matrix $W_l$.

### Residual Normal Distributions

Instead of parameterizing $q_\phi(z_l | z_{>l}, x)$ directly, NVAE parameterizes the **residual** relative to the prior:

$$q_\phi(z_l | z_{>l}, x) = \mathcal{N}(\mu_{\text{prior}} + \Delta\mu, \sigma_{\text{prior}} \cdot \Delta\sigma)$$

where $\Delta\mu$ and $\Delta\sigma$ are the encoder's corrections to the prior. This makes the KL divergence between posterior and prior naturally small, improving training stability.

### Warm-Up and Scheduling

NVAE uses KL annealing with a long warm-up period, learning rate warm-up, and gradient clipping to stabilize the training of very deep hierarchies.

---

## Results and Significance

NVAE achieved state-of-the-art FID scores among VAE models at the time of publication, demonstrating that the traditional blurriness of VAE samples was a solvable architectural problem, deep hierarchies with proper training techniques can match or approach GAN quality, and likelihood-based training provides stable, reproducible results without mode collapse.

### Comparison (circa 2020)

| Model | FID (CIFAR-10) | Type |
|-------|----------------|------|
| Standard VAE | ~100+ | VAE |
| VQ-VAE-2 + PixelSNAIL | ~31 | Discrete VAE + AR |
| **NVAE** | **~51** | Hierarchical VAE |
| StyleGAN2 | ~2.8 | GAN |

While still behind state-of-the-art GANs, NVAE closed the gap significantly and later work (VDMs, etc.) continued to improve.

---

## Summary

| Innovation | Purpose |
|------------|---------|
| **Deep hierarchy** | 30+ latent groups at multiple scales |
| **Depthwise separable conv** | Parameter efficiency |
| **Squeeze-and-excitation** | Channel attention |
| **Spectral regularization** | Training stability |
| **Residual parameterization** | Small KL by default |

---

## Exercises

### Exercise 1: Residual Cell

Implement and test the NVAE residual cell on CIFAR-10. Compare parameter count against standard convolution residual blocks.

### Exercise 2: Spectral Regularization

Add spectral regularization to a standard hierarchical VAE. Does it improve training stability?

---

## What's Next

The next section covers [Training Optimization](../training/optimization.md) strategies for VAEs.
