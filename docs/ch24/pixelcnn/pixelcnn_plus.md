# PixelCNN++

## Overview

PixelCNN++ (Salimans et al., 2017) introduces several improvements over Gated PixelCNN that significantly improve both sample quality and training efficiency.

## Key Improvements

### Discretized Logistic Mixture Likelihood

Instead of softmax over 256 values, model each pixel with a mixture of logistics:

$$p(x \mid \pi, \mu, s) = \sum_{k=1}^{K} \pi_k [\sigma((x + 0.5 - \mu_k) / s_k) - \sigma((x - 0.5 - \mu_k) / s_k)]$$

This is more parameter-efficient (mixture components share structure) and avoids the 256-way classification problem.

### Downsampling with Strided Convolutions

Use a U-Net-like architecture with downsampling and upsampling, processing at multiple resolutions. Short-range dependencies are handled at full resolution; long-range dependencies at lower resolution.

### Skip Connections

Dense skip connections between layers at the same resolution, following the ResNet/U-Net pattern.

### Dropout Regularization

Dropout applied to the residual path for regularization.

## Architecture

```
Input → [ResBlocks at full res] → Downsample
     → [ResBlocks at half res] → Downsample  
     → [ResBlocks at quarter res] → Upsample
     → [ResBlocks at half res + skip] → Upsample
     → [ResBlocks at full res + skip] → Output
```

## Results

PixelCNN++ achieves significantly better bits per dimension than Gated PixelCNN:

| Model | CIFAR-10 BPD |
|-------|-------------|
| PixelCNN | 3.14 |
| Gated PixelCNN | 3.03 |
| PixelCNN++ | 2.92 |

The discretized logistic mixture accounts for most of the improvement, followed by the multi-scale architecture.
