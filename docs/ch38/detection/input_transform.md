# Input Transformation Defenses

## Introduction

**Input transformation defenses** preprocess inputs before classification to remove or reduce adversarial perturbations. These methods modify the input using techniques such as denoising, compression, or reconstruction, aiming to "purify" adversarial examples while preserving clean classification accuracy.

## Common Transformations

### JPEG Compression

JPEG compression removes high-frequency components in the DCT domain, which often carry adversarial perturbations:

$$
\mathbf{x}_{\text{clean}} \approx \text{JPEG}(\mathbf{x}_{\text{adv}}, q)
$$

where $q$ is the quality factor. Lower quality removes more perturbation but also more signal.

### Randomized Resizing and Padding

Xie et al. (2018) proposed randomly resizing and padding inputs before classification. The randomness creates stochastic gradients that hinder gradient-based attacks:

```python
import torch
import torch.nn.functional as F

def random_resize_pad(x, target_size=32, resize_range=(28, 36)):
    """Randomly resize and pad input for defense."""
    rnd_size = torch.randint(resize_range[0], resize_range[1], (1,)).item()
    
    # Resize
    x_resized = F.interpolate(
        x, size=(rnd_size, rnd_size), mode='bilinear', align_corners=False
    )
    
    # Random padding to reach target size
    pad_top = torch.randint(0, target_size - rnd_size + 1, (1,)).item()
    pad_left = torch.randint(0, target_size - rnd_size + 1, (1,)).item()
    pad_bottom = target_size - rnd_size - pad_top
    pad_right = target_size - rnd_size - pad_left
    
    return F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))
```

### Denoising Autoencoders

Train a denoising autoencoder to reconstruct clean inputs from adversarial ones:

$$
\hat{\mathbf{x}} = D_\phi(\mathbf{x}_{\text{adv}}) \approx \mathbf{x}
$$

The denoiser is trained on pairs of (adversarial, clean) examples. The classifier then operates on the denoised input.

### Diffusion-Based Purification

Recent work uses **diffusion models** to purify adversarial examples by adding noise and then denoising:

1. Add Gaussian noise: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_{\text{adv}} + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$
2. Denoise using the diffusion model: $\hat{\mathbf{x}}_0 = \text{denoise}(\mathbf{x}_t, t)$
3. Classify: $\hat{y} = f(\hat{\mathbf{x}}_0)$

The noise injection disrupts adversarial structure, and the diffusion model reconstructs a clean approximation.

## Limitations

### The Gradient Masking Problem

Input transformation defenses often provide **gradient masking** rather than true robustness:

- The transformation may be non-differentiable, preventing gradient-based attacks
- However, the defense may still be vulnerable to:
    - Backward Pass Differentiable Approximation (BPDA)
    - Transfer attacks from models without the defense
    - Expectation over Transformation (EOT) attacks

### Accuracy Degradation

Transformations that remove adversarial perturbations also degrade clean inputs, reducing clean accuracy. The trade-off between perturbation removal and information preservation is inherent.

## Best Practices

1. **Always evaluate with adaptive attacks**: Use BPDA or EOT to attack through the transformation
2. **Combine with other defenses**: Use transformations alongside adversarial training, not as a replacement
3. **Monitor clean accuracy**: Ensure the transformation doesn't unacceptably degrade performance on clean inputs

## Summary

Input transformation defenses are intuitive and easy to implement but suffer from the gradient masking problem. They are most effective as part of a multi-layered defense strategy and must always be evaluated against adaptive adversaries.

## References

1. Xie, C., et al. (2018). "Mitigating Adversarial Effects Through Randomization." ICLR.
2. Nie, W., et al. (2022). "Diffusion Models for Adversarial Purification." ICML.
3. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples." ICML.
