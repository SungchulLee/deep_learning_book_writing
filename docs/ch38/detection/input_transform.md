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

## Exercises

**Exercise 1.**
For a linear classifier $f(x) = w^T x + b$, compute the minimal $\ell_\infty$ perturbation needed to change the predicted class. Relate this to the robustness of neural networks.

??? success "Solution to Exercise 1"
    For a linear classifier, the distance to the decision boundary under $\ell_\infty$ norm is $\frac{|w^T x + b|}{\|w\|_1}$. The minimal perturbation is $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$. For neural networks, the local linear approximation $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$ explains why FGSM (which uses the sign of the gradient) is effective. The vulnerability of high-dimensional models comes from the fact that $\|w\|_1$ grows with dimension while $|w^T x + b|$ does not necessarily, making the robustness margin shrink. $\square$

---

**Exercise 2.**
Implement the attack or defense described in this section for a ResNet-18 model on CIFAR-10. Report clean accuracy and robust accuracy under PGD-20 attack with $\epsilon = 8/255$.

??? success "Solution to Exercise 2"
    A standard ResNet-18 achieves $\sim$93% clean accuracy but $\sim$0% robust accuracy under PGD-20 ($\epsilon = 8/255$, step size $2/255$). After applying the method from this section, typical results depend on the specific technique: adversarial training achieves $\sim$83% clean / $\sim$50% robust; certified defenses achieve lower but provable bounds. The accuracy-robustness trade-off is fundamental: improving robustness typically costs 5--15% clean accuracy. Report results averaged over 3 random seeds with standard errors. $\square$

---

**Exercise 3.**
Prove that no defense can simultaneously achieve high accuracy on clean data and high robustness against $\ell_\infty$ perturbations without increasing model capacity, under the assumption that the data distribution has overlapping class-conditional supports within the perturbation ball.

??? success "Solution to Exercise 3"
    If two classes have support overlap within distance $\epsilon$ (i.e., $\exists x_1 \in \text{class 1}, x_2 \in \text{class 2}$ with $\|x_1 - x_2\|_\infty \leq 2\epsilon$), then any classifier robust at both $x_1$ and $x_2$ must misclassify at least one of them (the perturbation balls overlap). This is the fundamental accuracy-robustness trade-off: the fraction of overlapping support determines the unavoidable accuracy loss. For natural image distributions, significant overlap exists at $\epsilon = 8/255$, explaining the observed 10--15% accuracy drop. Increased model capacity (wider networks) can better approximate the complex robust decision boundary, partially mitigating the trade-off. $\square$

---

**Exercise 4.**
Discuss how adversarial robustness concerns manifest in a financial machine learning system (e.g., fraud detection or trading signal generation). How does the threat model differ from computer vision?

??? success "Solution to Exercise 4"
    In finance, adversaries are strategic agents (fraudsters, market manipulators) who actively adapt to detection systems. Key differences from vision: (1) the perturbation space is constrained by what is economically feasible (a fraudster cannot change their entire transaction history); (2) attacks are sequential and adaptive (the adversary observes the system's response and adjusts); (3) the cost of false positives and false negatives is asymmetric (blocking a legitimate transaction vs. missing fraud); (4) $\ell_p$ norms are not meaningful -- domain-specific perturbation models are needed. Defenses must be robust to adaptive adversaries, which rules out many detection-based approaches that can be evaded once the detection criterion is known. $\square$
