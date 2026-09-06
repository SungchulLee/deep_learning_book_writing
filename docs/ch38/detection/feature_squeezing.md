# Feature Squeezing
## Introduction

**Feature squeezing** (Xu et al., 2018) detects adversarial examples by comparing model predictions on the original input versus reduced-complexity ("squeezed") versions. The key insight is that adversarial perturbations rely on precise, often subtle pixel patterns that are disrupted by input compression, while clean images are largely unaffected.

## Core Idea

Given an input $\mathbf{x}$ and a set of squeezing functions $\{s_1, s_2, \ldots\}$, compare predictions:

$$
\text{Detection Score} = \max_j \|f(\mathbf{x}) - f(s_j(\mathbf{x}))\|_1
$$

If the maximum prediction difference exceeds a threshold, the input is flagged as adversarial.

## Squeezing Operations

### Bit-Depth Reduction

Reduce the number of bits per color channel, quantizing pixel values:

$$
s_{\text{bit}}(\mathbf{x}; b) = \text{round}(\mathbf{x} \cdot 2^b) / 2^b
$$

For example, reducing from 8-bit (256 levels) to 4-bit (16 levels) eliminates subtle perturbations while preserving image structure.

### Spatial Smoothing

Apply spatial filters that blur high-frequency perturbations:

- **Median filter**: Non-linear, preserves edges, effective against salt-and-pepper perturbations
- **Gaussian blur**: Linear smoothing, reduces high-frequency noise
- **Non-local means**: Adaptive denoising based on patch similarity

### JPEG Compression

JPEG discards high-frequency components in the DCT domain, naturally removing many adversarial perturbations.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class FeatureSqueezing:
    """
    Feature Squeezing detector for adversarial examples.
    
    Compares model predictions on original vs squeezed inputs.
    Large prediction differences indicate adversarial manipulation.
    
    Parameters
    ----------
    model : nn.Module
        Target classifier
    squeezers : list[callable]
        List of squeezing functions
    threshold : float
        Detection threshold on L1 prediction difference
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.1,
        bit_depth: int = 4,
        median_kernel: int = 3,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.threshold = threshold
        self.bit_depth = bit_depth
        self.median_kernel = median_kernel
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def _squeeze_bit_depth(
        self, x: torch.Tensor, bits: int
    ) -> torch.Tensor:
        """Reduce bit depth of input."""
        levels = 2 ** bits
        return torch.round(x * levels) / levels
    
    def _squeeze_median(
        self, x: torch.Tensor, kernel_size: int = 3
    ) -> torch.Tensor:
        """Apply median filter."""
        # Median filter per channel
        pad = kernel_size // 2
        x_pad = F.pad(x, [pad] * 4, mode='reflect')
        
        # Unfold to get patches
        patches = x_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        return patches.contiguous().view(*patches.shape[:4], -1).median(dim=-1)[0]
    
    def _squeeze_gaussian(
        self, x: torch.Tensor, sigma: float = 1.0
    ) -> torch.Tensor:
        """Apply Gaussian blur."""
        k = int(4 * sigma + 1)
        if k % 2 == 0:
            k += 1
        
        coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.outer(kernel_1d)
        
        C = x.shape[1]
        kernel = kernel_2d.expand(C, 1, k, k)
        
        return F.conv2d(x, kernel, padding=k//2, groups=C)
    
    def detect(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect adversarial examples via feature squeezing.
        
        Returns detection scores and binary decisions.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # Original prediction
            pred_orig = F.softmax(self.model(x), dim=1)
            
            # Squeezed predictions
            max_diff = torch.zeros(len(x), device=self.device)
            
            # Squeezer 1: Bit depth reduction
            x_squeezed = self._squeeze_bit_depth(x, self.bit_depth)
            pred_sq = F.softmax(self.model(x_squeezed), dim=1)
            diff = (pred_orig - pred_sq).abs().sum(dim=1)
            max_diff = torch.max(max_diff, diff)
            
            # Squeezer 2: Median filter
            x_squeezed = self._squeeze_median(x, self.median_kernel)
            pred_sq = F.softmax(self.model(x_squeezed), dim=1)
            diff = (pred_orig - pred_sq).abs().sum(dim=1)
            max_diff = torch.max(max_diff, diff)
            
            # Squeezer 3: Gaussian blur
            x_squeezed = self._squeeze_gaussian(x, sigma=1.0)
            pred_sq = F.softmax(self.model(x_squeezed), dim=1)
            diff = (pred_orig - pred_sq).abs().sum(dim=1)
            max_diff = torch.max(max_diff, diff)
        
        return {
            'detection_score': max_diff,
            'is_adversarial': max_diff > self.threshold
        }
```

## Limitations

- **Adaptive attacks**: An adversary aware of the squeezing operations can craft perturbations that survive squeezing
- **Accuracy impact**: Squeezing clean inputs can also alter predictions, causing false positives
- **Threshold selection**: The detection threshold requires careful calibration on clean data

## Summary

Feature squeezing provides a simple, model-agnostic detection layer. It is most effective as part of a defense-in-depth strategy rather than a standalone defense, particularly against non-adaptive adversaries.

## References

1. Xu, W., Evans, D., & Qi, Y. (2018). "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks." NDSS.

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
