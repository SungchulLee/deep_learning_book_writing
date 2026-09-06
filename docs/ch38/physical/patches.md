# Adversarial Patches
## Introduction

**Adversarial patches** are localized, physically realizable adversarial perturbations that can be printed and placed in the physical world to fool classifiers. Unlike $\ell_p$-bounded perturbations that modify every pixel imperceptibly, patches modify a small region of the input with unconstrained magnitude. This makes them a practical attack vector for physical-world systems.

## Mathematical Formulation

An adversarial patch $P$ is applied to an input $\mathbf{x}$ via a patch application operator $A$:

$$
\mathbf{x}_{\text{patched}} = A(\mathbf{x}, P, l, t) = (1 - M_l) \odot \mathbf{x} + M_l \odot t(P)
$$

where:

- $M_l$ is a binary mask at location $l$ defining where the patch is placed
- $t(\cdot)$ applies a spatial transformation (rotation, scaling, perspective)
- $\odot$ denotes element-wise multiplication

### Universal Patch Optimization

A **universal adversarial patch** fools the model regardless of the underlying image:

$$
P^* = \arg\max_P \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \mathbb{E}_{l \sim \mathcal{U}} \mathbb{E}_{t \sim \mathcal{T}} \left[ \mathcal{L}(f_\theta(A(\mathbf{x}, P, l, t)), y_{\text{target}}) \right]
$$

The expectations are over:

- Random images from the data distribution
- Random patch locations
- Random physical transformations

### Expectation over Transformations (EOT)

To ensure robustness to physical-world variations, patches are optimized using **Expectation over Transformations** (Athalye et al., 2018):

$$
\nabla_P \mathbb{E}_{t \sim \mathcal{T}} [\mathcal{L}(f(t(A(\mathbf{x}, P))))] \approx \frac{1}{K} \sum_{k=1}^K \nabla_P \mathcal{L}(f(t_k(A(\mathbf{x}, P))))
$$

Transformations include rotation, scaling, brightness changes, and perspective warps.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AdversarialPatch:
    """
    Universal Adversarial Patch generator.
    
    Creates a patch that, when placed on any image,
    causes the model to predict a target class.
    
    Parameters
    ----------
    model : nn.Module
        Target classifier
    patch_size : tuple
        Size of the patch (H, W)
    target_class : int
        Class the patch should trigger
    """
    
    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int] = (8, 8),
        target_class: int = 0,
        learning_rate: float = 0.01,
        num_transforms: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.patch_size = patch_size
        self.target_class = target_class
        self.lr = learning_rate
        self.num_transforms = num_transforms
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        
        # Initialize random patch
        self.patch = torch.rand(
            1, 3, *patch_size, device=self.device, requires_grad=True
        )
    
    def _apply_patch(
        self, x: torch.Tensor, patch: torch.Tensor,
        location: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply patch at a random or specified location."""
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        
        if location is None:
            r = torch.randint(0, H - ph + 1, (1,)).item()
            c = torch.randint(0, W - pw + 1, (1,)).item()
        else:
            r, c = location
        
        x_patched = x.clone()
        patch_resized = F.interpolate(patch, size=(ph, pw), mode='bilinear')
        patch_resized = torch.clamp(patch_resized, 0, 1)
        x_patched[:, :, r:r+ph, c:c+pw] = patch_resized.expand(B, -1, -1, -1)
        
        return x_patched
    
    def _random_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random brightness/contrast for EOT."""
        brightness = 0.9 + 0.2 * torch.rand(1, device=self.device)
        return torch.clamp(x * brightness, 0, 1)
    
    def train_patch(
        self, train_loader, epochs: int = 10
    ) -> torch.Tensor:
        """
        Train universal adversarial patch.
        
        Returns trained patch tensor.
        """
        optimizer = torch.optim.Adam([self.patch], lr=self.lr)
        target = torch.tensor(
            [self.target_class], device=self.device
        )
        
        for epoch in range(epochs):
            total_loss = 0
            success = 0
            total = 0
            
            for x, y in train_loader:
                x = x.to(self.device)
                batch_target = target.expand(x.shape[0])
                
                optimizer.zero_grad()
                
                # EOT: average loss over random transformations
                loss = 0
                for _ in range(self.num_transforms):
                    x_patched = self._apply_patch(x, self.patch)
                    x_patched = self._random_transform(x_patched)
                    logits = self.model(x_patched)
                    loss += F.cross_entropy(logits, batch_target)
                
                loss /= self.num_transforms
                loss.backward()
                optimizer.step()
                
                # Clamp patch to valid range
                with torch.no_grad():
                    self.patch.clamp_(0, 1)
                
                total_loss += loss.item() * len(x)
                with torch.no_grad():
                    pred = self.model(self._apply_patch(x, self.patch)).argmax(1)
                    success += (pred == batch_target).sum().item()
                    total += len(x)
            
            print(f"Epoch {epoch+1}: Loss={total_loss/total:.4f}, "
                  f"Success={success/total:.2%}")
        
        return self.patch.detach()
```

## Physical-World Considerations

### Printing and Deployment

Adversarial patches must survive the physical pipeline:

1. **Digital to print**: Color calibration, printer resolution limits
2. **Environmental factors**: Lighting, shadows, viewing angle
3. **Camera capture**: Lens distortion, auto-exposure, noise

EOT training accounts for these by sampling from the distribution of physical transformations.

### Patch Properties

| Property | Digital Patches | Physical Patches |
|----------|----------------|-----------------|
| Perturbation type | Pixel-level | Printed material |
| Constraint | Location + size | Location + size + robustness |
| Transformation invariance | Optional | Essential |
| Success rate | >90% | 60-85% |

## Financial Applications

While primarily studied in computer vision, the patch concept extends to financial settings:

- **Feature injection**: Analogous to inserting a small number of adversarial features into a transaction record
- **Sensor manipulation**: Physically interfering with data collection sensors (e.g., satellite imagery for commodity trading)
- **Document forgery**: Small localized modifications to scanned documents to fool automated processing

## References

1. Brown, T. B., et al. (2017). "Adversarial Patch." arXiv preprint arXiv:1712.09665.
2. Athalye, A., et al. (2018). "Synthesizing Robust Adversarial Examples." ICML.
3. Eykholt, K., et al. (2018). "Robust Physical-World Attacks on Deep Learning Visual Classification." CVPR.

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
