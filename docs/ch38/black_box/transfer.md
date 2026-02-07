# Transfer Attacks

## Introduction

**Transfer attacks** exploit a fundamental property of adversarial examples: perturbations crafted against one model (the **surrogate**) often transfer to fool a different model (the **target**). This enables **zero-query** black-box attacks—the adversary never needs to interact with the target model directly.

## Transferability Phenomenon

### Why Do Adversarial Examples Transfer?

Several factors explain cross-model transferability:

1. **Shared feature representations**: Models trained on similar data learn similar features, so adversarial perturbations that disrupt these features transfer across models
2. **Similar decision boundaries**: Models with similar architectures and training procedures develop similar decision boundaries
3. **Non-robust features are universal**: The non-robust features exploited by adversarial attacks are shared across model families

### Formal Setup

Given a surrogate model $f_s$ with full white-box access and a target model $f_t$ with no access:

$$
\boldsymbol{\delta}^* = \text{WhiteBoxAttack}(f_s, \mathbf{x}, y) \implies f_t(\mathbf{x} + \boldsymbol{\delta}^*) \neq y \text{ (with non-trivial probability)}
$$

Transfer rates vary widely depending on the relationship between surrogate and target models.

## Improving Transferability

### Momentum Iterative FGSM (MI-FGSM)

Dong et al. (2018) showed that adding **momentum** to iterative attacks significantly improves transferability. Standard PGD can overfit to the surrogate model's specific loss landscape; momentum stabilizes the attack direction.

The MI-FGSM update rule:

$$
\begin{aligned}
\mathbf{g}^{(t)} &= \mu \cdot \mathbf{g}^{(t-1)} + \frac{\nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)}{\|\nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)\|_1} \\
\mathbf{x}^{(t+1)} &= \Pi_\varepsilon\left(\mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\mathbf{g}^{(t)})\right)
\end{aligned}
$$

where $\mu$ is the momentum decay factor (typically 1.0).

### Ensemble Attacks

Attacking an ensemble of surrogate models improves transfer rates:

$$
\mathcal{L}_{\text{ensemble}} = \sum_{k=1}^K w_k \cdot \mathcal{L}(f_k(\mathbf{x} + \boldsymbol{\delta}), y)
$$

where $w_k$ are ensemble weights. Perturbations that fool multiple surrogates are more likely to transfer to an unseen target.

### Input Diversity (DI-FGSM)

Xie et al. (2019) proposed applying random transformations to the input during attack generation:

$$
\nabla_\mathbf{x} \mathcal{L}(f_s(T(\mathbf{x}^{(t)})), y)
$$

where $T$ applies random resizing and padding. This prevents the attack from overfitting to specific input patterns.

### Translation-Invariant Attack (TI-FGSM)

Convolving the gradient with a kernel makes the perturbation translation-invariant:

$$
\mathbf{g}^{(t)} = W * \nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)
$$

where $W$ is typically a Gaussian kernel.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

class TransferAttack:
    """
    Transfer-based black-box attack using momentum and ensemble.
    
    Generates adversarial examples against surrogate model(s)
    that transfer to an unknown target model.
    
    Parameters
    ----------
    surrogate_models : list[nn.Module]
        One or more surrogate models
    epsilon : float
        Perturbation budget
    alpha : float
        Step size
    num_iter : int
        Number of attack iterations
    momentum : float
        Momentum decay factor (0 = no momentum)
    input_diversity : bool
        Whether to use input diversity augmentation
    """
    
    def __init__(
        self,
        surrogate_models: List[nn.Module],
        epsilon: float = 8/255,
        alpha: Optional[float] = None,
        num_iter: int = 20,
        momentum: float = 1.0,
        input_diversity: bool = True,
        device: Optional[torch.device] = None
    ):
        self.surrogates = surrogate_models
        self.epsilon = epsilon
        self.alpha = alpha if alpha else epsilon / num_iter * 2
        self.num_iter = num_iter
        self.momentum = momentum
        self.input_diversity = input_diversity
        self.device = device or next(surrogate_models[0].parameters()).device
        
        for model in self.surrogates:
            model.eval()
            model.to(self.device)
    
    def _input_diversity_transform(
        self, x: torch.Tensor, p: float = 0.5
    ) -> torch.Tensor:
        """Apply random resizing and padding for input diversity."""
        if not self.input_diversity or torch.rand(1).item() > p:
            return x
        
        img_size = x.shape[-1]
        rnd = torch.randint(img_size, img_size + 8, (1,)).item()
        
        x_resized = F.interpolate(
            x, size=(rnd, rnd), mode='bilinear', align_corners=False
        )
        
        pad_top = torch.randint(0, rnd - img_size + 1, (1,)).item()
        pad_bottom = rnd - img_size - pad_top
        pad_left = torch.randint(0, rnd - img_size + 1, (1,)).item()
        pad_right = rnd - img_size - pad_left
        
        x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))
        x_padded = F.interpolate(
            x_padded, size=(img_size, img_size),
            mode='bilinear', align_corners=False
        )
        
        return x_padded
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate transferable adversarial examples."""
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_adv = x.clone()
        g_momentum = torch.zeros_like(x)
        
        for t in range(self.num_iter):
            x_adv.requires_grad_(True)
            
            # Ensemble loss over surrogate models
            total_loss = 0
            for model in self.surrogates:
                x_input = self._input_diversity_transform(x_adv)
                logits = model(x_input)
                
                if targeted:
                    loss = -F.cross_entropy(logits, target_labels)
                else:
                    loss = F.cross_entropy(logits, y)
                
                total_loss += loss / len(self.surrogates)
            
            # Compute gradient
            for model in self.surrogates:
                model.zero_grad()
            total_loss.backward()
            grad = x_adv.grad.data
            
            # Normalize gradient (L1 normalization)
            grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            
            # Apply momentum
            g_momentum = self.momentum * g_momentum + grad_norm
            
            # Update
            x_adv = x_adv.detach() + self.alpha * g_momentum.sign()
            
            # Project
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
```

## Transfer Rate Analysis

### Factors Affecting Transfer

| Factor | Effect on Transfer Rate | Explanation |
|--------|------------------------|-------------|
| Architecture similarity | Higher similarity → higher transfer | Similar architectures learn similar features |
| Training data overlap | More overlap → higher transfer | Shared data induces shared representations |
| Momentum | Significantly improves | Avoids surrogate-specific overfitting |
| Ensemble surrogates | Improves substantially | Cross-model features are more universal |
| Input diversity | Moderate improvement | Reduces input-specific overfitting |
| Iterations | Diminishing returns | More iterations overfit to surrogate |

### Typical Transfer Rates (CIFAR-10, $\varepsilon = 8/255$)

| Surrogate → Target | FGSM | MI-FGSM-20 | Ensemble MI-FGSM |
|---------------------|------|-----------|-------------------|
| ResNet-18 → VGG-16 | ~30% | ~50% | ~65% |
| ResNet-18 → DenseNet | ~35% | ~55% | ~70% |
| ResNet-18 → ResNet-50 | ~45% | ~65% | ~75% |

## Financial Applications

Transfer attacks are particularly relevant in financial settings where:

- An adversary may know the general type of model used (e.g., gradient boosting for credit scoring) but not the exact parameters
- Public models trained on similar data can serve as surrogates
- Regulatory requirements often specify model classes, enabling gray-box knowledge

## References

1. Dong, Y., et al. (2018). "Boosting Adversarial Attacks with Momentum." CVPR.
2. Xie, C., et al. (2019). "Improving Transferability of Adversarial Examples with Input Diversity." CVPR.
3. Tramèr, F., et al. (2018). "Ensemble Adversarial Training: Attacks and Defenses." ICLR.
4. Papernot, N., McDaniel, P., & Goodfellow, I. (2016). "Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples." arXiv.
