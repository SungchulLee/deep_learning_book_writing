# Fast Gradient Sign Method (FGSM)

## Introduction

The **Fast Gradient Sign Method (FGSM)** is the foundational gradient-based adversarial attack, introduced by Goodfellow et al. (2015). Its simplicity—requiring only a single gradient computation—makes it computationally efficient and serves as the building block for more sophisticated attacks.

## Mathematical Foundation

### The Linear Hypothesis

FGSM is motivated by the observation that neural networks behave approximately linearly in high-dimensional space. Consider a linear model:

$$
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}
$$

For a perturbation $\boldsymbol{\delta}$, the change in output is:

$$
f(\mathbf{x} + \boldsymbol{\delta}) - f(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\delta}
$$

To maximize this under $\ell_\infty$ constraint $\|\boldsymbol{\delta}\|_\infty \leq \varepsilon$, the optimal perturbation is:

$$
\delta_i^* = \varepsilon \cdot \text{sign}(w_i)
$$

This yields maximum change:

$$
\mathbf{w}^\top \boldsymbol{\delta}^* = \varepsilon \|\mathbf{w}\|_1
$$

In high dimensions, even small $\varepsilon$ produces significant $\varepsilon \|\mathbf{w}\|_1$.

### Extension to Neural Networks

For a neural network $f_\theta$ with loss function $\mathcal{L}$, we linearize via first-order Taylor expansion:

$$
\mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \approx \mathcal{L}(f_\theta(\mathbf{x}), y) + \boldsymbol{\delta}^\top \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)
$$

The gradient $\nabla_\mathbf{x} \mathcal{L}$ acts as the "weights" in our linear approximation. Applying the same logic:

$$
\boxed{\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y))}
$$

This is the **FGSM attack**.

### Intuition

The gradient $\nabla_\mathbf{x} \mathcal{L}$ tells us the direction in input space that most increases the loss. The sign function extracts only the direction ($+1$ or $-1$) for each coordinate, then we take the maximum allowed step ($\varepsilon$) in that direction.

**Visual Interpretation:**

```
Clean input x ────────────────> Loss L(f(x), y)
                gradient
                  ∇L
                   │
                   ▼
Perturbed x + ε·sign(∇L) ───> Loss L(f(x_adv), y) ≫ L(f(x), y)
```

## Algorithm

### Untargeted FGSM

**Input:** Clean example $\mathbf{x}$, true label $y$, model $f_\theta$, epsilon $\varepsilon$

**Output:** Adversarial example $\mathbf{x}_{\text{adv}}$

1. Compute loss: $\mathcal{L} = \text{CrossEntropy}(f_\theta(\mathbf{x}), y)$
2. Compute gradient: $\mathbf{g} = \nabla_\mathbf{x} \mathcal{L}$
3. Compute perturbation: $\boldsymbol{\delta} = \varepsilon \cdot \text{sign}(\mathbf{g})$
4. Generate adversarial example: $\mathbf{x}_{\text{adv}} = \text{clip}(\mathbf{x} + \boldsymbol{\delta}, 0, 1)$

**Complexity:** $O(1)$ — single forward-backward pass

### Targeted FGSM

For targeted attack toward class $y_{\text{target}}$:

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} - \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y_{\text{target}}))
$$

Note the **minus sign**: we descend the loss to pull predictions toward the target.

### L2 FGSM Variant

For $\ell_2$ constraint instead of $\ell_\infty$:

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \frac{\nabla_\mathbf{x} \mathcal{L}}{\|\nabla_\mathbf{x} \mathcal{L}\|_2}
$$

This normalizes the gradient to unit length, then scales by $\varepsilon$.

## PyTorch Implementation

### Complete FGSM Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

class FGSM:
    """
    Fast Gradient Sign Method (FGSM) Attack.
    
    Generates adversarial examples by taking a single gradient step
    in the direction that maximizes the loss.
    
    Mathematical Formulation
    ------------------------
    x_adv = x + ε · sign(∇_x L(f(x), y))
    
    where:
    - ∇_x L is the gradient of loss w.r.t. input
    - ε is the perturbation budget
    - sign(·) returns element-wise signs
    
    Parameters
    ----------
    model : nn.Module
        Neural network to attack
    epsilon : float
        Maximum L∞ perturbation (default: 8/255 for images)
    loss_fn : nn.Module, optional
        Loss function (default: CrossEntropyLoss)
    clip_min : float
        Minimum valid input value (default: 0.0)
    clip_max : float
        Maximum valid input value (default: 1.0)
    device : torch.device, optional
        Computation device
    
    Example
    -------
    >>> model = load_pretrained_model()
    >>> attack = FGSM(model, epsilon=8/255)
    >>> x_adv = attack.generate(images, labels)
    >>> metrics = attack.evaluate(images, labels, x_adv)
    >>> print(f"Attack success rate: {metrics['attack_success_rate']:.2%}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        loss_fn: Optional[nn.Module] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.
        
        Parameters
        ----------
        x : torch.Tensor
            Clean images, shape (N, C, H, W)
        y : torch.Tensor
            True labels, shape (N,)
        targeted : bool
            If True, perform targeted attack
        target_labels : torch.Tensor, optional
            Target labels for targeted attack
            
        Returns
        -------
        x_adv : torch.Tensor
            Adversarial examples, shape (N, C, H, W)
        """
        # Move inputs to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Enable gradient computation for input
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self.model(x_adv)
        
        # Compute loss
        if targeted:
            if target_labels is None:
                raise ValueError("target_labels required for targeted attack")
            target_labels = target_labels.to(self.device)
            loss = self.loss_fn(logits, target_labels)
        else:
            loss = self.loss_fn(logits, y)
        
        # Backward pass to compute gradient
        self.model.zero_grad()
        loss.backward()
        
        # Get gradient with respect to input
        grad = x_adv.grad.data
        
        # Compute perturbation
        if targeted:
            # Descend for targeted attack
            perturbation = -self.epsilon * torch.sign(grad)
        else:
            # Ascend for untargeted attack
            perturbation = self.epsilon * torch.sign(grad)
        
        # Apply perturbation and clip
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()
    
    def generate_l2(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples using L2-FGSM.
        
        Instead of sign(gradient), normalizes gradient to unit L2 norm.
        
        Returns
        -------
        x_adv : torch.Tensor
            Adversarial examples
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_adv = x.clone().detach().requires_grad_(True)
        logits = self.model(x_adv)
        
        if targeted:
            if target_labels is None:
                raise ValueError("target_labels required for targeted attack")
            loss = self.loss_fn(logits, target_labels.to(self.device))
        else:
            loss = self.loss_fn(logits, y)
        
        self.model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.data
        
        # Normalize gradient per example
        grad_flat = grad.view(grad.shape[0], -1)
        grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        grad_normalized = grad_flat / (grad_norm + 1e-8)
        grad_normalized = grad_normalized.view(grad.shape)
        
        if targeted:
            perturbation = -self.epsilon * grad_normalized
        else:
            perturbation = self.epsilon * grad_normalized
        
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate attack effectiveness.
        
        Parameters
        ----------
        x : torch.Tensor
            Clean images
        y : torch.Tensor
            True labels
        x_adv : torch.Tensor
            Adversarial images
        verbose : bool
            Print results
            
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        with torch.no_grad():
            # Clean predictions
            clean_logits = self.model(x.to(self.device))
            clean_pred = clean_logits.argmax(dim=1)
            clean_correct = (clean_pred == y.to(self.device)).sum().item()
            
            # Adversarial predictions
            adv_logits = self.model(x_adv.to(self.device))
            adv_pred = adv_logits.argmax(dim=1)
            adv_correct = (adv_pred == y.to(self.device)).sum().item()
            
            # Perturbation statistics
            delta = (x_adv - x).view(len(x), -1)
            linf_norm = delta.abs().max(dim=1)[0].mean().item()
            l2_norm = torch.norm(delta, p=2, dim=1).mean().item()
        
        n = len(y)
        metrics = {
            'clean_accuracy': clean_correct / n,
            'robust_accuracy': adv_correct / n,
            'attack_success_rate': 1 - adv_correct / n,
            'avg_linf_perturbation': linf_norm,
            'avg_l2_perturbation': l2_norm
        }
        
        if verbose:
            print("=" * 50)
            print("FGSM Attack Results")
            print("=" * 50)
            print(f"Epsilon: {self.epsilon:.4f}")
            print(f"Clean Accuracy: {metrics['clean_accuracy']:.2%}")
            print(f"Robust Accuracy: {metrics['robust_accuracy']:.2%}")
            print(f"Attack Success Rate: {metrics['attack_success_rate']:.2%}")
            print(f"Avg L∞ Perturbation: {linf_norm:.6f}")
            print(f"Avg L2 Perturbation: {l2_norm:.4f}")
            print("=" * 50)
        
        return metrics
    
    def visualize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor,
        class_names: Optional[list] = None,
        num_examples: int = 5,
        figsize: Tuple[int, int] = (15, 9)
    ) -> plt.Figure:
        """
        Visualize clean images, adversarial images, and perturbations.
        
        Parameters
        ----------
        x : torch.Tensor
            Clean images
        y : torch.Tensor
            True labels
        x_adv : torch.Tensor
            Adversarial images
        class_names : list, optional
            List of class names
        num_examples : int
            Number of examples to show
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.Figure
            Visualization figure
        """
        # Get predictions
        with torch.no_grad():
            clean_pred = self.model(x.to(self.device)).argmax(dim=1)
            adv_pred = self.model(x_adv.to(self.device)).argmax(dim=1)
        
        # Convert to numpy
        x_np = x[:num_examples].cpu().numpy()
        x_adv_np = x_adv[:num_examples].cpu().numpy()
        y_np = y[:num_examples].cpu().numpy()
        clean_pred_np = clean_pred[:num_examples].cpu().numpy()
        adv_pred_np = adv_pred[:num_examples].cpu().numpy()
        
        perturbations = x_adv_np - x_np
        
        fig, axes = plt.subplots(3, num_examples, figsize=figsize)
        
        for i in range(num_examples):
            # Determine if grayscale or RGB
            if x_np.shape[1] == 1:
                clean_img = x_np[i, 0]
                adv_img = x_adv_np[i, 0]
                pert_img = perturbations[i, 0]
                cmap = 'gray'
            else:
                clean_img = np.transpose(x_np[i], (1, 2, 0))
                adv_img = np.transpose(x_adv_np[i], (1, 2, 0))
                pert_img = np.transpose(perturbations[i], (1, 2, 0))
                cmap = None
            
            # Row 1: Clean images
            axes[0, i].imshow(np.clip(clean_img, 0, 1), cmap=cmap)
            true_label = class_names[y_np[i]] if class_names else y_np[i]
            pred_label = class_names[clean_pred_np[i]] if class_names else clean_pred_np[i]
            axes[0, i].set_title(f'Clean\nTrue: {true_label}', fontsize=9)
            axes[0, i].axis('off')
            
            # Row 2: Adversarial images
            axes[1, i].imshow(np.clip(adv_img, 0, 1), cmap=cmap)
            pred_label = class_names[adv_pred_np[i]] if class_names else adv_pred_np[i]
            color = 'red' if adv_pred_np[i] != y_np[i] else 'green'
            axes[1, i].set_title(f'Adversarial\nPred: {pred_label}', 
                                fontsize=9, color=color)
            axes[1, i].axis('off')
            
            # Row 3: Perturbations (magnified)
            pert_magnified = pert_img * 10 + 0.5
            axes[2, i].imshow(np.clip(pert_magnified, 0, 1), cmap='RdBu_r')
            axes[2, i].set_title('Perturbation\n(10× magnified)', fontsize=9)
            axes[2, i].axis('off')
        
        plt.suptitle(f'FGSM Attack (ε = {self.epsilon:.4f})', fontsize=12)
        plt.tight_layout()
        
        return fig
```

### Usage Example

```python
import torch
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Load pretrained model
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model.load_state_dict(torch.load('cifar10_resnet18.pth'))

# Create FGSM attack
attack = FGSM(model, epsilon=8/255)

# Get a batch
images, labels = next(iter(testloader))

# Generate adversarial examples
adv_images = attack.generate(images, labels)

# Evaluate
metrics = attack.evaluate(images, labels, adv_images)

# Visualize
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
fig = attack.visualize(images, labels, adv_images, class_names=class_names)
plt.savefig('fgsm_visualization.png', dpi=150, bbox_inches='tight')
```

## Analysis and Properties

### Epsilon Sensitivity

The choice of $\varepsilon$ critically affects attack success:

| Epsilon (8-bit) | Epsilon (float) | Effect |
|-----------------|-----------------|--------|
| 1/255 | 0.004 | Imperceptible, low success |
| 4/255 | 0.016 | Subtle, moderate success |
| 8/255 | 0.031 | **Standard**, high success |
| 16/255 | 0.063 | Visible, very high success |

**Experiment: Epsilon vs. Attack Success Rate**

```python
def epsilon_sensitivity_study(
    attack_class,
    model,
    x,
    y,
    epsilons=[1/255, 2/255, 4/255, 8/255, 16/255, 32/255]
):
    """Study how attack success rate varies with epsilon."""
    results = []
    
    for eps in epsilons:
        attack = attack_class(model, epsilon=eps)
        x_adv = attack.generate(x, y)
        metrics = attack.evaluate(x, y, x_adv, verbose=False)
        
        results.append({
            'epsilon': eps,
            'epsilon_255': int(eps * 255),
            **metrics
        })
    
    return results
```

### Strengths and Limitations

**Strengths:**
- Extremely fast (single gradient computation)
- Effective baseline for robustness evaluation
- Easy to implement and understand
- Works against most undefended models

**Limitations:**
- Single-step attack is suboptimal
- Easily defended against via adversarial training
- May fail against gradient-masked models
- Not suitable for finding minimal perturbations

### Comparison to Random Noise

FGSM vastly outperforms random noise of the same magnitude:

```python
def compare_fgsm_to_random(model, x, y, epsilon):
    """Compare FGSM to random perturbations."""
    device = next(model.parameters()).device
    
    # FGSM
    attack = FGSM(model, epsilon=epsilon)
    x_adv_fgsm = attack.generate(x, y)
    
    # Random uniform noise
    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv_random = torch.clamp(x + noise, 0, 1)
    
    # Evaluate both
    with torch.no_grad():
        y_dev = y.to(device)
        
        fgsm_pred = model(x_adv_fgsm.to(device)).argmax(dim=1)
        fgsm_success = (fgsm_pred != y_dev).float().mean().item()
        
        random_pred = model(x_adv_random.to(device)).argmax(dim=1)
        random_success = (random_pred != y_dev).float().mean().item()
    
    print(f"FGSM Success Rate: {fgsm_success:.2%}")
    print(f"Random Noise Success Rate: {random_success:.2%}")
    
    return fgsm_success, random_success
```

Typical results on CIFAR-10 with $\varepsilon = 8/255$:
- FGSM: ~60-80% success rate
- Random: ~5-10% success rate

## Variants and Extensions

### Fast Gradient Method (FGM)

Uses actual gradient instead of sign:

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \nabla_\mathbf{x} \mathcal{L}
$$

This is non-normalized and may violate $\ell_\infty$ constraints.

### Randomized FGSM

Adds random initialization before FGSM:

$$
\begin{aligned}
\mathbf{x}' &= \mathbf{x} + \alpha \cdot \mathbf{u}, \quad \mathbf{u} \sim \text{Uniform}[-1, 1]^d \\
\mathbf{x}_{\text{adv}} &= \mathbf{x}' + (\varepsilon - \alpha) \cdot \text{sign}(\nabla_{\mathbf{x}'} \mathcal{L})
\end{aligned}
$$

This helps escape poor local optima and is used in adversarial training.

### FGSM with Momentum

Accumulates gradient history (leads to MI-FGSM, covered in PGD section):

$$
\mathbf{g}_t = \mu \cdot \mathbf{g}_{t-1} + \frac{\nabla_\mathbf{x} \mathcal{L}}{\|\nabla_\mathbf{x} \mathcal{L}\|_1}
$$

## Connection to Adversarial Training

FGSM is central to adversarial training. The min-max optimization:

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y)}[\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)]
$$

approximates the inner max using FGSM:

$$
\boldsymbol{\delta}^{\text{FGSM}} = \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y))
$$

Training on FGSM adversarial examples provides basic robustness, though PGD-based training is stronger.

## Summary

| Aspect | FGSM |
|--------|------|
| **Formula** | $\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L})$ |
| **Complexity** | $O(1)$ — one forward-backward pass |
| **Strength** | Moderate (baseline) |
| **Standard $\varepsilon$** | $8/255$ for CIFAR-10 |
| **Key limitation** | Single-step, easily defended |

FGSM provides the conceptual foundation for understanding adversarial attacks. While stronger attacks like PGD and C&W exist, FGSM remains valuable for efficient evaluation and adversarial training.

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
2. Kurakin, A., Goodfellow, I., & Bengio, S. (2017). "Adversarial Examples in the Physical World." ICLR Workshop.
3. Tramèr, F., et al. (2018). "Ensemble Adversarial Training: Attacks and Defenses." ICLR.
