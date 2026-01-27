# EBM Applications

## Learning Objectives

After completing this section, you will be able to:

1. Apply EBMs for out-of-distribution detection
2. Use energy minimization for image denoising
3. Implement compositional generation
4. Leverage EBMs for adversarial robustness
5. Understand real-world deployment considerations

## Introduction

Energy-Based Models offer unique capabilities beyond traditional generation. Their explicit energy function enables applications like anomaly detection, denoising, and compositional reasoning that are difficult with other generative approaches.

## Out-of-Distribution Detection

### Motivation

EBMs naturally provide anomaly scores: data the model has seen should have low energy, while novel or anomalous data should have high energy.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

def ood_detection_with_ebm(energy_net, in_distribution_data, ood_data):
    """
    Use energy as OOD score.
    
    Lower energy → In-distribution
    Higher energy → Out-of-distribution
    """
    with torch.no_grad():
        # Compute energies
        in_dist_energies = energy_net(in_distribution_data).numpy()
        ood_energies = energy_net(ood_data).numpy()
    
    # Create labels (0 = in-dist, 1 = OOD)
    labels = np.concatenate([
        np.zeros(len(in_dist_energies)),
        np.ones(len(ood_energies))
    ])
    scores = np.concatenate([in_dist_energies, ood_energies])
    
    # Higher energy = more likely OOD, so use energy directly as score
    auroc = roc_auc_score(labels, scores)
    
    return {
        'auroc': auroc,
        'in_dist_mean_energy': in_dist_energies.mean(),
        'ood_mean_energy': ood_energies.mean()
    }


class EBMOODDetector:
    """
    Out-of-distribution detector using EBM energy.
    """
    
    def __init__(self, energy_net, threshold=None):
        self.energy_net = energy_net
        self.threshold = threshold
    
    def fit_threshold(self, in_dist_data, percentile=95):
        """Set threshold based on in-distribution data."""
        with torch.no_grad():
            energies = self.energy_net(in_dist_data).numpy()
        self.threshold = np.percentile(energies, percentile)
        return self.threshold
    
    def predict(self, x):
        """Predict if samples are OOD (1) or in-distribution (0)."""
        with torch.no_grad():
            energies = self.energy_net(x).numpy()
        return (energies > self.threshold).astype(int)
    
    def score(self, x):
        """Return OOD scores (higher = more likely OOD)."""
        with torch.no_grad():
            return self.energy_net(x).numpy()
```

### Example: MNIST vs Fashion-MNIST

```python
def ood_experiment():
    """
    Train on MNIST, detect Fashion-MNIST as OOD.
    """
    from torchvision import datasets, transforms
    
    # Load data
    transform = transforms.ToTensor()
    mnist = datasets.MNIST('./data', train=False, transform=transform)
    fashion = datasets.FashionMNIST('./data', train=False, transform=transform)
    
    # Assume energy_net is trained on MNIST
    # ... training code ...
    
    # Evaluate OOD detection
    mnist_samples = torch.stack([mnist[i][0] for i in range(1000)])
    fashion_samples = torch.stack([fashion[i][0] for i in range(1000)])
    
    results = ood_detection_with_ebm(energy_net, mnist_samples, fashion_samples)
    
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"MNIST mean energy: {results['in_dist_mean_energy']:.2f}")
    print(f"Fashion mean energy: {results['ood_mean_energy']:.2f}")
```

## Image Denoising

### Energy Minimization for Denoising

Given a noisy image $\mathbf{y} = \mathbf{x} + \boldsymbol{\epsilon}$, find clean image by minimizing:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \left[ E_\theta(\mathbf{x}) + \lambda \|\mathbf{x} - \mathbf{y}\|^2 \right]$$

The energy term encourages natural images, while the data fidelity term keeps us close to the observation.

```python
def denoise_with_ebm(energy_net, noisy_image, n_steps=100, 
                     step_size=0.01, data_weight=10.0):
    """
    Denoise image via energy minimization with data fidelity.
    
    Minimizes: E(x) + λ||x - y||²
    """
    x = noisy_image.clone().requires_grad_(True)
    
    for _ in range(n_steps):
        # Energy gradient
        energy = energy_net(x)
        energy_grad = torch.autograd.grad(energy.sum(), x)[0]
        
        # Data fidelity gradient
        data_grad = 2 * data_weight * (x - noisy_image)
        
        # Combined gradient descent
        total_grad = energy_grad + data_grad
        x = x.detach() - step_size * total_grad
        x = x.clamp(0, 1)
        x.requires_grad_(True)
    
    return x.detach()


def progressive_denoising(energy_net, noisy_image, 
                          noise_levels=[1.0, 0.5, 0.25, 0.1]):
    """
    Multi-scale denoising starting from high noise tolerance.
    """
    x = noisy_image.clone()
    
    for data_weight in noise_levels:
        # Lower weight = more trust in energy (more denoising)
        # Higher weight = more trust in observation (preserve details)
        x = denoise_with_ebm(energy_net, x, 
                            n_steps=50, 
                            data_weight=data_weight)
    
    return x
```

## Compositional Generation

### Combining Concepts

A powerful feature of EBMs is compositional generation by combining energies:

$$p(\mathbf{x} | \text{concept}_1, \text{concept}_2) \propto \exp(-E_1(\mathbf{x}) - E_2(\mathbf{x}))$$

```python
class CompositionalEBM:
    """
    Combine multiple EBMs for compositional generation.
    """
    
    def __init__(self, energy_nets: list, weights: list = None):
        """
        Parameters
        ----------
        energy_nets : list
            List of energy networks for different concepts
        weights : list
            Importance weights for each concept
        """
        self.energy_nets = energy_nets
        self.weights = weights or [1.0] * len(energy_nets)
    
    def combined_energy(self, x):
        """Compute weighted sum of energies."""
        total = 0
        for net, weight in zip(self.energy_nets, self.weights):
            total += weight * net(x)
        return total
    
    def sample(self, n_samples, shape, n_steps=200, 
               step_size=0.01, noise_scale=0.005):
        """
        Sample from combined distribution.
        """
        x = torch.randn(n_samples, *shape)
        
        for _ in range(n_steps):
            x.requires_grad_(True)
            
            # Combined energy gradient
            energy = self.combined_energy(x)
            grad = torch.autograd.grad(energy.sum(), x)[0]
            
            noise = torch.randn_like(x) * noise_scale
            x = x.detach() - step_size * grad + noise
            x = x.clamp(0, 1)
        
        return x


def compositional_generation_example():
    """
    Example: Generate images with multiple attributes.
    
    E.g., "red" + "car" = red car images
    """
    # Assume we have trained concept-specific EBMs
    # energy_red: low energy for red objects
    # energy_car: low energy for car images
    
    composer = CompositionalEBM(
        energy_nets=[energy_red, energy_car],
        weights=[1.0, 1.0]  # Equal importance
    )
    
    # Sample images that are both red AND cars
    samples = composer.sample(n_samples=16, shape=(3, 64, 64))
    
    return samples
```

### Negation

We can also negate concepts:

$$p(\mathbf{x} | \text{concept}_1, \neg\text{concept}_2) \propto \exp(-E_1(\mathbf{x}) + E_2(\mathbf{x}))$$

```python
def sample_with_negation(energy_positive, energy_negative, 
                         n_samples, shape, neg_weight=0.5):
    """
    Sample concept1 but NOT concept2.
    """
    x = torch.randn(n_samples, *shape)
    
    for _ in range(200):
        x.requires_grad_(True)
        
        # Want low E_pos, high E_neg
        combined = energy_positive(x) - neg_weight * energy_negative(x)
        grad = torch.autograd.grad(combined.sum(), x)[0]
        
        x = x.detach() - 0.01 * grad + 0.005 * torch.randn_like(x)
        x = x.clamp(0, 1)
    
    return x
```

## Adversarial Robustness

### EBMs as Robust Classifiers

Grathwohl et al. (2020) showed that classifiers can be viewed as EBMs, enabling adversarial purification:

```python
class JointEnergyClassifier(nn.Module):
    """
    Joint Energy-Based Model that combines classification and generation.
    
    p(x, y) ∝ exp(-E(x, y))
    p(y|x) = softmax(-E(x, y))
    """
    
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone  # Feature extractor
        self.energy_head = nn.Linear(backbone.output_dim, n_classes)
    
    def forward(self, x):
        """Return logits (negative class-conditional energies)."""
        features = self.backbone(x)
        return self.energy_head(features)
    
    def energy(self, x, y=None):
        """
        Compute energy.
        
        If y is None: marginal energy E(x) = -logsumexp(-E(x,y))
        If y is given: conditional energy E(x, y)
        """
        logits = self.forward(x)
        
        if y is None:
            # Marginal: -log Σ_y exp(-E(x,y)) = -logsumexp(logits)
            return -torch.logsumexp(logits, dim=1)
        else:
            # Conditional: -logits[y]
            return -logits.gather(1, y.unsqueeze(1)).squeeze(1)


def adversarial_purification(energy_net, adv_images, n_steps=50):
    """
    Purify adversarial examples via energy minimization.
    
    Idea: Push adversarial examples back to the data manifold.
    """
    x = adv_images.clone()
    
    for _ in range(n_steps):
        x.requires_grad_(True)
        energy = energy_net.energy(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        
        # Move toward lower energy (natural images)
        x = x.detach() - 0.01 * grad
        x = x.clamp(0, 1)
    
    return x
```

## Image Inpainting

### Energy-Based Inpainting

Fill in missing regions by minimizing energy while matching known pixels:

```python
def inpaint_with_ebm(energy_net, image, mask, n_steps=200):
    """
    Inpaint masked regions using EBM.
    
    Parameters
    ----------
    image : torch.Tensor
        Original image with missing regions
    mask : torch.Tensor
        Binary mask (1 = known, 0 = unknown)
    """
    # Initialize unknown regions with noise
    x = image.clone()
    x[mask == 0] = torch.rand_like(x[mask == 0])
    
    for _ in range(n_steps):
        x.requires_grad_(True)
        
        energy = energy_net(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        
        # Only update unknown regions
        update = -0.01 * grad + 0.005 * torch.randn_like(x)
        x = x.detach()
        x[mask == 0] = (x + update)[mask == 0]
        x = x.clamp(0, 1)
    
    return x
```

## Key Takeaways

!!! success "Application Summary"
    | Application | Method | Key Insight |
    |-------------|--------|-------------|
    | OOD Detection | Energy as score | Low energy = in-distribution |
    | Denoising | Energy + data fidelity | Balance naturalness and observation |
    | Composition | Add energies | Intersection of concepts |
    | Adversarial | Energy minimization | Purify to data manifold |
    | Inpainting | Constrained optimization | Match known, minimize energy |

!!! tip "Practical Considerations"
    - EBM quality directly affects application performance
    - Langevin step size and steps need tuning per application
    - Composition works best with well-calibrated energies
    - OOD detection may need threshold calibration

## Exercises

1. **OOD Benchmark**: Evaluate OOD detection on CIFAR-10 (in-dist) vs SVHN (OOD).

2. **Multi-Concept Composition**: Train separate EBMs for digit identity and style, then compose.

3. **Adversarial Defense**: Compare purification effectiveness against different attack types (FGSM, PGD, CW).

## References

- Grathwohl, W., et al. (2020). Your Classifier is Secretly an Energy Based Model. ICLR.
- Du, Y., Li, S., & Mordatch, I. (2020). Compositional Visual Generation with Energy Based Models. NeurIPS.
- Liu, W., et al. (2021). Energy-Based Out-of-Distribution Detection. NeurIPS.
