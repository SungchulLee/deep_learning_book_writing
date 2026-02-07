# EBM for Classification and Beyond

## Learning Objectives

After completing this section, you will be able to:

1. Apply EBMs for out-of-distribution detection using energy scores
2. Implement compositional generation by combining energy functions
3. Use energy minimization for image denoising and inpainting
4. Understand the practical deployment considerations for EBM-based systems

## Out-of-Distribution Detection

### Energy as Anomaly Score

EBMs naturally provide anomaly scores: data the model has learned should have low energy, while novel or anomalous data should have high energy. This makes EBMs particularly attractive for safety-critical applications where detecting inputs outside the training distribution is essential.

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
        in_dist_energies = energy_net(in_distribution_data).numpy()
        ood_energies = energy_net(ood_data).numpy()
    
    labels = np.concatenate([
        np.zeros(len(in_dist_energies)),
        np.ones(len(ood_energies))
    ])
    scores = np.concatenate([in_dist_energies, ood_energies])
    
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
        """Set threshold based on in-distribution energy quantile."""
        with torch.no_grad():
            energies = self.energy_net(in_dist_data).numpy()
        self.threshold = np.percentile(energies, percentile)
        return self.threshold
    
    def predict(self, x):
        """Predict OOD (1) or in-distribution (0)."""
        with torch.no_grad():
            energies = self.energy_net(x).numpy()
        return (energies > self.threshold).astype(int)
    
    def score(self, x):
        """Return OOD scores (higher = more likely OOD)."""
        with torch.no_grad():
            return self.energy_net(x).numpy()
```

## Compositional Generation

### Combining Energy Functions

One of the most powerful features of EBMs is compositional generation—combining multiple independently trained energy functions to generate samples satisfying multiple constraints simultaneously:

$$p(\mathbf{x} | \text{concept}_1, \text{concept}_2) \propto \exp(-E_1(\mathbf{x}) - E_2(\mathbf{x}))$$

Since energies add, the combined distribution places high probability only where all individual distributions agree—the intersection of concepts.

```python
class CompositionalEBM:
    """
    Combine multiple EBMs for compositional generation.
    """
    
    def __init__(self, energy_nets: list, weights: list = None):
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
        """Sample from combined distribution via Langevin dynamics."""
        x = torch.randn(n_samples, *shape)
        
        for _ in range(n_steps):
            x.requires_grad_(True)
            energy = self.combined_energy(x)
            grad = torch.autograd.grad(energy.sum(), x)[0]
            
            noise = torch.randn_like(x) * noise_scale
            x = x.detach() - step_size * grad + noise
            x = x.clamp(0, 1)
        
        return x
```

### Negation

We can also negate concepts—generating samples that satisfy one constraint but not another:

$$p(\mathbf{x} | \text{concept}_1, \neg\text{concept}_2) \propto \exp(-E_1(\mathbf{x}) + \alpha \cdot E_2(\mathbf{x}))$$

```python
def sample_with_negation(energy_positive, energy_negative, 
                         n_samples, shape, neg_weight=0.5):
    """
    Sample concept1 but NOT concept2.
    """
    x = torch.randn(n_samples, *shape)
    
    for _ in range(200):
        x.requires_grad_(True)
        combined = energy_positive(x) - neg_weight * energy_negative(x)
        grad = torch.autograd.grad(combined.sum(), x)[0]
        
        x = x.detach() - 0.01 * grad + 0.005 * torch.randn_like(x)
        x = x.clamp(0, 1)
    
    return x
```

## Image Denoising

### Energy Minimization for Denoising

Given a noisy observation $\mathbf{y} = \mathbf{x} + \boldsymbol{\epsilon}$, we can recover the clean image by minimizing a combined objective that balances the energy prior (encouraging natural images) with data fidelity (staying close to the observation):

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \left[ E_\theta(\mathbf{x}) + \lambda \|\mathbf{x} - \mathbf{y}\|^2 \right]$$

```python
def denoise_with_ebm(energy_net, noisy_image, n_steps=100, 
                     step_size=0.01, data_weight=10.0):
    """
    Denoise image via energy minimization with data fidelity.
    """
    x = noisy_image.clone().requires_grad_(True)
    
    for _ in range(n_steps):
        energy = energy_net(x)
        energy_grad = torch.autograd.grad(energy.sum(), x)[0]
        data_grad = 2 * data_weight * (x - noisy_image)
        
        total_grad = energy_grad + data_grad
        x = x.detach() - step_size * total_grad
        x = x.clamp(0, 1)
        x.requires_grad_(True)
    
    return x.detach()


def progressive_denoising(energy_net, noisy_image, 
                          noise_levels=[1.0, 0.5, 0.25, 0.1]):
    """
    Multi-scale denoising starting from high noise tolerance.
    
    Lower data_weight = more trust in energy (aggressive denoising)
    Higher data_weight = more trust in observation (preserve details)
    """
    x = noisy_image.clone()
    
    for data_weight in noise_levels:
        x = denoise_with_ebm(energy_net, x, 
                            n_steps=50, 
                            data_weight=data_weight)
    return x
```

## Image Inpainting

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
    x = image.clone()
    x[mask == 0] = torch.rand_like(x[mask == 0])
    
    for _ in range(n_steps):
        x.requires_grad_(True)
        
        energy = energy_net(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        
        update = -0.01 * grad + 0.005 * torch.randn_like(x)
        x = x.detach()
        x[mask == 0] = (x + update)[mask == 0]
        x = x.clamp(0, 1)
    
    return x
```

## Application Summary

| Application | Method | Key Insight |
|-------------|--------|-------------|
| OOD Detection | Energy as score | Low energy = in-distribution |
| Denoising | Energy + data fidelity | Balance naturalness and observation |
| Composition | Add energies | Intersection of concepts |
| Negation | Subtract energy | Exclusion of concepts |
| Adversarial Defense | Energy minimization | Purify to data manifold |
| Inpainting | Constrained optimization | Match known pixels, minimize energy elsewhere |

## Key Takeaways

!!! success "Core Concepts"
    1. EBMs provide natural anomaly scores via energy, enabling OOD detection without additional training
    2. Compositional generation via energy addition allows modular, reusable concept models
    3. Denoising and inpainting are naturally formulated as constrained energy minimization
    4. All applications rely on the same core mechanism: Langevin dynamics on the energy landscape
    5. The quality of all downstream applications depends directly on the quality of the trained energy function

!!! tip "Practical Considerations"
    - EBM quality directly affects application performance—invest in good training
    - Langevin step size and number of steps need tuning per application
    - Composition works best with well-calibrated energies (similar magnitude scales)
    - OOD detection may need threshold calibration on a validation set

## Exercises

1. **OOD benchmark**: Train an EBM on one dataset and evaluate OOD detection against multiple OOD sources. How does performance vary with the degree of distribution shift?

2. **Multi-concept composition**: Train separate EBMs for different attributes (e.g., digit identity and stroke thickness on MNIST). Generate samples at the intersection of specific attribute values.

3. **Adversarial defense**: Compare energy-based purification effectiveness against different attack types (FGSM, PGD, CW). How many Langevin steps are needed for effective purification?

## References

- Grathwohl, W., et al. (2020). Your Classifier is Secretly an Energy Based Model. *ICLR*.
- Du, Y., Li, S., & Mordatch, I. (2020). Compositional Visual Generation with Energy Based Models. *NeurIPS*.
- Liu, W., et al. (2021). Energy-Based Out-of-Distribution Detection. *NeurIPS*.
