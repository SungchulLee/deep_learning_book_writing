# Modern Neural Energy-Based Models

## Learning Objectives

After completing this section, you will be able to:

1. Design deep neural networks as energy functions
2. Train neural EBMs with MCMC-based methods
3. Implement Langevin dynamics for sampling and training
4. Apply modern EBM architectures to image generation
5. Understand connections to diffusion models

## Introduction

Modern Energy-Based Models combine classical EBM theory with deep neural networks. Since 2019, neural EBMs have achieved competitive results on image generation while offering unique capabilities like out-of-distribution detection and compositional generation.

## Neural Energy Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEnergyNetwork(nn.Module):
    """
    Convolutional energy network for images.
    Based on Du & Mordatch (2019).
    """
    
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, 1)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(-1)
```

## Langevin Dynamics Sampling

The core sampling method for neural EBMs:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\epsilon}{2} \nabla_{\mathbf{x}} E(\mathbf{x}_t) + \sqrt{\epsilon} \boldsymbol{\xi}_t$$

```python
def langevin_dynamics(energy_net, x_init, n_steps=100, 
                      step_size=0.01, noise_scale=0.005):
    """
    Sample via Langevin dynamics.
    """
    x = x_init.clone()
    
    for _ in range(n_steps):
        x.requires_grad_(True)
        energy = energy_net(x).sum()
        grad = torch.autograd.grad(energy, x)[0]
        
        noise = torch.randn_like(x) * noise_scale
        x = x.detach() - step_size * grad + noise
        x = x.clamp(0, 1)  # For images
    
    return x
```

## Training with Contrastive Divergence

```python
def train_neural_ebm(energy_net, train_loader, n_epochs=10):
    """Train neural EBM with MCMC-based contrastive divergence."""
    
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=1e-4)
    buffer = []  # Replay buffer
    
    for epoch in range(n_epochs):
        for data, _ in train_loader:
            # Positive phase: energy on data
            pos_energy = energy_net(data).mean()
            
            # Negative phase: energy on MCMC samples
            if buffer and torch.rand(1) > 0.05:
                init = buffer[torch.randint(len(buffer), (data.shape[0],))]
            else:
                init = torch.rand_like(data)
            
            neg_samples = langevin_dynamics(energy_net, init, n_steps=60)
            neg_energy = energy_net(neg_samples).mean()
            
            # Update buffer
            buffer.extend([s.detach() for s in neg_samples])
            buffer = buffer[-10000:]
            
            # Contrastive loss
            loss = pos_energy - neg_energy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Applications

### Out-of-Distribution Detection

EBMs naturally provide anomaly scores via energy:
- In-distribution data → Low energy
- Out-of-distribution data → High energy

### Compositional Generation

Combine multiple concepts by adding energies:
$$E_{\text{combined}}(\mathbf{x}) = E_{\text{concept1}}(\mathbf{x}) + E_{\text{concept2}}(\mathbf{x})$$

### Image Denoising

Use energy minimization to remove noise while preserving structure.

## Connection to Diffusion Models

Modern diffusion models can be viewed as:
- Learning score functions at multiple noise levels
- Using learned scores for iterative denoising
- EBMs provide the theoretical foundation

## Key Takeaways

!!! success "Core Concepts"
    1. Neural networks parameterize flexible energy functions
    2. Langevin dynamics enables sampling from learned distributions
    3. Contrastive training balances data vs model energies
    4. Replay buffers improve training stability
    5. Deep connections exist to score-based and diffusion models

## References

- Du, Y., & Mordatch, I. (2019). Implicit Generation and Modeling with Energy Based Models. NeurIPS.
- Grathwohl, W., et al. (2020). Your Classifier is Secretly an Energy Based Model. ICLR.
- Song, Y., & Kingma, D. P. (2021). How to Train Your Energy-Based Models.
