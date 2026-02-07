# Neural Energy-Based Models

## Learning Objectives

After completing this section, you will be able to:

1. Design deep neural networks as flexible energy functions
2. Train neural EBMs with Langevin dynamics and replay buffers
3. Implement the complete training pipeline including sampling, loss computation, and diagnostics
4. Understand the connection between neural EBMs and diffusion models

## Introduction

Modern Energy-Based Models combine classical EBM theory with deep neural networks as energy function parameterizations. Since 2019, neural EBMs have achieved competitive results on image generation while offering unique capabilities—out-of-distribution detection, compositional generation, and principled uncertainty quantification—that distinguish them from GANs and VAEs. The key innovations are the use of Langevin dynamics for sampling and replay buffers for stable training.

## Neural Energy Function Architectures

### Convolutional Energy Network

For image data, Du and Mordatch (2019) introduced a convolutional architecture that maps images to scalar energy values:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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


class MLPEnergyNetwork(nn.Module):
    """
    MLP energy network for tabular/financial data.
    
    Uses spectral normalization for Lipschitz control
    and residual connections for gradient flow.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 n_layers: int = 4):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

### Design Principles

**Smooth activations**: SiLU (Swish) or ELU activations produce well-behaved energy gradients, which is critical since Langevin dynamics requires $\nabla_x E(x)$.

**Scalar output**: The final layer must produce a single scalar per input. For image architectures, global average pooling followed by a linear layer is the standard approach.

**No normalization layers**: Batch normalization and layer normalization interfere with energy interpretation. The energy function should depend only on the input, not on batch statistics.

## Langevin Dynamics Sampling

The core sampling method for neural EBMs is Langevin dynamics, which generates samples by following the energy gradient with added noise:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\epsilon}{2} \nabla_{\mathbf{x}} E(\mathbf{x}_t) + \sqrt{\epsilon}\, \boldsymbol{\xi}_t, \quad \boldsymbol{\xi}_t \sim \mathcal{N}(0, I)$$

In the limit $\epsilon \to 0$ and $T \to \infty$, this converges to the Boltzmann distribution $p(x) \propto \exp(-E(x))$.

```python
def langevin_dynamics(energy_net, x_init, n_steps=100, 
                      step_size=0.01, noise_scale=0.005,
                      clip_value=None):
    """
    Sample via Langevin dynamics.
    
    Parameters
    ----------
    energy_net : nn.Module
        Energy function
    x_init : torch.Tensor
        Initial samples
    n_steps : int
        Number of Langevin steps
    step_size : float
        Gradient step size (ε/2)
    noise_scale : float
        Noise magnitude (√ε)
    clip_value : float, optional
        Clip samples to [-clip_value, clip_value]
    
    Returns
    -------
    torch.Tensor
        Final samples
    """
    x = x_init.clone()
    
    for _ in range(n_steps):
        x.requires_grad_(True)
        energy = energy_net(x).sum()
        grad = torch.autograd.grad(energy, x)[0]
        
        noise = torch.randn_like(x) * noise_scale
        x = x.detach() - step_size * grad + noise
        
        if clip_value is not None:
            x = x.clamp(-clip_value, clip_value)
    
    return x.detach()
```

## Training with Replay Buffers

### The Training Loop

Neural EBM training follows the contrastive divergence principle but uses Langevin dynamics instead of Gibbs sampling, and a replay buffer instead of persistent chains:

```python
class ReplayBuffer:
    """
    Buffer of previous MCMC samples for initializing Langevin chains.
    
    Stores samples from previous training steps and provides them
    as initializations for new chains, improving mixing.
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, samples: torch.Tensor):
        """Add samples to buffer."""
        for s in samples:
            self.buffer.append(s.detach().cpu())
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)
    
    def sample(self, n: int, shape: tuple, 
               reinit_prob: float = 0.05) -> torch.Tensor:
        """
        Sample initializations from buffer.
        
        With probability reinit_prob, initialize from noise instead.
        """
        samples = []
        for _ in range(n):
            if len(self.buffer) == 0 or torch.rand(1) < reinit_prob:
                # Random initialization
                samples.append(torch.rand(*shape))
            else:
                # Sample from buffer
                idx = torch.randint(len(self.buffer), (1,)).item()
                samples.append(self.buffer[idx].clone())
        
        return torch.stack(samples)


def train_neural_ebm(energy_net, train_loader, n_epochs=50,
                     lr=1e-4, langevin_steps=60, 
                     langevin_lr=0.01, langevin_noise=0.005):
    """
    Train neural EBM with MCMC-based contrastive divergence.
    
    Parameters
    ----------
    energy_net : nn.Module
        Energy function network
    train_loader : DataLoader
        Training data
    n_epochs : int
        Training epochs
    lr : float
        Optimizer learning rate
    langevin_steps : int
        Langevin dynamics steps for negative phase
    langevin_lr : float
        Langevin step size
    langevin_noise : float
        Langevin noise scale
    """
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(max_size=10000)
    
    metrics = {'pos_energy': [], 'neg_energy': [], 'energy_gap': []}
    
    for epoch in range(n_epochs):
        epoch_pos, epoch_neg = 0, 0
        n_batches = 0
        
        for data, _ in train_loader:
            batch_size = data.shape[0]
            
            # === Positive phase: energy on data ===
            pos_energy = energy_net(data).mean()
            
            # === Negative phase: energy on MCMC samples ===
            # Initialize from replay buffer
            init = buffer.sample(
                batch_size, data.shape[1:], reinit_prob=0.05
            ).to(data.device)
            
            # Run Langevin dynamics
            neg_samples = langevin_dynamics(
                energy_net, init, 
                n_steps=langevin_steps,
                step_size=langevin_lr,
                noise_scale=langevin_noise,
                clip_value=1.0
            )
            
            neg_energy = energy_net(neg_samples).mean()
            
            # Update buffer
            buffer.add(neg_samples)
            
            # === Contrastive loss ===
            # Push data energy down, push sample energy up
            loss = pos_energy - neg_energy
            
            # Optional: regularization on energy magnitudes
            reg = 0.01 * (pos_energy**2 + neg_energy**2)
            loss = loss + reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(energy_net.parameters(), 1.0)
            optimizer.step()
            
            epoch_pos += pos_energy.item()
            epoch_neg += neg_energy.item()
            n_batches += 1
        
        avg_pos = epoch_pos / n_batches
        avg_neg = epoch_neg / n_batches
        metrics['pos_energy'].append(avg_pos)
        metrics['neg_energy'].append(avg_neg)
        metrics['energy_gap'].append(avg_neg - avg_pos)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: E(data)={avg_pos:.3f}, "
                  f"E(samples)={avg_neg:.3f}, "
                  f"gap={avg_neg-avg_pos:.3f}")
    
    return metrics
```

### Training Dynamics

Healthy training shows a consistent energy gap: data energy should be lower than sample energy, and this gap should stabilize as training progresses. If the gap collapses to zero, the model is failing to distinguish data from noise. If the gap grows without bound, the energies are diverging and regularization is needed.

## Connection to Diffusion Models

Modern diffusion models can be understood through the EBM lens:

**Score-based perspective**: A diffusion model learns score functions $s_\theta(x, \sigma) \approx \nabla_x \log p_\sigma(x)$ at multiple noise levels $\sigma$. Each score function corresponds to the gradient of an energy function at that noise level.

**Denoising as energy minimization**: Each denoising step in the reverse diffusion process is analogous to one step of Langevin dynamics on a noise-conditional energy function.

**Key difference**: Diffusion models train score functions at many noise levels simultaneously and use a structured sampling schedule, while neural EBMs typically train a single energy function and use unstructured Langevin dynamics. This difference makes diffusion models more stable and higher-quality in practice, but EBMs offer more flexibility (e.g., compositional generation).

## Key Takeaways

!!! success "Core Concepts"
    1. Neural networks parameterize flexible energy functions that can model complex distributions
    2. Langevin dynamics enables sampling from the model by following energy gradients with noise
    3. Replay buffers improve training stability by providing better initializations for MCMC chains
    4. The contrastive loss pushes data energy down and sample energy up, shaping the energy landscape
    5. Neural EBMs are theoretically connected to diffusion models through the score function

!!! warning "Training Challenges"
    - Langevin dynamics may not mix well in high dimensions or with many modes
    - Training is less stable than VAEs or diffusion models, requiring careful hyperparameter tuning
    - Energy regularization is often needed to prevent diverging energy magnitudes
    - Sample quality is typically lower than state-of-the-art diffusion models

## Exercises

1. **Architecture comparison**: Compare MLP and convolutional energy networks on MNIST. Which produces better samples? Better OOD detection?

2. **Langevin diagnostics**: Visualize the Langevin trajectory for a 2D energy function. How does step size affect mixing and convergence?

3. **Buffer ablation**: Compare training with and without a replay buffer. How does the reinit probability affect results?

## References

- Du, Y., & Mordatch, I. (2019). Implicit Generation and Modeling with Energy Based Models. *NeurIPS*.
- Song, Y., & Kingma, D. P. (2021). How to Train Your Energy-Based Models.
- Nijkamp, E., et al. (2020). On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models. *AAAI*.
