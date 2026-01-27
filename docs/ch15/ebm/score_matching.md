# Score Matching

## Learning Objectives

After completing this section, you will be able to:

1. Define the score function and understand its properties
2. Derive the score matching objective
3. Implement denoising score matching
4. Compare score matching with maximum likelihood
5. Connect score matching to diffusion models

## Introduction

Score matching, introduced by Aapo Hyvärinen in 2005, provides an elegant alternative to maximum likelihood for training energy-based models. The key insight is that we can match the gradients of log probability (the "score") without ever computing the intractable partition function. This approach has become foundational to modern score-based generative models and diffusion models.

## The Score Function

### Definition

The **score function** is the gradient of the log probability density:

$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

For an energy-based model:
$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$$

**Critical observation**: The partition function $Z$ disappears in the gradient!

### Geometric Interpretation

The score function points toward regions of higher probability:
- At a mode: $\mathbf{s}(\mathbf{x}^*) = 0$
- Away from modes: $\mathbf{s}(\mathbf{x})$ points toward the nearest mode
- Magnitude indicates steepness of the probability landscape

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def visualize_score_field():
    """
    Visualize the score function for a 2D distribution.
    """
    # Define a mixture of Gaussians energy
    def energy(x):
        """Energy for 2-component Gaussian mixture."""
        mean1 = torch.tensor([-2.0, 0.0])
        mean2 = torch.tensor([2.0, 0.0])
        
        dist1 = ((x - mean1) ** 2).sum(dim=-1)
        dist2 = ((x - mean2) ** 2).sum(dim=-1)
        
        # Mixture energy (negative log of mixture density)
        return -torch.log(torch.exp(-dist1/2) + torch.exp(-dist2/2) + 1e-10)
    
    # Create grid
    x = torch.linspace(-5, 5, 30)
    y = torch.linspace(-3, 3, 20)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    points.requires_grad_(True)
    
    # Compute energy and score
    E = energy(points)
    score = -torch.autograd.grad(E.sum(), points)[0]  # score = -∇E
    
    # Reshape for plotting
    U = score[:, 0].detach().numpy().reshape(30, 20)
    V = score[:, 1].detach().numpy().reshape(30, 20)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Energy contours
    E_grid = E.detach().numpy().reshape(30, 20)
    ax1 = axes[0]
    cont = ax1.contourf(X.numpy(), Y.numpy(), E_grid, levels=30, cmap='viridis')
    plt.colorbar(cont, ax=ax1, label='Energy')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Energy Landscape')
    
    # Score field (arrows)
    ax2 = axes[1]
    ax2.contourf(X.numpy(), Y.numpy(), E_grid, levels=30, cmap='viridis', alpha=0.3)
    
    # Subsample for cleaner arrows
    skip = 2
    ax2.quiver(X.numpy()[::skip, ::skip], Y.numpy()[::skip, ::skip],
               U[::skip, ::skip], V[::skip, ::skip],
               color='red', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Score Field (∇ log p)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    print("Interpretation:")
    print("• Arrows point toward higher probability regions")
    print("• Arrow magnitude indicates gradient strength")
    print("• At modes, arrows converge to zero")

visualize_score_field()
```

## Score Matching Objective

### The Fisher Divergence

Score matching minimizes the **Fisher divergence** between data and model:

$$J(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}}[\|\mathbf{s}_\theta(\mathbf{x}) - \mathbf{s}_{\text{data}}(\mathbf{x})\|^2]$$

where $\mathbf{s}_\theta = \nabla_{\mathbf{x}} \log p_\theta$ and $\mathbf{s}_{\text{data}} = \nabla_{\mathbf{x}} \log p_{\text{data}}$.

### The Problem

We don't have access to $\mathbf{s}_{\text{data}}$!

### Hyvärinen's Solution

Through integration by parts (under regularity conditions):

$$J(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}))\right] + C$$

The constant $C$ doesn't depend on $\theta$!

**The practical objective**:
$$J_{\text{SM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|\nabla_{\mathbf{x}} E_\theta(\mathbf{x})\|^2 + \nabla_{\mathbf{x}}^2 E_\theta(\mathbf{x})\right]$$

where $\nabla_{\mathbf{x}}^2 E = \text{tr}(\mathbf{H}_E)$ is the Laplacian (trace of Hessian).

## Implementation

```python
class EnergyNetwork(nn.Module):
    """
    Neural network energy function for score matching.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return energy E(x)."""
        return self.net(x).squeeze(-1)


def score_matching_loss(energy_net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the explicit score matching loss.
    
    L = ½ E[‖∇E‖²] + E[∇²E]
    
    Parameters
    ----------
    energy_net : nn.Module
        Energy function network
    x : torch.Tensor
        Data samples, shape (batch, dim)
    
    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    x = x.requires_grad_(True)
    
    # Compute energy
    energy = energy_net(x)
    
    # Compute score (gradient of energy)
    score = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # First term: ½ ‖∇E‖²
    score_norm_sq = (score ** 2).sum(dim=1)
    
    # Second term: trace of Hessian (Laplacian)
    laplacian = torch.zeros(x.shape[0], device=x.device)
    for i in range(x.shape[1]):
        # ∂²E/∂x_i²
        grad_i = torch.autograd.grad(
            outputs=score[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        laplacian += grad_i[:, i]
    
    # Total loss
    loss = 0.5 * score_norm_sq.mean() + laplacian.mean()
    
    return loss


def train_with_score_matching():
    """
    Train an energy-based model using score matching.
    """
    print("="*70)
    print("TRAINING WITH EXPLICIT SCORE MATCHING")
    print("="*70)
    
    # Generate 2D data from mixture of Gaussians
    n_samples = 5000
    
    def sample_mixture():
        """Sample from 3-component mixture."""
        means = torch.tensor([
            [-2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0]
        ])
        idx = torch.randint(0, 3, (n_samples,))
        samples = means[idx] + torch.randn(n_samples, 2) * 0.5
        return samples
    
    data = sample_mixture()
    print(f"Generated {n_samples} samples from 3-component mixture")
    
    # Create energy network
    energy_net = EnergyNetwork(input_dim=2, hidden_dim=64)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    # Training
    n_epochs = 500
    batch_size = 128
    losses = []
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = data[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = score_matching_loss(energy_net, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        losses.append(epoch_loss / n_batches)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")
    
    return energy_net, data, losses

energy_net, data, losses = train_with_score_matching()
```

## Denoising Score Matching

### The Challenge with Explicit Score Matching

Computing the Laplacian (trace of Hessian) is expensive:
- Requires $d$ backward passes for $d$ dimensions
- Scales poorly to high dimensions (e.g., images)

### The Denoising Score Matching Insight

Vincent (2011) showed an elegant equivalence. If we corrupt data with noise:
$$\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

Then the optimal score for the noisy distribution is:
$$\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}$$

### Denoising Score Matching Objective

$$J_{\text{DSM}}(\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[\|\mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\|^2\right]$$

**Key advantages**:
- No Hessian computation
- Single forward + backward pass
- Scalable to high dimensions

```python
def denoising_score_matching_loss(energy_net: nn.Module, 
                                   x: torch.Tensor, 
                                   noise_std: float = 0.5) -> torch.Tensor:
    """
    Denoising score matching loss.
    
    Much more efficient than explicit score matching!
    
    L = E[‖∇E(x̃) + ε/σ‖²]
    where x̃ = x + σε, ε ~ N(0, I)
    
    Parameters
    ----------
    energy_net : nn.Module
        Energy function network
    x : torch.Tensor
        Clean data samples
    noise_std : float
        Standard deviation of added noise
    """
    # Add noise
    noise = torch.randn_like(x)
    x_noisy = x + noise_std * noise
    x_noisy.requires_grad_(True)
    
    # Compute energy and its gradient (score)
    energy = energy_net(x_noisy)
    score = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x_noisy,
        create_graph=True
    )[0]
    
    # Target: the score should approximate -noise/σ
    target = -noise / noise_std
    
    # MSE loss
    loss = ((score - target) ** 2).sum(dim=1).mean()
    
    return loss


def train_with_dsm():
    """
    Train with denoising score matching.
    """
    print("\n" + "="*70)
    print("TRAINING WITH DENOISING SCORE MATCHING")
    print("="*70)
    
    # Same data
    n_samples = 5000
    means = torch.tensor([[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    idx = torch.randint(0, 3, (n_samples,))
    data = means[idx] + torch.randn(n_samples, 2) * 0.5
    
    # Train
    energy_net = EnergyNetwork(input_dim=2, hidden_dim=64)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    n_epochs = 1000
    batch_size = 128
    losses = []
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = data[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = denoising_score_matching_loss(energy_net, batch, noise_std=0.3)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        losses.append(epoch_loss / n_batches)
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")
    
    # Visualize learned energy
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('DSM Loss')
    axes[0].set_title('Training Progress')
    axes[0].grid(True, alpha=0.3)
    
    # Learned energy landscape
    x_grid = torch.linspace(-4, 4, 100)
    y_grid = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        energies = energy_net(grid).numpy().reshape(100, 100)
    
    axes[1].contourf(X.numpy(), Y.numpy(), energies, levels=30, cmap='viridis')
    axes[1].scatter(data[:500, 0], data[:500, 1], c='red', s=1, alpha=0.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Learned Energy Landscape')
    axes[1].set_aspect('equal')
    
    # Data histogram for comparison
    axes[2].hist2d(data[:, 0].numpy(), data[:, 1].numpy(), bins=50, cmap='viridis')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Data Distribution')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return energy_net

dsm_net = train_with_dsm()
```

## Multi-Scale Denoising Score Matching

### Motivation

A single noise level $\sigma$ may not capture structure at all scales:
- Small $\sigma$: captures fine details but struggles in low-density regions
- Large $\sigma$: good coverage but loses detail

### Solution: Multiple Noise Levels

Train with a range of noise levels $\{\sigma_1, \ldots, \sigma_L\}$:

$$J_{\text{MDSM}}(\theta) = \sum_{i=1}^{L} \lambda_i \mathbb{E}[\|\mathbf{s}_\theta(\mathbf{x} + \sigma_i \boldsymbol{\epsilon}, \sigma_i) + \frac{\boldsymbol{\epsilon}}{\sigma_i}\|^2]$$

This is the foundation of **Noise Conditional Score Networks (NCSN)** and **diffusion models**.

```python
class NoiseConditionalEnergyNetwork(nn.Module):
    """
    Energy network conditioned on noise level.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Embed noise level
        self.noise_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU()
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Noisy input, shape (batch, dim)
        sigma : torch.Tensor
            Noise level, shape (batch, 1)
        """
        # Embed noise level
        sigma_embed = self.noise_embed(sigma)
        
        # Concatenate and process
        combined = torch.cat([x, sigma_embed], dim=-1)
        return self.net(combined).squeeze(-1)


def multiscale_dsm_loss(energy_net: NoiseConditionalEnergyNetwork,
                        x: torch.Tensor,
                        sigma_min: float = 0.01,
                        sigma_max: float = 1.0,
                        n_levels: int = 10) -> torch.Tensor:
    """
    Multi-scale denoising score matching.
    """
    # Sample noise levels (geometric spacing)
    sigmas = torch.exp(
        torch.linspace(np.log(sigma_max), np.log(sigma_min), n_levels)
    )
    
    # Random noise level for each sample
    sigma_idx = torch.randint(0, n_levels, (x.shape[0],))
    sigma = sigmas[sigma_idx].view(-1, 1)
    
    # Add noise
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    x_noisy.requires_grad_(True)
    
    # Compute score
    energy = energy_net(x_noisy, sigma)
    score = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x_noisy,
        create_graph=True
    )[0]
    
    # Target and loss (weighted by sigma)
    target = -noise / sigma
    loss = ((score - target) ** 2 * sigma ** 2).sum(dim=1).mean()
    
    return loss
```

## Connection to Diffusion Models

### The Key Link

Score matching is the foundation of diffusion models:

1. **Forward process**: Gradually add noise to data
2. **Score learning**: Learn $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ at each noise level
3. **Reverse process**: Use learned scores for denoising

The score function tells us how to denoise!

### Langevin Dynamics Sampling

Given learned scores, we can sample via Langevin dynamics:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\epsilon}{2} \nabla_{\mathbf{x}} E(\mathbf{x}_t) + \sqrt{\epsilon} \boldsymbol{\xi}_t$$

where $\boldsymbol{\xi}_t \sim \mathcal{N}(0, \mathbf{I})$.

```python
def langevin_sampling(energy_net: nn.Module,
                      n_samples: int = 100,
                      n_steps: int = 1000,
                      step_size: float = 0.01,
                      noise_scale: float = 0.01,
                      dim: int = 2) -> torch.Tensor:
    """
    Sample from learned EBM using Langevin dynamics.
    
    x_{t+1} = x_t - ε/2 ∇E(x_t) + √ε ξ
    """
    # Initialize from random noise
    x = torch.randn(n_samples, dim) * 2
    
    for _ in range(n_steps):
        x.requires_grad_(True)
        
        # Compute gradient
        energy = energy_net(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        
        # Langevin update
        noise = torch.randn_like(x) * noise_scale
        x = x.detach() - step_size * grad + noise
    
    return x.detach()
```

## Key Takeaways

!!! success "Core Concepts"
    1. The score function $\nabla_{\mathbf{x}} \log p$ points toward higher probability
    2. Score matching avoids computing the partition function
    3. Denoising score matching is efficient and scalable
    4. Multi-scale DSM captures structure at all scales
    5. Score matching is the foundation of diffusion models

!!! info "Historical Impact"
    Score matching (2005) → NCSN (2019) → DDPM (2020) → Stable Diffusion (2022)
    
    This line of research revolutionized generative modeling!

## Exercises

1. **Explicit vs Denoising**: Compare the computational cost and accuracy of explicit score matching vs DSM on a 2D example.

2. **Noise Schedule Design**: Experiment with different noise level schedules (geometric, linear, cosine) for multi-scale DSM.

3. **Langevin Sampling**: Implement annealed Langevin dynamics that starts from high noise and gradually decreases.

## References

- Hyvärinen, A. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. JMLR.
- Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. Neural Computation.
- Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS.
- Song, Y., & Ermon, S. (2020). Improved Techniques for Training Score-Based Generative Models. NeurIPS.
