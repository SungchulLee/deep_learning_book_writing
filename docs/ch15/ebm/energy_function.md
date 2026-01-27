# Energy Function Formulation

## Learning Objectives

After completing this section, you will be able to:

1. Understand the fundamental relationship between energy and probability
2. Define and analyze energy functions in various dimensions
3. Connect energy-based modeling to statistical physics concepts
4. Implement energy functions in PyTorch for practical applications

## Introduction

Energy-Based Models (EBMs) represent one of the most elegant frameworks in machine learning, drawing deep connections to statistical physics and thermodynamics. At their core, EBMs learn to assign scalar energy values to configurations of variables, where low energy corresponds to high probability and high energy to low probability.

## The Energy Function

### Definition

An energy function $E: \mathcal{X} \rightarrow \mathbb{R}$ maps configurations $x \in \mathcal{X}$ to scalar energy values. The key insight is that this energy determines probability through the Boltzmann distribution:

$$p(x) = \frac{\exp(-E(x)/T)}{Z}$$

where:
- $E(x)$ is the energy of configuration $x$
- $T$ is the temperature parameter (often set to 1)
- $Z = \int \exp(-E(x)/T) dx$ is the partition function (normalization constant)

### Physical Intuition

The energy function captures the "naturalness" or "compatibility" of a configuration:

| Configuration Type | Energy Level | Probability |
|-------------------|--------------|-------------|
| Highly compatible | Low | High |
| Moderately compatible | Medium | Medium |
| Incompatible | High | Low |

This mirrors physical systems where systems naturally settle into low-energy states.

### Simple 1D Energy Function Example

Consider a double-well potential that creates a bimodal distribution:

$$E(x) = \frac{1}{2}(x^2 - 4)^2$$

This energy function has minima at $x = \pm 2$, creating two probability peaks.

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def simple_energy_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Double-well potential energy function.
    
    Creates energy minima at x = ±2, corresponding to 
    probability maxima in the Boltzmann distribution.
    
    Parameters
    ----------
    x : torch.Tensor
        Input values of any shape
    
    Returns
    -------
    torch.Tensor
        Energy values with same shape as input
    """
    return 0.5 * (x**2 - 4)**2

# Visualize energy landscape
x = torch.linspace(-4, 4, 1000)
energy = simple_energy_1d(x)

plt.figure(figsize=(10, 5))
plt.plot(x.numpy(), energy.numpy(), 'b-', linewidth=2)
plt.fill_between(x.numpy(), energy.numpy(), alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('Energy E(x)', fontsize=12)
plt.title('Double-Well Energy Function', fontsize=14)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.scatter([-2, 2], [0, 0], color='red', s=100, zorder=5, 
            label='Energy minima')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## From Energy to Probability

### The Boltzmann Distribution

The probability distribution induced by an energy function is:

$$p(x) = \frac{1}{Z} \exp\left(-\frac{E(x)}{T}\right)$$

Key properties:
1. **Normalization**: $\int p(x) dx = 1$ (ensured by $Z$)
2. **Monotonicity**: Lower energy → Higher probability
3. **Temperature sensitivity**: $T$ controls distribution sharpness

```python
def boltzmann_distribution(x: torch.Tensor, 
                           energy_fn, 
                           temperature: float = 1.0) -> torch.Tensor:
    """
    Compute Boltzmann distribution from energy function.
    
    Uses log-sum-exp trick for numerical stability.
    
    Parameters
    ----------
    x : torch.Tensor
        Points at which to evaluate probability
    energy_fn : callable
        Energy function E(x)
    temperature : float
        Temperature parameter (default 1.0)
    
    Returns
    -------
    torch.Tensor
        Normalized probability density
    """
    energy = energy_fn(x)
    
    # Unnormalized log probability
    log_unnorm = -energy / temperature
    
    # Numerical integration for partition function
    dx = x[1] - x[0]
    log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
    
    # Normalized probability
    log_prob = log_unnorm - log_Z
    return torch.exp(log_prob)

# Compare distributions at different temperatures
temperatures = [0.5, 1.0, 2.0]
x = torch.linspace(-4, 4, 1000)

plt.figure(figsize=(12, 5))
for T in temperatures:
    prob = boltzmann_distribution(x, simple_energy_1d, T)
    plt.plot(x.numpy(), prob.numpy(), linewidth=2, label=f'T = {T}')

plt.xlabel('x', fontsize=12)
plt.ylabel('p(x)', fontsize=12)
plt.title('Temperature Effects on Boltzmann Distribution', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
```

### Temperature Interpretation

The temperature parameter $T$ controls the "sharpness" of the distribution:

| Temperature | Distribution Behavior | Physical Analogy |
|------------|----------------------|------------------|
| $T \to 0$ | Concentrates at global minimum | Frozen state |
| $T = 1$ | Standard Boltzmann distribution | Room temperature |
| $T \to \infty$ | Approaches uniform distribution | Boiling/melting |

**Mathematical insight**: As $T \to 0$, $p(x) \to \delta(x - x^*)$ where $x^* = \arg\min E(x)$.

## 2D Energy Landscapes

### Multivariate Energy Functions

For higher-dimensional data, energy functions create complex landscapes with multiple modes:

```python
def energy_2d_mixture(xy: torch.Tensor) -> torch.Tensor:
    """
    2D energy function from Gaussian mixture.
    
    Creates three energy wells corresponding to three modes.
    
    Parameters
    ----------
    xy : torch.Tensor
        Shape (N, 2) tensor of 2D points
    
    Returns
    -------
    torch.Tensor
        Shape (N,) tensor of energy values
    """
    # Mixture component parameters
    means = torch.tensor([
        [-2.0, -2.0],
        [2.0, 2.0],
        [-2.0, 2.0]
    ])
    weights = torch.tensor([0.4, 0.4, 0.2])
    variances = torch.tensor([0.5, 0.3, 0.6])
    
    # Compute mixture density
    density = torch.zeros(xy.shape[0])
    for mean, weight, var in zip(means, weights, variances):
        diff = xy - mean
        mahal = (diff ** 2).sum(dim=1) / var
        density += weight * torch.exp(-0.5 * mahal)
    
    # Energy is negative log density
    return -torch.log(density + 1e-10)

# Visualize 2D energy landscape
x = torch.linspace(-4, 4, 100)
y = torch.linspace(-4, 4, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')
grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

energy = energy_2d_mixture(grid).reshape(100, 100)

plt.figure(figsize=(10, 8))
plt.contourf(X.numpy(), Y.numpy(), energy.numpy(), levels=30, cmap='viridis')
plt.colorbar(label='Energy E(x,y)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('2D Energy Landscape', fontsize=14)
plt.scatter([-2, 2, -2], [-2, 2, 2], c='red', s=100, 
            marker='*', edgecolors='white', linewidths=2,
            label='Energy minima')
plt.legend()
plt.show()
```

## Properties of Energy Functions

### Desirable Properties

For an effective EBM, the energy function should:

1. **Smoothness**: $E(x)$ should be differentiable almost everywhere
2. **Boundedness**: Energy should be bounded below to ensure integrability
3. **Expressiveness**: Must capture complex data distributions
4. **Computability**: Gradients $\nabla_x E(x)$ must be tractable

### Parameterized Energy Functions

In practice, we parameterize the energy function with learnable parameters $\theta$:

$$E_\theta(x) = f_\theta(x)$$

where $f_\theta$ can be a neural network. This allows learning from data.

```python
import torch.nn as nn

class SimpleEnergyNetwork(nn.Module):
    """
    Neural network parameterized energy function.
    
    Architecture: x → hidden layers → scalar energy
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Smooth activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy for input configurations."""
        return self.net(x).squeeze(-1)

# Example usage
energy_net = SimpleEnergyNetwork(input_dim=2)
sample_points = torch.randn(100, 2)
energies = energy_net(sample_points)
print(f"Energy shape: {energies.shape}")  # (100,)
```

## Connection to Statistical Physics

### Free Energy and Entropy

The partition function connects to thermodynamic quantities:

$$F = -T \log Z$$

where $F$ is the Helmholtz free energy. The entropy is:

$$S = -\sum_x p(x) \log p(x) = \frac{\langle E \rangle - F}{T}$$

### Boltzmann's Insight

Ludwig Boltzmann's fundamental insight was that macroscopic thermodynamic properties emerge from microscopic probability distributions. In machine learning, this translates to:

- **Microscopic**: Individual data points $x$
- **Energy**: Learned compatibility function $E_\theta(x)$
- **Macroscopic**: Distribution over data space $p_\theta(x)$

## Key Takeaways

!!! success "Core Concepts"
    1. Energy functions map configurations to scalar values
    2. The Boltzmann distribution converts energy to probability: $p(x) \propto \exp(-E(x))$
    3. Lower energy = Higher probability
    4. Temperature controls distribution sharpness
    5. Neural networks provide flexible parameterization

!!! warning "Common Pitfalls"
    - The partition function $Z$ is typically intractable to compute exactly
    - Energy functions must be bounded below for valid probability distributions
    - Numerical stability requires careful implementation (log-sum-exp tricks)

## Exercises

1. **Temperature Analysis**: Implement a function that computes the variance of the Boltzmann distribution as a function of temperature. Plot variance vs. temperature for the double-well potential.

2. **3D Energy Landscape**: Extend the 2D mixture energy to 3D and visualize isosurfaces at different energy levels.

3. **Neural Energy Function**: Train a neural network energy function to model a given dataset using score matching (covered in later sections).

## References

- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence.
- Song, Y., & Kingma, D. P. (2021). How to train your energy-based models.
