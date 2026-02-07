# Energy Functions

## Learning Objectives

After completing this section, you will be able to:

1. Define energy functions formally and analyze their properties
2. Implement energy functions in 1D and 2D with PyTorch
3. Convert energy functions to probability distributions via the Boltzmann distribution
4. Parameterize energy functions with neural networks for learning from data

## Definition and Properties

An energy function $E: \mathcal{X} \rightarrow \mathbb{R}$ maps configurations $x \in \mathcal{X}$ to scalar energy values. The energy determines probability through the Boltzmann distribution:

$$p(x) = \frac{\exp(-E(x)/T)}{Z}$$

where $E(x)$ is the energy of configuration $x$, $T$ is the temperature parameter (often set to 1), and $Z = \int \exp(-E(x)/T)\,dx$ is the partition function.

### Desirable Properties

For an effective energy-based model, the energy function should satisfy:

**Smoothness**: $E(x)$ should be differentiable almost everywhere to enable gradient-based sampling and training. Non-differentiable energy functions can still define valid distributions but make optimization and MCMC sampling significantly harder.

**Boundedness below**: The energy must satisfy $E(x) \geq E_{\min} > -\infty$ to ensure the partition function $Z$ converges. An energy function unbounded below would assign infinite probability mass to its infimum, producing an improper distribution.

**Expressiveness**: The energy function must be flexible enough to capture the complexity of the target distribution. A quadratic energy can only represent Gaussian distributions; capturing multimodality, heavy tails, or complex correlations requires more expressive parameterizations.

**Computational tractability**: Gradients $\nabla_x E(x)$ must be efficiently computable, since both sampling (Langevin dynamics) and several training methods (score matching) rely on energy gradients.

## 1D Energy Functions

### Double-Well Potential

Consider a double-well potential that creates a bimodal distribution:

$$E(x) = \frac{1}{2}(x^2 - 4)^2$$

This energy function has minima at $x = \pm 2$ (where $E = 0$) and a local maximum at $x = 0$ (where $E = 8$), creating two probability peaks separated by an energy barrier.

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

Key properties of this mapping:

1. **Normalization**: $\int p(x)\,dx = 1$ is ensured by the partition function $Z$
2. **Monotonicity**: Lower energy always corresponds to higher probability
3. **Temperature sensitivity**: $T$ controls the sharpness of the distribution

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

Mathematically, as $T \to 0$, $p(x) \to \delta(x - x^*)$ where $x^* = \arg\min E(x)$. This limiting behavior connects energy minimization (an optimization problem) to maximum a posteriori inference (a probabilistic problem).

## 2D Energy Landscapes

For higher-dimensional data, energy functions create complex landscapes with multiple modes. Visualizing these landscapes builds intuition for the challenges of sampling and training in high dimensions.

### Gaussian Mixture Energy

A natural way to construct multimodal energy functions is through Gaussian mixtures. Given mixture components with means $\mu_k$, weights $\pi_k$, and variances $\sigma_k^2$, the energy is the negative log of the mixture density:

$$E(x) = -\log \sum_k \pi_k \, \mathcal{N}(x; \mu_k, \sigma_k^2 I)$$

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

## Parameterized Energy Functions

In practice, we parameterize the energy function with learnable parameters $\theta$ so that it can be trained from data:

$$E_\theta(x) = f_\theta(x)$$

where $f_\theta$ can be any neural network that outputs a scalar. The key design choice is that the network's output is unconstrained—it does not need to satisfy any normalization or positivity constraints, unlike density models that must output valid probabilities.

```python
import torch.nn as nn

class SimpleEnergyNetwork(nn.Module):
    """
    Neural network parameterized energy function.
    
    Architecture: x → hidden layers → scalar energy
    
    The network outputs a single scalar for each input,
    interpreted as the energy of that configuration.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Smooth activation for well-behaved gradients
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

### Architecture Considerations

The choice of neural architecture for $f_\theta$ depends on the data domain:

**Tabular/financial data**: Fully-connected networks with smooth activations (SiLU, ELU) work well. Smooth activations produce well-behaved energy gradients, which is important for Langevin dynamics sampling.

**Image data**: Convolutional architectures with progressive downsampling, as introduced by Du and Mordatch (2019). The final layer typically uses global average pooling followed by a linear projection to a scalar.

**Sequential data**: Transformer or recurrent architectures that process variable-length sequences and output a scalar energy. For financial time series, attention-based architectures can capture long-range temporal dependencies in the energy function.

## Connection to Statistical Physics

### Free Energy and Entropy

The partition function connects to fundamental thermodynamic quantities:

$$F = -T \log Z$$

where $F$ is the Helmholtz free energy. The entropy is:

$$S = -\sum_x p(x) \log p(x) = \frac{\langle E \rangle - F}{T}$$

This decomposition—free energy equals average energy minus temperature times entropy—has a direct analog in machine learning: the evidence lower bound (ELBO) in variational inference decomposes into a reconstruction term (energy) and a KL divergence term (entropy).

### Boltzmann's Insight for Machine Learning

Ludwig Boltzmann's fundamental insight was that macroscopic thermodynamic properties emerge from microscopic probability distributions. In machine learning, this translates to:

- **Microscopic**: Individual data points $x$
- **Energy**: Learned compatibility function $E_\theta(x)$
- **Macroscopic**: Distribution over data space $p_\theta(x)$

The partition function $Z$ plays the role of connecting these scales, encoding all distributional information in a single quantity—just as in physics, where derivatives of $\log Z$ yield all thermodynamic observables.

## Key Takeaways

!!! success "Core Concepts"
    1. Energy functions map configurations to scalar values, with lower energy indicating higher probability
    2. The Boltzmann distribution $p(x) \propto \exp(-E(x)/T)$ converts energy to probability
    3. Temperature $T$ controls distribution sharpness: low $T$ concentrates at minima, high $T$ spreads uniformly
    4. Neural networks provide flexible, learnable parameterizations of energy functions
    5. The partition function $Z$ connects energy functions to normalized distributions but is typically intractable

!!! warning "Common Pitfalls"
    - The partition function $Z$ is typically intractable to compute exactly—never assume you can evaluate likelihoods directly
    - Energy functions must be bounded below for valid probability distributions
    - Numerical stability requires careful implementation (log-sum-exp tricks for partition function estimates)
    - Smooth activations (SiLU, ELU) are preferred over ReLU for energy networks because sampling methods require well-behaved gradients

## Exercises

1. **Temperature analysis**: Implement a function that computes the variance of the Boltzmann distribution as a function of temperature for the double-well potential $E(x) = \frac{1}{2}(x^2-4)^2$. Plot variance vs. temperature and identify the temperature at which variance grows most rapidly.

2. **3D energy landscape**: Extend the 2D mixture energy to 3D and visualize isosurfaces at different energy levels using `matplotlib`'s 3D plotting or `plotly`.

3. **Neural energy function**: Design a neural energy network for 10-dimensional input. Initialize it randomly and visualize how its energy landscape changes during the first few steps of training with score matching (the training procedure is covered in Section 26.3).

## References

- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence.
- Song, Y., & Kingma, D. P. (2021). How to train your energy-based models.
