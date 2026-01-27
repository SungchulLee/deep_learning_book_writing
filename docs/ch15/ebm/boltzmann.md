# Boltzmann Distribution

## Learning Objectives

After completing this section, you will be able to:

1. Derive the Boltzmann distribution from first principles
2. Understand the role of the partition function
3. Analyze temperature effects mathematically
4. Connect Boltzmann statistics to maximum entropy principles

## Introduction

The Boltzmann distribution is the cornerstone of energy-based modeling, providing the mathematical bridge between energy functions and probability distributions. Named after Ludwig Boltzmann (1844-1906), this distribution emerges naturally from maximum entropy principles and forms the theoretical foundation for understanding equilibrium in both physical and computational systems.

## Mathematical Foundation

### The Canonical Distribution

Given an energy function $E(x)$, the Boltzmann (or Gibbs) distribution at temperature $T$ is:

$$p(x) = \frac{1}{Z(T)} \exp\left(-\frac{E(x)}{T}\right)$$

where the partition function $Z(T)$ ensures normalization:

$$Z(T) = \int_{\mathcal{X}} \exp\left(-\frac{E(x)}{T}\right) dx$$

For discrete states: $Z(T) = \sum_{x \in \mathcal{X}} \exp\left(-E(x)/T\right)$

### Derivation from Maximum Entropy

The Boltzmann distribution can be derived by maximizing entropy subject to a constraint on average energy.

**Objective**: Find $p(x)$ that maximizes:

$$H[p] = -\int p(x) \log p(x) dx$$

**Subject to**:
1. Normalization: $\int p(x) dx = 1$
2. Energy constraint: $\int p(x) E(x) dx = \langle E \rangle$

Using Lagrange multipliers $\alpha$ and $\beta$:

$$\mathcal{L} = -\int p \log p \, dx - \alpha\left(\int p \, dx - 1\right) - \beta\left(\int p E \, dx - \langle E \rangle\right)$$

Taking the functional derivative and setting to zero:

$$\frac{\delta \mathcal{L}}{\delta p} = -\log p - 1 - \alpha - \beta E = 0$$

Solving: $p(x) = \exp(-1 - \alpha - \beta E(x)) = \frac{1}{Z} \exp(-\beta E(x))$

where $\beta = 1/T$ is the inverse temperature.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def demonstrate_maxent_boltzmann():
    """
    Demonstrate that Boltzmann distribution maximizes entropy.
    
    Compare entropy of Boltzmann distribution with other distributions
    that have the same average energy.
    """
    # Define energy function (simple quadratic)
    def energy(x):
        return x**2
    
    # Define grid
    x = np.linspace(-5, 5, 1000)
    dx = x[1] - x[0]
    E = energy(x)
    
    # Target average energy
    target_energy = 1.0
    
    # Find temperature that gives target average energy
    def avg_energy_error(T):
        if T <= 0:
            return np.inf
        unnorm = np.exp(-E / T)
        Z = unnorm.sum() * dx
        p = unnorm / Z
        return (np.sum(p * E) * dx - target_energy)**2
    
    result = minimize_scalar(avg_energy_error, bounds=(0.1, 10), method='bounded')
    T_opt = result.x
    
    # Boltzmann distribution at optimal temperature
    boltz_unnorm = np.exp(-E / T_opt)
    Z = boltz_unnorm.sum() * dx
    p_boltz = boltz_unnorm / Z
    
    # Compute entropy of Boltzmann
    H_boltz = -np.sum(p_boltz * np.log(p_boltz + 1e-10)) * dx
    
    # Compare with truncated Gaussian (same mean energy)
    sigma = np.sqrt(2 * target_energy)
    p_gauss = np.exp(-x**2 / (2 * sigma**2))
    p_gauss = p_gauss / (p_gauss.sum() * dx)
    H_gauss = -np.sum(p_gauss * np.log(p_gauss + 1e-10)) * dx
    
    print(f"Boltzmann entropy: {H_boltz:.4f}")
    print(f"Gaussian entropy: {H_gauss:.4f}")
    print(f"Boltzmann has {'higher' if H_boltz > H_gauss else 'lower'} entropy")
    
    return p_boltz, p_gauss, x

p_boltz, p_gauss, x = demonstrate_maxent_boltzmann()
```

## The Partition Function

### Definition and Properties

The partition function $Z$ is arguably the most important quantity in statistical mechanics:

$$Z(T) = \int \exp(-E(x)/T) \, dx$$

**Key properties**:

1. **Normalization**: Ensures $\int p(x) dx = 1$
2. **Free Energy**: $F = -T \log Z$
3. **Generating Function**: Derivatives yield thermodynamic quantities

### Thermodynamic Quantities from Z

The partition function encodes all thermodynamic information:

| Quantity | Formula |
|----------|---------|
| Free Energy | $F = -T \log Z$ |
| Average Energy | $\langle E \rangle = -\frac{\partial \log Z}{\partial \beta}$ where $\beta = 1/T$ |
| Entropy | $S = \frac{\langle E \rangle - F}{T}$ |
| Heat Capacity | $C = \frac{\partial \langle E \rangle}{\partial T}$ |

```python
def compute_thermodynamic_quantities(energy_fn, x_range, T):
    """
    Compute thermodynamic quantities from partition function.
    
    Parameters
    ----------
    energy_fn : callable
        Energy function E(x)
    x_range : torch.Tensor
        Domain points for numerical integration
    T : float
        Temperature
    
    Returns
    -------
    dict
        Dictionary of thermodynamic quantities
    """
    dx = x_range[1] - x_range[0]
    E = energy_fn(x_range)
    
    # Partition function
    log_unnorm = -E / T
    log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
    Z = torch.exp(log_Z)
    
    # Probability distribution
    p = torch.exp(log_unnorm - log_Z)
    
    # Average energy
    avg_E = torch.sum(p * E) * dx
    
    # Free energy
    F = -T * log_Z
    
    # Entropy
    S = (avg_E - F) / T
    
    # Variance of energy (for heat capacity)
    var_E = torch.sum(p * (E - avg_E)**2) * dx
    C = var_E / (T**2)  # Heat capacity
    
    return {
        'Z': Z.item(),
        'F': F.item(),
        'avg_E': avg_E.item(),
        'S': S.item(),
        'C': C.item()
    }

# Example: analyze double-well potential
x = torch.linspace(-4, 4, 1000)
double_well = lambda x: 0.5 * (x**2 - 4)**2

for T in [0.5, 1.0, 2.0, 5.0]:
    quantities = compute_thermodynamic_quantities(double_well, x, T)
    print(f"T={T}: ⟨E⟩={quantities['avg_E']:.3f}, S={quantities['S']:.3f}, C={quantities['C']:.3f}")
```

### The Intractability Problem

For high-dimensional spaces, the partition function becomes intractable:

$$Z = \int_{\mathbb{R}^d} \exp(-E(x)) \, dx$$

**Why intractable?**
- Exponential growth: $O(n^d)$ for grid methods
- No closed-form for most energy functions
- Monte Carlo estimation has high variance

This intractability motivates training methods that avoid computing $Z$:
- Contrastive Divergence
- Score Matching
- Noise Contrastive Estimation

## Temperature Analysis

### Low Temperature Limit ($T \to 0$)

As $T \to 0$, the distribution concentrates at energy minima:

$$\lim_{T \to 0} p(x) = \frac{1}{|S^*|} \sum_{x^* \in S^*} \delta(x - x^*)$$

where $S^* = \{x : E(x) = \min_y E(y)\}$ is the set of global minima.

### High Temperature Limit ($T \to \infty$)

As $T \to \infty$, the distribution becomes uniform:

$$\lim_{T \to \infty} p(x) = \frac{1}{|\mathcal{X}|}$$

The exponential factor $\exp(-E(x)/T) \to 1$ for all $x$.

### Temperature Sweep Analysis

```python
def analyze_temperature_sweep():
    """
    Analyze how Boltzmann distribution changes with temperature.
    """
    x = torch.linspace(-4, 4, 1000)
    E = 0.5 * (x**2 - 4)**2  # Double-well
    dx = x[1] - x[0]
    
    temperatures = torch.tensor([0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution shape vs temperature
    ax1 = axes[0, 0]
    for T in [0.5, 1.0, 2.0, 5.0]:
        log_unnorm = -E / T
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        ax1.plot(x.numpy(), p.numpy(), linewidth=2, label=f'T = {T}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x)')
    ax1.set_title('Distribution Shape vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy vs temperature
    ax2 = axes[0, 1]
    entropies = []
    for T in temperatures:
        log_unnorm = -E / T.item()
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        H = -torch.sum(p * torch.log(p + 1e-10)) * dx
        entropies.append(H.item())
    
    ax2.plot(temperatures.numpy(), entropies, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('Entropy H')
    ax2.set_title('Entropy Increases with Temperature')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average energy vs temperature
    ax3 = axes[1, 0]
    avg_energies = []
    for T in temperatures:
        log_unnorm = -E / T.item()
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        avg_E = torch.sum(p * E) * dx
        avg_energies.append(avg_E.item())
    
    ax3.plot(temperatures.numpy(), avg_energies, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Temperature T')
    ax3.set_ylabel('⟨E⟩')
    ax3.set_title('Average Energy Increases with Temperature')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mode probability vs temperature
    ax4 = axes[1, 1]
    # Find probability mass in each mode
    left_mode = x < 0
    right_mode = x >= 0
    
    for T in [0.5, 1.0, 2.0]:
        log_unnorm = -E / T
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        
        left_prob = torch.sum(p[left_mode]) * dx
        right_prob = torch.sum(p[right_mode]) * dx
        
        ax4.bar([T - 0.1, T + 0.1], [left_prob.item(), right_prob.item()], 
                width=0.15, label=f'T={T}' if T == 0.5 else None)
    
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Mode Probability')
    ax4.set_title('Mode Occupation vs Temperature')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

analyze_temperature_sweep()
```

## Log-Probability and Score Function

### Log-Probability

The log-probability under the Boltzmann distribution is:

$$\log p(x) = -\frac{E(x)}{T} - \log Z(T)$$

The partition function term $\log Z$ is a constant w.r.t. $x$ but depends on parameters.

### Score Function

The score function is the gradient of log-probability:

$$s(x) = \nabla_x \log p(x) = -\frac{1}{T} \nabla_x E(x)$$

**Key insight**: The score function depends only on the energy gradient, not on $Z$!

This enables score matching training (covered in depth later).

```python
def compute_score(energy_net, x, create_graph=True):
    """
    Compute score function s(x) = -∇E(x).
    
    Parameters
    ----------
    energy_net : nn.Module
        Neural network energy function
    x : torch.Tensor
        Input points, shape (batch, dim)
    create_graph : bool
        Whether to create graph for higher-order gradients
    
    Returns
    -------
    torch.Tensor
        Score values, same shape as x
    """
    x = x.requires_grad_(True)
    energy = energy_net(x).sum()
    
    score = torch.autograd.grad(
        outputs=energy,
        inputs=x,
        create_graph=create_graph
    )[0]
    
    return -score  # Score is negative gradient of energy
```

## Connections to Other Distributions

### Exponential Family

The Boltzmann distribution is a member of the exponential family:

$$p(x; \theta) = h(x) \exp(\eta(\theta)^T T(x) - A(\theta))$$

For the Boltzmann distribution:
- Natural parameter: $\eta = -1/T$
- Sufficient statistic: $T(x) = E(x)$
- Log-partition function: $A(\theta) = \log Z$

### Gibbs Measure

In measure-theoretic terms, the Boltzmann distribution defines a Gibbs measure:

$$\mu(dx) = \frac{1}{Z} \exp(-E(x)) \, \lambda(dx)$$

where $\lambda$ is the reference measure (Lebesgue or counting).

## Key Takeaways

!!! success "Core Concepts"
    1. The Boltzmann distribution $p(x) \propto \exp(-E(x)/T)$ arises from maximum entropy
    2. The partition function $Z$ ensures normalization but is typically intractable
    3. Temperature controls the spread: low $T$ → peaked, high $T$ → uniform
    4. The score function $\nabla_x \log p = -\nabla_x E/T$ avoids the partition function
    5. Thermodynamic quantities can be derived from $Z$

!!! info "Historical Note"
    Ludwig Boltzmann's work connecting microscopic mechanics to macroscopic thermodynamics was initially controversial. His equation $S = k \log W$ (relating entropy to the number of microstates) is inscribed on his tombstone in Vienna.

## Exercises

1. **Partition Function Bounds**: For the double-well energy $E(x) = (x^2-1)^2$, derive upper and lower bounds on $Z$ without computing it exactly.

2. **Temperature Crossover**: Find the temperature at which the double-well distribution transitions from bimodal to unimodal. Hint: examine the second derivative at $x=0$.

3. **Heat Capacity Peak**: Show that for a two-level system with energies $0$ and $\epsilon$, the heat capacity has a maximum at $T^* = \epsilon / (2\sinh^{-1}(1))$.

## References

- Jaynes, E. T. (1957). Information theory and statistical mechanics. Physical Review.
- MacKay, D. J. (2003). Information Theory, Inference and Learning Algorithms. Cambridge University Press.
- Boltzmann, L. (1877). On the Relationship between the Second Fundamental Theorem of the Mechanical Theory of Heat and Probability Calculations.
