# Boltzmann Distribution

## Learning Objectives

After completing this section, you will be able to:

1. Derive the Boltzmann distribution from maximum entropy principles
2. Analyze temperature effects on distribution shape and thermodynamic quantities
3. Compute the score function and understand its partition-function-free property
4. Connect the Boltzmann distribution to the exponential family and Gibbs measures

## Mathematical Foundation

### The Canonical Distribution

Given an energy function $E(x)$, the Boltzmann (or Gibbs) distribution at temperature $T$ is:

$$p(x) = \frac{1}{Z(T)} \exp\left(-\frac{E(x)}{T}\right)$$

where the partition function $Z(T)$ ensures normalization:

$$Z(T) = \int_{\mathcal{X}} \exp\left(-\frac{E(x)}{T}\right) dx$$

For discrete states: $Z(T) = \sum_{x \in \mathcal{X}} \exp(-E(x)/T)$.

### Derivation from Maximum Entropy

The Boltzmann distribution is not an arbitrary choice—it is the unique distribution that maximizes entropy subject to a constraint on average energy. This derivation, due to Jaynes (1957), provides the deepest justification for the EBM framework.

**Objective**: Find $p(x)$ that maximizes:

$$H[p] = -\int p(x) \log p(x)\,dx$$

**Subject to**:

1. Normalization: $\int p(x)\,dx = 1$
2. Energy constraint: $\int p(x) E(x)\,dx = \langle E \rangle$

Using Lagrange multipliers $\alpha$ and $\beta$:

$$\mathcal{L} = -\int p \log p \, dx - \alpha\left(\int p \, dx - 1\right) - \beta\left(\int p E \, dx - \langle E \rangle\right)$$

Taking the functional derivative and setting to zero:

$$\frac{\delta \mathcal{L}}{\delta p} = -\log p - 1 - \alpha - \beta E = 0$$

Solving: $p(x) = \exp(-1 - \alpha - \beta E(x)) = \frac{1}{Z} \exp(-\beta E(x))$

where $\beta = 1/T$ is the inverse temperature. The Lagrange multiplier $\beta$ is determined by the energy constraint, and $\alpha$ is fixed by normalization. This derivation establishes that the Boltzmann distribution is the least biased distribution consistent with a given average energy—a powerful justification from information theory.

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

## Temperature Analysis

### Low Temperature Limit ($T \to 0$)

As $T \to 0$, the distribution concentrates at energy minima:

$$\lim_{T \to 0} p(x) = \frac{1}{|S^*|} \sum_{x^* \in S^*} \delta(x - x^*)$$

where $S^* = \{x : E(x) = \min_y E(y)\}$ is the set of global minima. In this limit, EBM inference reduces to optimization: finding the most probable configuration becomes equivalent to minimizing the energy function.

### High Temperature Limit ($T \to \infty$)

As $T \to \infty$, the distribution becomes uniform:

$$\lim_{T \to \infty} p(x) = \frac{1}{|\mathcal{X}|}$$

The exponential factor $\exp(-E(x)/T) \to 1$ for all $x$, and the energy landscape becomes irrelevant. All configurations become equally probable regardless of their energy.

### Temperature Sweep

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

## Score Function

### Definition

The score function is the gradient of log-probability with respect to the input:

$$s(x) = \nabla_x \log p(x) = -\frac{1}{T} \nabla_x E(x)$$

**Key insight**: The score function depends only on the energy gradient, not on the partition function $Z$. This property is the foundation of score matching (Section 26.3), which enables training EBMs without ever computing the intractable normalization constant.

### Interpretation

The score function points in the direction of increasing log-probability, i.e., toward lower energy. It describes the local "flow" of the probability landscape: at any point $x$, following the score function moves toward higher-density regions.

For the Boltzmann distribution, the score function is simply the (scaled) negative energy gradient. This means that training a model to predict $s(x) = -\nabla_x E(x)/T$ is equivalent to learning the energy function up to a constant—which is all that matters, since the partition function $Z$ is just a constant offset in log-probability.

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

For the Boltzmann distribution, the natural parameter is $\eta = -1/T$, the sufficient statistic is $T(x) = E(x)$, and the log-partition function is $A(\theta) = \log Z$. This connection means that the rich theory of exponential families—including conjugate priors, maximum likelihood estimation, and information geometry—applies directly to Boltzmann distributions.

### Gibbs Measure

In measure-theoretic terms, the Boltzmann distribution defines a Gibbs measure:

$$\mu(dx) = \frac{1}{Z} \exp(-E(x)) \, \lambda(dx)$$

where $\lambda$ is the reference measure (Lebesgue measure for continuous variables, counting measure for discrete). The Gibbs measure framework extends naturally to infinite-dimensional settings and is central to the mathematical foundations of statistical mechanics and lattice field theories.

## Key Takeaways

!!! success "Core Concepts"
    1. The Boltzmann distribution $p(x) \propto \exp(-E(x)/T)$ arises uniquely from maximum entropy under an energy constraint
    2. Temperature controls the spread: low $T$ concentrates at energy minima, high $T$ approaches uniform
    3. The score function $\nabla_x \log p = -\nabla_x E/T$ avoids the partition function, enabling practical training methods
    4. Thermodynamic quantities (free energy, entropy, heat capacity) are all derivable from the partition function
    5. The Boltzmann distribution belongs to the exponential family, connecting to a broad statistical theory

!!! info "Historical Note"
    Ludwig Boltzmann's work connecting microscopic mechanics to macroscopic thermodynamics was initially controversial—so much so that it contributed to his depression and eventual suicide in 1906. His equation $S = k \log W$ (relating entropy to the number of microstates) is inscribed on his tombstone in Vienna. Jaynes' maximum entropy interpretation, developed in the 1950s, provided a purely information-theoretic justification for Boltzmann's distribution that does not require any physical assumptions.

## Exercises

1. **Partition function bounds**: For the double-well energy $E(x) = (x^2-1)^2$, derive upper and lower bounds on $Z$ without computing it exactly. Hint: use Jensen's inequality and Laplace's method.

2. **Temperature crossover**: Find the temperature at which the double-well distribution $E(x) = \frac{1}{2}(x^2-4)^2$ transitions from bimodal to unimodal. Hint: examine the second derivative of $\log p(x)$ at $x=0$.

3. **Heat capacity peak**: Show that for a two-level system with energies $0$ and $\epsilon$, the heat capacity $C = \text{Var}[E]/T^2$ has a maximum (the Schottky anomaly) at $T^* = \epsilon / (2\sinh^{-1}(1))$.

## References

- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*.
- MacKay, D. J. (2003). *Information Theory, Inference and Learning Algorithms*. Cambridge University Press.
- Boltzmann, L. (1877). On the Relationship between the Second Fundamental Theorem of the Mechanical Theory of Heat and Probability Calculations.
