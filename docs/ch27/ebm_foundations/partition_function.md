# Partition Function

## Learning Objectives

After completing this section, you will be able to:

1. Understand the partition function as the central quantity in energy-based modeling
2. Derive thermodynamic quantities as derivatives of $\log Z$
3. Explain why the partition function is intractable and the consequences for training
4. Apply numerical methods for partition function estimation in low dimensions

## Definition and Role

The partition function is the normalization constant of the Boltzmann distribution:

$$Z(T) = \int_{\mathcal{X}} \exp\left(-\frac{E(x)}{T}\right) dx$$

For discrete state spaces: $Z(T) = \sum_{x \in \mathcal{X}} \exp(-E(x)/T)$.

Despite being "just" a normalization constant, $Z$ is arguably the most important quantity in statistical mechanics and energy-based modeling. It encodes the complete distributional information about the system: every expectation, moment, and thermodynamic quantity can be extracted from $Z$ and its derivatives.

## Thermodynamic Quantities from $Z$

The partition function serves as a generating function for thermodynamic quantities:

| Quantity | Formula |
|----------|---------|
| Free Energy | $F = -T \log Z$ |
| Average Energy | $\langle E \rangle = -\frac{\partial \log Z}{\partial \beta}$ where $\beta = 1/T$ |
| Entropy | $S = \frac{\langle E \rangle - F}{T} = \beta \langle E \rangle + \log Z$ |
| Heat Capacity | $C = \frac{\partial \langle E \rangle}{\partial T} = \frac{\text{Var}[E]}{T^2}$ |
| Energy Variance | $\text{Var}[E] = \frac{\partial^2 \log Z}{\partial \beta^2}$ |

The heat capacity formula $C = \text{Var}[E]/T^2$ is particularly instructive: it shows that fluctuations in energy (a microscopic quantity) determine the rate of change of average energy with temperature (a macroscopic quantity). This fluctuation-response relation is a hallmark of statistical mechanics.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Partition function (log-sum-exp for stability)
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

# Analyze double-well potential across temperatures
x = torch.linspace(-4, 4, 1000)
double_well = lambda x: 0.5 * (x**2 - 4)**2

print("Thermodynamic quantities for double-well potential:")
print(f"{'T':>5} {'⟨E⟩':>10} {'F':>10} {'S':>10} {'C':>10}")
print("-" * 50)
for T in [0.5, 1.0, 2.0, 5.0]:
    q = compute_thermodynamic_quantities(double_well, x, T)
    print(f"{T:5.1f} {q['avg_E']:10.3f} {q['F']:10.3f} {q['S']:10.3f} {q['C']:10.3f}")
```

### Phase Transition Signatures

In physical systems, phase transitions manifest as singularities in thermodynamic quantities derived from $Z$. While EBMs in machine learning do not undergo true phase transitions (which require infinite system size), analogous phenomena appear:

**Sharp energy barriers**: When the energy landscape has well-separated modes with high barriers, the heat capacity $C(T)$ shows a peak at a characteristic temperature. Below this temperature, the system is "frozen" in a single mode; above it, transitions between modes become frequent.

**Mode competition**: At the critical temperature, the distribution transitions from multimodal to effectively unimodal. This temperature marks where the entropy cost of localizing in one mode is balanced by the energy benefit.

```python
def visualize_phase_transition_analogy():
    """
    Show how thermodynamic quantities signal structural changes
    in the Boltzmann distribution.
    """
    x = torch.linspace(-4, 4, 1000)
    E = 0.5 * (x**2 - 4)**2  # Double-well
    
    temperatures = torch.linspace(0.1, 8.0, 100)
    quantities = {'T': [], 'avg_E': [], 'S': [], 'C': []}
    
    for T in temperatures:
        q = compute_thermodynamic_quantities(lambda x: 0.5*(x**2-4)**2, x, T.item())
        quantities['T'].append(T.item())
        quantities['avg_E'].append(q['avg_E'])
        quantities['S'].append(q['S'])
        quantities['C'].append(q['C'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(quantities['T'], quantities['avg_E'], 'r-', linewidth=2)
    axes[0].set_xlabel('Temperature T')
    axes[0].set_ylabel('⟨E⟩')
    axes[0].set_title('Average Energy')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(quantities['T'], quantities['S'], 'b-', linewidth=2)
    axes[1].set_xlabel('Temperature T')
    axes[1].set_ylabel('Entropy S')
    axes[1].set_title('Entropy')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(quantities['T'], quantities['C'], 'g-', linewidth=2)
    axes[2].set_xlabel('Temperature T')
    axes[2].set_ylabel('Heat Capacity C')
    axes[2].set_title('Heat Capacity (Peak ≈ Transition)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_phase_transition_analogy()
```

## The Intractability Problem

For high-dimensional spaces, the partition function becomes intractable:

$$Z = \int_{\mathbb{R}^d} \exp(-E(x)) \, dx$$

### Why Intractable?

**Exponential growth of integration domain**: Grid-based numerical integration requires $O(n^d)$ evaluations, where $n$ is the number of grid points per dimension. For $d = 100$ (a modest dimensionality for financial data) with even $n = 10$, this requires $10^{100}$ evaluations—far exceeding the number of atoms in the observable universe.

**No closed-form for general energy functions**: Closed-form partition functions exist only for special cases (Gaussian energy → Gaussian integral, linear energy → Laplace transform). Neural network energy functions have no closed-form $Z$.

**High variance of Monte Carlo estimates**: The naive estimator $\hat{Z} = \frac{1}{N}\sum_i \exp(-E(x_i))$ for $x_i$ drawn uniformly has extremely high variance because $\exp(-E(x))$ varies over many orders of magnitude across the domain.

### Consequences for Training

The intractability of $Z$ has profound consequences:

**No direct likelihood evaluation**: We cannot compute $\log p(x) = -E(x) - \log Z$ exactly, which means standard maximum likelihood training is not directly applicable.

**Gradient requires model expectations**: The MLE gradient $\nabla_\theta \log p(x) = -\nabla_\theta E(x) + \mathbb{E}_{p_\theta}[\nabla_\theta E]$ requires samples from the model distribution (the "negative phase"), which itself requires MCMC sampling from the intractable distribution.

**Model comparison is difficult**: Without likelihoods, comparing EBMs against each other or against other model families requires alternative metrics (FID for images, energy-based evaluation, etc.).

### Training Methods That Avoid $Z$

The intractability of $Z$ has motivated several elegant training approaches, each covered in detail in Section 26.3:

**Contrastive Divergence**: Approximates the model expectation using short-run MCMC starting from data. The key insight is that starting near the data distribution reduces the mixing time needed.

**Score matching**: Matches the gradient of the model's log-density to that of the data, completely avoiding $Z$ since $\nabla_x \log p(x) = -\nabla_x E(x)/T$ is independent of $Z$.

**Noise Contrastive Estimation**: Transforms density estimation into a classification problem—distinguishing data from noise—which requires only the energy function, not the partition function.

## Partition Function Estimation

While exact computation of $Z$ is intractable in high dimensions, several estimation methods are useful for evaluation, model comparison, and low-dimensional diagnostics.

### Annealed Importance Sampling (AIS)

AIS estimates $Z$ by constructing a sequence of intermediate distributions that bridge from a tractable reference to the target:

$$\hat{Z} = Z_0 \cdot \prod_{k=1}^{K} \frac{p_{k}(x^{(k)})}{p_{k-1}(x^{(k)})}$$

where $p_0$ is a tractable distribution (e.g., uniform or Gaussian) and $p_K = p_\theta$ is the target. This approach is the gold standard for estimating partition functions of trained EBMs.

```python
def annealed_importance_sampling(energy_fn, dim, n_chains=100,
                                  n_intermediate=100, n_gibbs=10):
    """
    Estimate log Z using Annealed Importance Sampling.
    
    Parameters
    ----------
    energy_fn : callable
        Energy function E(x)
    dim : int
        Dimensionality
    n_chains : int
        Number of parallel chains
    n_intermediate : int
        Number of intermediate distributions
    n_gibbs : int
        Gibbs steps per intermediate distribution
    
    Returns
    -------
    float
        Estimated log Z
    """
    # Inverse temperatures from 0 (uniform) to 1 (target)
    betas = torch.linspace(0, 1, n_intermediate + 1)
    
    # Initialize from reference distribution (standard normal)
    x = torch.randn(n_chains, dim)
    
    # Log importance weights
    log_weights = torch.zeros(n_chains)
    
    for k in range(1, n_intermediate + 1):
        # Energy at current and previous temperatures
        E = energy_fn(x)
        
        # Accumulate log weight
        log_weights += -(betas[k] - betas[k-1]) * E
        
        # Transition: Langevin dynamics at intermediate temperature
        for _ in range(n_gibbs):
            x.requires_grad_(True)
            E_current = energy_fn(x) * betas[k]
            grad = torch.autograd.grad(E_current.sum(), x)[0]
            x = x.detach() - 0.01 * grad + 0.005 * torch.randn_like(x)
    
    # Log Z estimate via log-mean-exp
    log_Z_ref = 0.5 * dim * np.log(2 * np.pi)  # Log Z of standard normal
    log_Z = torch.logsumexp(log_weights, dim=0) - np.log(n_chains) + log_Z_ref
    
    return log_Z.item()
```

### Bridge Sampling

Bridge sampling uses samples from both the reference and target distributions to estimate the ratio $Z_\text{target}/Z_\text{reference}$. It can be more efficient than AIS when both distributions are easy to sample from, but is less commonly used for EBMs where sampling from the target is itself the challenge.

## Free Energy as the Fundamental Quantity

In practice, the free energy $F = -T \log Z$ is often more useful than $Z$ itself. The free energy satisfies a variational principle:

$$F = \min_q \left[\mathbb{E}_q[E(x)] + T \cdot H[q]\right]$$

where the minimum is over all distributions $q$ and $H[q]$ is the entropy of $q$. The optimal $q^*$ is exactly the Boltzmann distribution.

This variational characterization connects directly to variational inference in machine learning: the evidence lower bound (ELBO) is a free energy bound, and variational autoencoders can be understood as performing free energy minimization with a restricted family of variational distributions.

## Key Takeaways

!!! success "Core Concepts"
    1. The partition function $Z = \int \exp(-E(x)/T)\,dx$ normalizes the Boltzmann distribution and generates all thermodynamic quantities through its derivatives
    2. $Z$ is intractable to compute for high-dimensional, general-purpose energy functions—this is the central computational challenge of EBMs
    3. Training methods (CD, score matching, NCE) are fundamentally strategies for avoiding the need to compute $Z$
    4. The free energy $F = -T\log Z$ satisfies a variational principle that connects to variational inference
    5. Annealed Importance Sampling provides the best practical estimates of $Z$ for evaluation purposes

!!! warning "Common Misconceptions"
    - The partition function is not "just a constant"—it depends on model parameters $\theta$ and must be accounted for during training
    - Intractability of $Z$ does not mean EBMs cannot be trained effectively—it means we need clever approximations
    - Low-dimensional estimates of $Z$ (via grid integration) do not generalize to high dimensions due to the curse of dimensionality

## Exercises

1. **Exact vs. estimated**: For a 2D energy function of your choice, compute $Z$ exactly via grid integration and compare with AIS estimates. How many intermediate distributions does AIS need for accurate estimation?

2. **Variational free energy**: Implement the variational free energy $F_q = \mathbb{E}_q[E] + T \cdot H[q]$ for a family of Gaussian variational distributions $q$. Optimize $q$ and compare the variational bound to the true free energy.

3. **Thermodynamic integration**: Implement thermodynamic integration $\log Z_1 - \log Z_0 = \int_0^1 \langle E_1 - E_0 \rangle_\beta \, d\beta$ to estimate the log partition function ratio between two different energy functions.

## References

- Neal, R. M. (2001). Annealed Importance Sampling. *Statistics and Computing*.
- Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing constants via a simple identity. *Statistica Sinica*.
- Salakhutdinov, R., & Murray, I. (2008). On the quantitative analysis of deep belief networks. *ICML*.
