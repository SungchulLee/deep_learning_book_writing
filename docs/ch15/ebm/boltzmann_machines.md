# Boltzmann Machines

## Learning Objectives

After completing this section, you will be able to:

1. Understand stochastic binary units and their dynamics
2. Implement Gibbs sampling for Boltzmann machines
3. Distinguish between deterministic and stochastic EBMs
4. Analyze equilibrium distributions and convergence
5. Connect Boltzmann machines to modern probabilistic models

## Introduction

Boltzmann Machines (BMs), introduced by Hinton and Sejnowski in 1985, extend Hopfield networks by introducing stochastic dynamics. While Hopfield networks deterministically converge to energy minima, Boltzmann machines use thermal noise to sample from the entire Boltzmann distribution. This enables them to act as generative models that can learn probability distributions over data.

## From Hopfield to Boltzmann

### The Stochastic Extension

The key difference from Hopfield networks is the update rule:

| Model | Update Rule |
|-------|-------------|
| Hopfield | $s_i \leftarrow \text{sign}(h_i)$ (deterministic) |
| Boltzmann | $P(s_i = 1) = \sigma(h_i / T)$ (stochastic) |

where $h_i = \sum_j w_{ij} s_j + \theta_i$ is the local field and $\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function.

### Physical Interpretation

- **Temperature $T$**: Controls randomness level
  - $T \to 0$: Approaches deterministic (Hopfield-like)
  - $T \to \infty$: Random coin flips
  - $T = 1$: Standard Boltzmann machine

- **Thermal equilibrium**: After many updates, the distribution converges to:
$$P(\mathbf{s}) = \frac{1}{Z} \exp(-E(\mathbf{s})/T)$$

## Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from tqdm import tqdm

class BoltzmannMachine(nn.Module):
    """
    General Boltzmann Machine with stochastic binary units.
    
    Unlike Hopfield networks, BMs use probabilistic updates
    and can sample from the Boltzmann distribution at equilibrium.
    
    Parameters
    ----------
    n_visible : int
        Number of visible units
    n_hidden : int
        Number of hidden units (0 for no hidden layer)
    temperature : float
        Temperature parameter controlling randomness
    """
    
    def __init__(self, 
                 n_visible: int, 
                 n_hidden: int = 0, 
                 temperature: float = 1.0):
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_total = n_visible + n_hidden
        self.temperature = temperature
        
        # Initialize weights (symmetric, no self-connections)
        W = torch.randn(self.n_total, self.n_total) * 0.01
        W = (W + W.T) / 2  # Symmetrize
        W.fill_diagonal_(0)  # No self-connections
        self.register_buffer('W', W)
        
        # Biases
        self.register_buffer('theta', torch.zeros(self.n_total))
    
    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute energy E(s) = -½ sᵀWs - θᵀs
        
        Parameters
        ----------
        state : torch.Tensor
            Binary state in {0, 1}^n or {-1, +1}^n
        """
        # Convert {0,1} to {-1,+1} if needed
        if state.min() >= 0:
            s = 2 * state - 1
        else:
            s = state
        
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        quadratic = -0.5 * torch.einsum('bi,ij,bj->b', s, self.W, s)
        linear = -torch.einsum('i,bi->b', self.theta, s)
        
        return quadratic + linear
    
    def local_field(self, state: torch.Tensor) -> torch.Tensor:
        """Compute h_i = Σⱼ w_ij s_j + θ_i"""
        s = 2 * state - 1 if state.min() >= 0 else state
        return torch.mv(self.W, s) + self.theta
    
    def sample_probability(self, local_field: torch.Tensor) -> torch.Tensor:
        """
        Compute P(s_i = 1 | s_{-i}) = σ(h_i / T)
        """
        return torch.sigmoid(local_field / self.temperature)
    
    def sample_unit(self, 
                    state: torch.Tensor, 
                    unit_idx: int) -> torch.Tensor:
        """
        Sample a single unit given current state.
        
        P(s_i = 1) = σ((Σⱼ w_ij s_j + θ_i) / T)
        """
        new_state = state.clone()
        h_i = self.local_field(state)[unit_idx]
        prob_on = self.sample_probability(h_i)
        new_state[unit_idx] = (torch.rand(1) < prob_on).float()
        return new_state
    
    def gibbs_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform one complete Gibbs sampling sweep.
        
        Updates all units in random order.
        """
        new_state = state.clone()
        update_order = torch.randperm(self.n_total)
        
        for unit_idx in update_order:
            new_state = self.sample_unit(new_state, unit_idx.item())
        
        return new_state
    
    def sample(self, 
               n_steps: int = 1000, 
               initial_state: Optional[torch.Tensor] = None,
               return_trajectory: bool = False) -> torch.Tensor:
        """
        Generate sample(s) via Gibbs sampling.
        
        Parameters
        ----------
        n_steps : int
            Number of Gibbs sweeps
        initial_state : torch.Tensor, optional
            Starting state (random if None)
        return_trajectory : bool
            If True, return all intermediate states
        """
        # Initialize
        if initial_state is None:
            state = (torch.rand(self.n_total) > 0.5).float()
        else:
            state = initial_state.clone()
        
        trajectory = [state.clone()] if return_trajectory else None
        
        # Run Gibbs sampling
        for _ in range(n_steps):
            state = self.gibbs_step(state)
            if return_trajectory:
                trajectory.append(state.clone())
        
        if return_trajectory:
            return torch.stack(trajectory)
        return state
    
    def estimate_distribution(self, 
                              n_samples: int = 10000,
                              burn_in: int = 1000,
                              thin: int = 10) -> dict:
        """
        Estimate the equilibrium distribution via MCMC.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to collect
        burn_in : int  
            Samples to discard for equilibration
        thin : int
            Keep every thin-th sample to reduce autocorrelation
        """
        state = (torch.rand(self.n_total) > 0.5).float()
        
        # Burn-in
        for _ in range(burn_in):
            state = self.gibbs_step(state)
        
        # Collect samples
        samples = []
        energies = []
        
        for i in tqdm(range(n_samples * thin), desc="Sampling"):
            state = self.gibbs_step(state)
            
            if i % thin == 0:
                samples.append(state.clone())
                energies.append(self.energy(state).item())
        
        samples = torch.stack(samples)
        
        # Compute empirical distribution
        state_counts = {}
        for sample in samples:
            state_tuple = tuple(sample.numpy().astype(int))
            state_counts[state_tuple] = state_counts.get(state_tuple, 0) + 1
        
        empirical_probs = {
            state: count / n_samples 
            for state, count in state_counts.items()
        }
        
        return {
            'samples': samples,
            'energies': np.array(energies),
            'empirical_probs': empirical_probs,
            'state_counts': state_counts
        }


def compare_deterministic_vs_stochastic():
    """
    Compare Hopfield (deterministic) vs Boltzmann (stochastic) dynamics.
    """
    print("="*70)
    print("DETERMINISTIC VS STOCHASTIC DYNAMICS")
    print("="*70)
    
    # Simple pattern
    pattern = torch.tensor([1., 1., 1., 0., 0., 0.])
    n_units = len(pattern)
    
    # Create weight matrix (Hebbian)
    pattern_pm = 2 * pattern - 1
    W = torch.outer(pattern_pm, pattern_pm)
    W.fill_diagonal_(0)
    
    # Noisy starting point
    noisy = pattern.clone()
    noisy[2] = 0
    noisy[4] = 1
    
    print(f"Target pattern: {pattern.numpy().astype(int)}")
    print(f"Noisy input:    {noisy.numpy().astype(int)}")
    
    # Deterministic dynamics
    print("\n" + "-"*50)
    print("DETERMINISTIC (Hopfield-style)")
    state_det = noisy.clone()
    det_trajectory = [state_det.clone()]
    
    for _ in range(10):
        # Convert to ±1
        s = 2 * state_det - 1
        # Local fields
        h = torch.mv(W, s)
        # Deterministic update (random order)
        for i in torch.randperm(n_units):
            s[i] = torch.sign(h[i]) if h[i] != 0 else s[i]
            h = torch.mv(W, s)
        
        state_det = (s + 1) / 2
        det_trajectory.append(state_det.clone())
        
        if (det_trajectory[-1] == det_trajectory[-2]).all():
            break
    
    print(f"Converged to: {state_det.numpy().astype(int)}")
    print(f"Iterations: {len(det_trajectory) - 1}")
    
    # Stochastic dynamics
    print("\n" + "-"*50)
    print("STOCHASTIC (Boltzmann)")
    
    bm = BoltzmannMachine(n_visible=n_units, temperature=1.0)
    bm.W = W
    
    # Multiple runs to show distribution
    n_runs = 100
    final_states = []
    
    for _ in range(n_runs):
        state_stoch = noisy.clone()
        for _ in range(50):  # More iterations
            state_stoch = bm.gibbs_step(state_stoch)
        final_states.append(tuple(state_stoch.numpy().astype(int)))
    
    # Count outcomes
    from collections import Counter
    outcomes = Counter(final_states)
    
    print(f"Outcomes over {n_runs} runs:")
    for state, count in outcomes.most_common():
        print(f"  {state}: {count/n_runs*100:.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Deterministic trajectory
    traj_array = torch.stack(det_trajectory).numpy()
    axes[0].imshow(traj_array.T, cmap='binary', aspect='auto')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Neuron')
    axes[0].set_title('Deterministic: Single Path to Fixed Point')
    
    # Stochastic histogram
    unique_states = list(outcomes.keys())
    counts = [outcomes[s] for s in unique_states]
    axes[1].bar(range(len(unique_states)), counts)
    axes[1].set_xticks(range(len(unique_states)))
    axes[1].set_xticklabels([str(s) for s in unique_states], rotation=45, ha='right')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Stochastic: Distribution of Final States')
    
    # Temperature comparison
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    entropies = []
    
    for T in temperatures:
        bm_temp = BoltzmannMachine(n_visible=n_units, temperature=T)
        bm_temp.W = W
        
        # Quick sampling
        states = []
        state = noisy.clone()
        for _ in range(500):
            state = bm_temp.gibbs_step(state)
            states.append(tuple(state.numpy().astype(int)))
        
        # Compute entropy
        counts = Counter(states)
        probs = np.array(list(counts.values())) / 500
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
    
    axes[2].plot(temperatures, entropies, 'bo-', markersize=8, linewidth=2)
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Entropy of Sampled States')
    axes[2].set_title('Higher Temperature → More Exploration')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_deterministic_vs_stochastic()
```

## Gibbs Sampling and Equilibrium

### The Gibbs Sampling Algorithm

Gibbs sampling is an MCMC method that generates samples from the joint distribution by iteratively sampling from conditional distributions:

$$P(s_i = 1 | \mathbf{s}_{-i}) = \sigma\left(\frac{\sum_j w_{ij} s_j + \theta_i}{T}\right)$$

**Algorithm**:
1. Initialize state randomly
2. For each step:
   - Select unit $i$ (random or sequential)
   - Sample $s_i$ from $P(s_i | \mathbf{s}_{-i})$
3. Repeat until equilibrium

### Convergence to Equilibrium

Under mild conditions, Gibbs sampling converges to the Boltzmann distribution:

$$\lim_{t \to \infty} P(\mathbf{s}^{(t)} = \mathbf{s}) = \frac{\exp(-E(\mathbf{s})/T)}{Z}$$

**Detailed balance** ensures convergence:
$$P(\mathbf{s}) P(\mathbf{s} \to \mathbf{s}') = P(\mathbf{s}') P(\mathbf{s}' \to \mathbf{s})$$

```python
def verify_equilibrium_distribution():
    """
    Verify that Gibbs sampling converges to Boltzmann distribution.
    """
    print("="*70)
    print("VERIFYING EQUILIBRIUM DISTRIBUTION")
    print("="*70)
    
    # Small network for enumeration
    n_units = 4
    bm = BoltzmannMachine(n_visible=n_units, temperature=1.0)
    
    # Set specific weights
    bm.W = torch.tensor([
        [0.0, 1.5, -1.0, 0.5],
        [1.5, 0.0, 0.5, -1.0],
        [-1.0, 0.5, 0.0, 1.5],
        [0.5, -1.0, 1.5, 0.0]
    ])
    bm.theta = torch.tensor([0.5, -0.5, 0.0, 0.5])
    
    # Compute theoretical distribution (exhaustive enumeration)
    theoretical_probs = {}
    energies = {}
    
    for i in range(2**n_units):
        binary = format(i, f'0{n_units}b')
        state = torch.tensor([float(b) for b in binary])
        
        energy = bm.energy(state).item()
        energies[tuple(state.numpy().astype(int))] = energy
    
    # Compute partition function
    Z = sum(np.exp(-E / bm.temperature) for E in energies.values())
    
    for state, E in energies.items():
        theoretical_probs[state] = np.exp(-E / bm.temperature) / Z
    
    # Estimate empirically via Gibbs sampling
    results = bm.estimate_distribution(n_samples=5000, burn_in=500, thin=5)
    
    # Compare
    print("\nComparison of theoretical vs empirical probabilities:")
    print("-"*70)
    print(f"{'State':<20} {'Theoretical':>15} {'Empirical':>15} {'Energy':>10}")
    print("-"*70)
    
    sorted_states = sorted(theoretical_probs.keys(), 
                           key=lambda s: theoretical_probs[s], reverse=True)
    
    for state in sorted_states:
        theo = theoretical_probs[state]
        emp = results['empirical_probs'].get(state, 0)
        E = energies[state]
        print(f"{str(state):<20} {theo:>15.4f} {emp:>15.4f} {E:>10.2f}")
    
    # Compute correlation
    theo_vals = []
    emp_vals = []
    for state in theoretical_probs:
        theo_vals.append(theoretical_probs[state])
        emp_vals.append(results['empirical_probs'].get(state, 0))
    
    correlation = np.corrcoef(theo_vals, emp_vals)[0, 1]
    print(f"\nCorrelation: {correlation:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(theo_vals, emp_vals, alpha=0.7, s=100)
    max_val = max(max(theo_vals), max(emp_vals))
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect match')
    axes[0].set_xlabel('Theoretical Probability')
    axes[0].set_ylabel('Empirical Probability')
    axes[0].set_title(f'Gibbs Sampling Accuracy (corr={correlation:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Energy histogram
    axes[1].hist(results['energies'], bins=30, density=True, alpha=0.7, 
                 edgecolor='black', label='Sampled')
    
    # Theoretical energy distribution
    E_range = np.linspace(min(results['energies']), max(results['energies']), 100)
    # This is approximate - would need proper marginalization
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Energy Distribution from Gibbs Sampling')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return correlation

correlation = verify_equilibrium_distribution()
```

## Temperature Effects

### Analysis of Temperature

```python
def analyze_temperature_effects():
    """
    Analyze how temperature affects sampling behavior.
    """
    n_units = 6
    base_bm = BoltzmannMachine(n_visible=n_units, temperature=1.0)
    
    # Set weights to create an interesting energy landscape
    W = torch.randn(n_units, n_units) * 0.5
    W = (W + W.T) / 2
    W.fill_diagonal_(0)
    base_bm.W = W
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, T in enumerate(temperatures):
        bm = BoltzmannMachine(n_visible=n_units, temperature=T)
        bm.W = W.clone()
        bm.theta = base_bm.theta.clone()
        
        # Sample
        n_samples = 2000
        state = (torch.rand(n_units) > 0.5).float()
        energies = []
        
        for _ in range(n_samples):
            state = bm.gibbs_step(state)
            energies.append(bm.energy(state).item())
        
        # Plot
        ax = axes[idx]
        ax.hist(energies, bins=30, density=True, alpha=0.7, 
                edgecolor='black', color='steelblue')
        ax.set_xlabel('Energy', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'T = {T}', fontsize=13, fontweight='bold')
        
        # Statistics
        mean_E = np.mean(energies)
        std_E = np.std(energies)
        ax.axvline(mean_E, color='red', linestyle='--', linewidth=2)
        ax.text(0.95, 0.95, f'μ = {mean_E:.2f}\nσ = {std_E:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off')
    
    plt.suptitle('Temperature Effects on Boltzmann Machine Sampling', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("• Low T → Concentrated at low energies (exploitation)")
    print("• High T → Spread across energy levels (exploration)")
    print("• T controls the trade-off between exploitation and exploration")

analyze_temperature_effects()
```

## Visible and Hidden Units

### Two Types of Units

In practice, Boltzmann machines have two types of units:

- **Visible units ($\mathbf{v}$)**: Represent observed data
- **Hidden units ($\mathbf{h}$)**: Capture latent structure

The energy function becomes:
$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W}_{\text{vh}} \mathbf{h} - \mathbf{v}^T \mathbf{W}_{\text{vv}} \mathbf{v} - \mathbf{h}^T \mathbf{W}_{\text{hh}} \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h}$$

### Marginal Distribution

The marginal over visible units:
$$P(\mathbf{v}) = \sum_{\mathbf{h}} P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} \sum_{\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$$

This defines the **free energy**:
$$F(\mathbf{v}) = -\log \sum_{\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$$

So: $P(\mathbf{v}) = \frac{1}{Z} \exp(-F(\mathbf{v}))$

## Training Boltzmann Machines

### Maximum Likelihood Objective

Given data $\{\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(N)}\}$, maximize:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_n \log P(\mathbf{v}^{(n)}; \theta)$$

### Gradient of Log-Likelihood

The gradient has a elegant form:
$$\frac{\partial \log P(\mathbf{v})}{\partial w_{ij}} = \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}}$$

- **Positive phase**: Statistics with data clamped
- **Negative phase**: Statistics from model samples

### The Challenge

Computing $\langle s_i s_j \rangle_{\text{model}}$ requires sampling from the model, which is slow for general BMs due to dependencies between all units.

This motivates **Restricted Boltzmann Machines** (next section).

## Key Takeaways

!!! success "Core Concepts"
    1. Boltzmann machines extend Hopfield networks with stochastic dynamics
    2. Gibbs sampling converges to the Boltzmann distribution at equilibrium
    3. Temperature controls exploration vs exploitation
    4. Hidden units enable learning of latent structure
    5. Training requires balancing data statistics vs model statistics

!!! info "Historical Significance"
    Boltzmann machines established the foundation for:
    - Deep Belief Networks (Hinton, 2006)
    - Restricted Boltzmann Machines
    - Modern energy-based models
    - Variational autoencoders (through free energy concepts)

## Exercises

1. **Mixing Time**: Empirically estimate how many Gibbs steps are needed to reach equilibrium for different network sizes and temperatures.

2. **Annealing**: Implement simulated annealing starting at high $T$ and gradually decreasing. Compare to fixed-temperature sampling.

3. **Hidden Units**: Add hidden units to a Boltzmann machine and analyze how they affect the expressiveness of the visible distribution.

## References

- Hinton, G. E., & Sejnowski, T. J. (1986). Learning and relearning in Boltzmann machines. In Parallel Distributed Processing.
- Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive Science.
- Salakhutdinov, R. (2015). Learning Deep Generative Models. Annual Review of Statistics.
