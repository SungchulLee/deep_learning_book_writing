# Convergence Theorems and Mixing Times

## Introduction

Understanding when and how fast a Markov chain converges to its stationary distribution is fundamental for both theoretical analysis and practical applications. This section covers the key convergence theorems and introduces mixing times—a crucial concept for MCMC methods.

## Ergodic Theorem

### Statement

**Fundamental Convergence Theorem**: Let $P$ be the transition matrix of an **ergodic** (irreducible and aperiodic) Markov chain with stationary distribution $\pi$. Then for any initial distribution:

$$\lim_{n \to \infty} P^n_{ij} = \pi_j \quad \text{for all } i, j$$

Moreover, the convergence is **exponential**:

$$|P^n_{ij} - \pi_j| \leq C \cdot \rho^n$$

where $\rho = |\lambda_2| < 1$ is the second-largest eigenvalue magnitude.

### Implications

1. **Independence from Initial State**: The long-run behavior doesn't depend on where the chain started
2. **Unique Equilibrium**: There is exactly one stationary distribution
3. **Geometric Convergence**: Convergence happens exponentially fast

## Spectral Gap

### Definition

The **spectral gap** of a transition matrix $P$ is:

$$\gamma = 1 - |\lambda_2|$$

where $\lambda_2$ is the second-largest eigenvalue in absolute value (noting $\lambda_1 = 1$ always).

### Significance

- **Large spectral gap** → Fast convergence (good mixing)
- **Small spectral gap** → Slow convergence (poor mixing)
- $\gamma = 0$ implies the chain doesn't converge

### Relationship to Structure

| Chain Structure | Spectral Gap | Mixing |
|----------------|--------------|--------|
| Well-connected | Large | Fast |
| Bottleneck (weak links) | Small | Slow |
| Nearly periodic | Very small | Very slow |

## Mixing Time

### Definition

The **mixing time** quantifies how long until the chain is "close" to stationarity.

**Total Variation Distance**:
$$\|P^n(x, \cdot) - \pi\|_{TV} = \frac{1}{2} \sum_{y \in S} |P^n_{xy} - \pi_y|$$

**ε-Mixing Time**:
$$\tau_{mix}(\epsilon) = \min\{n : \max_x \|P^n(x, \cdot) - \pi\|_{TV} \leq \epsilon\}$$

The standard choice is $\epsilon = 1/4$, written simply as $\tau_{mix}$.

### Bounds

The mixing time is bounded by the spectral gap:

$$\frac{1}{\gamma} \leq \tau_{mix} \leq \frac{\log(1/\epsilon\pi_{min})}{\gamma}$$

where $\pi_{min} = \min_i \pi_i$.

## PyTorch Implementation

```python
import torch
import torch.linalg as LA
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class MixingTimeAnalyzer:
    """
    Analyze convergence and mixing properties of Markov chains.
    
    Key concepts:
    - Spectral gap: γ = 1 - |λ₂|
    - Mixing time: steps until close to stationary
    - Total variation distance: measure of distributional difference
    """
    
    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[List[str]] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            transition_matrix: N×N stochastic matrix
            state_names: Optional state names
        """
        self.P = transition_matrix.clone().double()
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"S{i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
        
        # Compute eigenvalues
        self._compute_eigenvalues()
        
        # Compute stationary distribution
        self._compute_stationary()
    
    def _compute_eigenvalues(self):
        """Compute eigenvalues of transition matrix."""
        self.eigenvalues, self.eigenvectors = LA.eig(self.P)
        
        # Sort by absolute value (descending)
        abs_vals = torch.abs(self.eigenvalues.real)
        sorted_idx = torch.argsort(abs_vals, descending=True)
        
        self.eigenvalues = self.eigenvalues[sorted_idx]
        self.eigenvectors = self.eigenvectors[:, sorted_idx]
    
    def _compute_stationary(self):
        """Compute stationary distribution."""
        # Find eigenvector with eigenvalue 1
        idx = torch.argmin(torch.abs(self.eigenvalues.real - 1.0))
        pi = self.eigenvectors[:, idx].real
        pi = torch.abs(pi)
        self.pi = pi / pi.sum()
    
    def spectral_gap(self) -> float:
        """
        Compute spectral gap γ = 1 - |λ₂|.
        
        Returns:
            Spectral gap (larger = faster mixing)
        """
        # Second eigenvalue (first is 1)
        lambda_2 = self.eigenvalues[1]
        
        return (1 - torch.abs(lambda_2)).item()
    
    def total_variation_distance(
        self,
        dist1: torch.Tensor,
        dist2: torch.Tensor
    ) -> float:
        """
        Compute total variation distance between two distributions.
        
        TV(μ, ν) = (1/2) Σ |μ(x) - ν(x)|
        
        Args:
            dist1, dist2: Probability distributions
            
        Returns:
            Total variation distance in [0, 1]
        """
        return 0.5 * torch.sum(torch.abs(dist1 - dist2)).item()
    
    def mixing_time(
        self,
        epsilon: float = 0.25,
        max_steps: int = 10000
    ) -> Dict:
        """
        Compute mixing time τ_mix(ε).
        
        Mixing time is the first n such that for ALL starting states,
        the TV distance to π is at most ε.
        
        Args:
            epsilon: Target distance (default 1/4)
            max_steps: Maximum steps to check
            
        Returns:
            Dictionary with mixing time and related info
        """
        results = {
            'epsilon': epsilon,
            'mixing_time': None,
            'distances_by_start': {}
        }
        
        # Track TV distance for each starting state
        max_distances = []
        
        for step in range(max_steps):
            P_n = torch.linalg.matrix_power(self.P, step)
            
            # Maximum TV distance over all starting states
            max_tv = 0
            for i in range(self.n_states):
                tv = self.total_variation_distance(P_n[i], self.pi)
                max_tv = max(max_tv, tv)
            
            max_distances.append(max_tv)
            
            # Check if mixed
            if max_tv <= epsilon and results['mixing_time'] is None:
                results['mixing_time'] = step
                # Continue to collect more data points
        
        results['max_tv_over_time'] = max_distances
        
        return results
    
    def convergence_rate(self, max_steps: int = 100) -> Dict:
        """
        Analyze convergence rate from different starting states.
        
        Returns:
            Dictionary with convergence data
        """
        results = {
            'spectral_gap': self.spectral_gap(),
            'second_eigenvalue': self.eigenvalues[1].item(),
            'theoretical_rate': torch.abs(self.eigenvalues[1]).item(),
            'distances': {}
        }
        
        # Compute TV distances for each starting state
        for start_idx in range(self.n_states):
            distances = []
            dist = torch.zeros(self.n_states, dtype=self.P.dtype)
            dist[start_idx] = 1.0
            
            for step in range(max_steps):
                tv = self.total_variation_distance(dist.float(), self.pi.float())
                distances.append(tv)
                dist = dist @ self.P
            
            results['distances'][self.state_names[start_idx]] = distances
        
        return results
    
    def check_ergodicity(self) -> Dict[str, bool]:
        """
        Check ergodicity conditions.
        
        Returns:
            Dictionary with irreducibility and aperiodicity status
        """
        # Check irreducibility via sum of powers
        P_sum = torch.zeros_like(self.P)
        P_k = self.P.clone()
        
        for _ in range(self.n_states):
            P_sum += P_k
            P_k = P_k @ self.P
        
        is_irreducible = torch.all(P_sum > 1e-10).item()
        
        # Check aperiodicity
        has_self_loop = torch.any(torch.diag(self.P) > 0).item()
        
        # More thorough periodicity check
        is_aperiodic = has_self_loop
        if not has_self_loop:
            # Check if any power less than n has positive diagonal
            P_k = self.P.clone()
            for k in range(2, self.n_states + 1):
                if torch.any(torch.diag(P_k) > 1e-10):
                    is_aperiodic = True
                    break
                P_k = P_k @ self.P
        
        return {
            'is_irreducible': is_irreducible,
            'is_aperiodic': is_aperiodic,
            'is_ergodic': is_irreducible and is_aperiodic
        }


def demonstrate_mixing_analysis():
    """
    Demonstrate mixing time analysis with fast vs slow mixing chains.
    """
    print("Mixing Time Analysis")
    print("=" * 70)
    
    # Example 1: Fast mixing chain (well-connected)
    print("\n1. Fast Mixing Chain (Large Spectral Gap)")
    print("-" * 50)
    
    P_fast = torch.tensor([
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ])
    
    analyzer_fast = MixingTimeAnalyzer(P_fast, ['A', 'B', 'C'])
    
    print(f"Transition Matrix:\n{P_fast}")
    print(f"\nSpectral gap: {analyzer_fast.spectral_gap():.6f}")
    print(f"Second eigenvalue: {analyzer_fast.eigenvalues[1].real:.6f}")
    
    ergodic = analyzer_fast.check_ergodicity()
    print(f"Ergodic: {ergodic['is_ergodic']}")
    
    mixing = analyzer_fast.mixing_time(epsilon=0.01)
    print(f"Mixing time (ε=0.01): {mixing['mixing_time']} steps")
    
    # Example 2: Slow mixing chain (bottleneck structure)
    print("\n" + "=" * 70)
    print("\n2. Slow Mixing Chain (Bottleneck Structure)")
    print("-" * 50)
    
    # Two clusters weakly connected
    P_slow = torch.tensor([
        [0.45, 0.45, 0.05, 0.05],
        [0.45, 0.45, 0.05, 0.05],
        [0.05, 0.05, 0.45, 0.45],
        [0.05, 0.05, 0.45, 0.45]
    ])
    
    analyzer_slow = MixingTimeAnalyzer(P_slow, ['A1', 'A2', 'B1', 'B2'])
    
    print(f"Transition Matrix (Two clusters with weak connections):\n{P_slow}")
    print(f"\nSpectral gap: {analyzer_slow.spectral_gap():.6f}")
    print(f"Second eigenvalue: {analyzer_slow.eigenvalues[1].real:.6f}")
    
    mixing_slow = analyzer_slow.mixing_time(epsilon=0.01, max_steps=500)
    print(f"Mixing time (ε=0.01): {mixing_slow['mixing_time']} steps")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison:")
    print(f"  Fast chain spectral gap: {analyzer_fast.spectral_gap():.6f}")
    print(f"  Slow chain spectral gap: {analyzer_slow.spectral_gap():.6f}")
    print(f"  Ratio: {analyzer_fast.spectral_gap() / analyzer_slow.spectral_gap():.1f}x")


demonstrate_mixing_analysis()
```

## Visualization

```python
def visualize_convergence(
    P: torch.Tensor,
    state_names: List[str] = None,
    max_steps: int = 50
):
    """
    Visualize convergence properties of a Markov chain.
    """
    analyzer = MixingTimeAnalyzer(P, state_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: TV distance over time (log scale)
    ax = axes[0, 0]
    conv_data = analyzer.convergence_rate(max_steps)
    
    for state, distances in conv_data['distances'].items():
        ax.semilogy(distances, marker='o', markersize=3,
                   label=f'Start: {state}', linewidth=2, alpha=0.7)
    
    # Add theoretical decay line
    rate = conv_data['theoretical_rate']
    theoretical = [rate ** n for n in range(max_steps)]
    ax.semilogy(theoretical, 'k--', linewidth=2, alpha=0.5,
               label=f'Theoretical: {rate:.3f}^n')
    
    ax.set_xlabel('Time Step n', fontsize=12)
    ax.set_ylabel('Total Variation Distance', fontsize=12)
    ax.set_title('Convergence to Stationary (Log Scale)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Eigenvalue spectrum
    ax = axes[0, 1]
    
    eigenvalues = analyzer.eigenvalues
    
    # Plot on complex plane
    ax.scatter(eigenvalues.real.numpy(), eigenvalues.imag.numpy(),
              s=200, c='blue', alpha=0.7, edgecolors='black', linewidth=2)
    
    # Highlight λ₁ = 1 and λ₂
    ax.scatter([1], [0], s=300, c='red', marker='*',
              label='λ₁ = 1', zorder=5)
    ax.scatter([eigenvalues[1].real.item()], [eigenvalues[1].imag.item()],
              s=300, c='green', marker='*',
              label=f'λ₂ = {eigenvalues[1].real:.3f}', zorder=5)
    
    # Unit circle
    theta = torch.linspace(0, 2 * 3.14159, 100)
    ax.plot(torch.cos(theta).numpy(), torch.sin(theta).numpy(),
           'k--', alpha=0.3, label='Unit circle')
    
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    ax.set_title(f'Eigenvalue Spectrum (gap = {analyzer.spectral_gap():.4f})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Plot 3: State distribution evolution
    ax = axes[1, 0]
    
    n_states = P.shape[0]
    state_names = state_names or [f'S{i}' for i in range(n_states)]
    
    # Start from state 0
    dist = torch.zeros(n_states, dtype=P.dtype)
    dist[0] = 1.0
    
    evolution = [dist.clone()]
    for _ in range(max_steps):
        dist = dist @ P
        evolution.append(dist.clone())
    
    evolution = torch.stack(evolution)
    
    for i in range(n_states):
        ax.plot(evolution[:, i].numpy(), marker='o', markersize=3,
               label=state_names[i], linewidth=2, alpha=0.7)
        ax.axhline(y=analyzer.pi[i].item(), linestyle='--',
                  color=f'C{i}', alpha=0.4)
    
    ax.set_xlabel('Time Step n', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Distribution Evolution (Start: {state_names[0]})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mixing time analysis
    ax = axes[1, 1]
    
    mixing_data = analyzer.mixing_time(epsilon=0.25, max_steps=max_steps)
    max_tv = mixing_data['max_tv_over_time']
    
    ax.semilogy(max_tv, 'b-', linewidth=2, label='Max TV distance')
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2,
              label='ε = 0.25 threshold')
    
    if mixing_data['mixing_time'] is not None:
        ax.axvline(x=mixing_data['mixing_time'], color='green',
                  linestyle='--', linewidth=2,
                  label=f'τ_mix = {mixing_data["mixing_time"]}')
    
    ax.set_xlabel('Time Step n', fontsize=12)
    ax.set_ylabel('Max TV Distance', fontsize=12)
    ax.set_title('Mixing Time Analysis', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Key Theorems Summary

### Theorem 1: Existence of Stationary Distribution
Every finite Markov chain has at least one stationary distribution.

### Theorem 2: Uniqueness (Irreducibility)
An irreducible Markov chain has exactly one stationary distribution.

### Theorem 3: Convergence (Ergodicity)
An ergodic (irreducible + aperiodic) Markov chain converges:
$$\lim_{n \to \infty} P^n_{ij} = \pi_j \quad \forall i, j$$

### Theorem 4: Exponential Convergence Rate
$$\|P^n(x, \cdot) - \pi\|_{TV} \leq C \cdot |\lambda_2|^n$$

### Theorem 5: Mixing Time Bounds
$$\frac{1}{\gamma} \leq \tau_{mix} \leq O\left(\frac{\log n}{\gamma}\right)$$

## Why Mixing Time Matters for MCMC

In Markov Chain Monte Carlo (MCMC):
1. We construct a chain with desired stationary distribution $\pi$
2. We run the chain to sample from $\pi$
3. Samples are approximately from $\pi$ only after $\tau_{mix}$ steps

**Practical Implications**:
- **Burn-in**: Discard first $\sim \tau_{mix}$ samples
- **Thinning**: Keep every $k$-th sample where $k \sim \tau_{mix}$
- **Efficiency**: Fast mixing = more independent samples

## Summary

| Concept | Definition | Importance |
|---------|------------|------------|
| **Spectral Gap** | $\gamma = 1 - \|\lambda_2\|$ | Controls convergence rate |
| **Mixing Time** | Steps until TV ≤ ε | Practical convergence measure |
| **Ergodicity** | Irreducible + Aperiodic | Guarantees convergence |
| **TV Distance** | $\frac{1}{2}\sum\|p_i - q_i\|$ | Measures distributional difference |

## Exercises

1. **Spectral Gap Computation**: For a 3×3 transition matrix of your choice, compute the spectral gap and verify that it predicts the convergence rate.

2. **Bottleneck Effect**: Construct a chain with two clusters. Show that reducing the inter-cluster transition probability increases mixing time.

3. **MCMC Preview**: Given a target distribution, design a transition matrix that has it as its stationary distribution.

## References

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times* (2nd ed.). AMS, 2017.
2. Montenegro, R. & Tetali, P. "Mathematical Aspects of Mixing Times in Markov Chains." *Foundations and Trends in Theoretical Computer Science*, 2006.
3. Diaconis, P. & Stroock, D. "Geometric Bounds for Eigenvalues of Markov Chains." *Annals of Applied Probability*, 1991.
