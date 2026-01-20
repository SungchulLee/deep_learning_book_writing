# Stationary Distributions

## Introduction

The stationary distribution is one of the most important concepts in Markov chain theory. It represents the long-run behavior of the chain—the proportion of time spent in each state as time goes to infinity. Understanding how to compute and interpret stationary distributions is essential for analyzing systems modeled by Markov chains.

## Definition and Properties

### Mathematical Definition

A probability distribution $\pi = (\pi_0, \pi_1, \ldots, \pi_{N-1})$ is a **stationary distribution** (or invariant distribution, equilibrium distribution) for a Markov chain with transition matrix $P$ if:

$$\pi = \pi P$$

Component-wise:

$$\pi_j = \sum_{i \in S} \pi_i P_{ij} \quad \text{for all } j \in S$$

### Interpretation

If the chain starts with distribution $\pi$ (i.e., $P(X_0 = i) = \pi_i$), then:
- At time 1: $P(X_1 = j) = \sum_i \pi_i P_{ij} = \pi_j$
- At time $n$: $P(X_n = j) = \pi_j$

The distribution remains unchanged over time—hence "stationary."

### Physical Interpretation

For ergodic chains, $\pi_i$ equals:
1. **Long-run proportion**: Fraction of time spent in state $i$ as $n \to \infty$
2. **Limiting probability**: $\lim_{n \to \infty} P(X_n = i)$ regardless of starting state
3. **Reciprocal of mean return time**: $\pi_i = 1/E[T_i]$ where $T_i$ is the first return time to state $i$

## Existence and Uniqueness

### Existence

**Theorem**: Every finite Markov chain has at least one stationary distribution.

**Proof sketch**: The set of probability distributions is compact, and the mapping $\pi \mapsto \pi P$ is continuous. By Brouwer's fixed point theorem, a fixed point exists.

### Uniqueness

**Theorem**: An irreducible Markov chain has exactly one stationary distribution.

**Theorem (Convergence)**: If the chain is also aperiodic (i.e., ergodic), then for any initial distribution:

$$\lim_{n \to \infty} P^n_{ij} = \pi_j \quad \text{for all } i, j$$

## Four Methods for Computing $\pi$

### Method 1: Eigenvector Method

Since $\pi P = \pi$, we have $\pi^T = P^T \pi^T$, so $\pi^T$ is a right eigenvector of $P^T$ with eigenvalue 1.

**Algorithm**:
1. Compute eigenvalues and eigenvectors of $P^T$
2. Find eigenvector corresponding to eigenvalue 1
3. Normalize to sum to 1

### Method 2: Linear System Solution

The equation $\pi P = \pi$ can be rewritten as $\pi(P - I) = 0$.

**Algorithm**:
1. Set up system: $(P^T - I)\pi^T = 0$
2. Replace one equation with normalization: $\sum_i \pi_i = 1$
3. Solve the resulting linear system

### Method 3: Power Iteration

For ergodic chains, $P^n$ converges to a matrix with all rows equal to $\pi$.

**Algorithm**:
1. Start with any initial distribution $\pi^{(0)}$
2. Iterate: $\pi^{(k+1)} = \pi^{(k)} P$
3. Continue until convergence: $\|\pi^{(k+1)} - \pi^{(k)}\| < \epsilon$

### Method 4: Simulation (Ergodic Theorem)

For ergodic chains, the time-average converges to the space-average.

**Algorithm**:
1. Simulate the chain for $T$ steps
2. Count visits to each state: $N_i = \sum_{t=0}^{T} \mathbf{1}\{X_t = i\}$
3. Estimate: $\hat{\pi}_i = N_i / T$

## PyTorch Implementation

```python
import torch
import torch.linalg as LA
from typing import Dict, Tuple, Optional
import numpy as np

class StationaryDistributionAnalyzer:
    """
    Compute stationary distributions using multiple methods.
    
    A stationary distribution π satisfies: π = πP
    This means π is a left eigenvector of P with eigenvalue 1.
    """
    
    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: list = None
    ):
        """
        Initialize analyzer.
        
        Args:
            transition_matrix: N×N stochastic matrix
            state_names: Optional state names
        """
        self.P = transition_matrix.clone().double()  # Use double precision
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"State_{i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
    
    def via_eigenvector(self) -> torch.Tensor:
        """
        Compute stationary distribution via eigenvector method.
        
        Mathematical basis:
        πP = π implies P^T π^T = π^T
        So π^T is a right eigenvector of P^T with eigenvalue 1.
        
        Returns:
            Stationary distribution π
        """
        # Compute eigenvalues and eigenvectors of P^T
        eigenvalues, eigenvectors = LA.eig(self.P.T)
        
        # Find index of eigenvalue closest to 1
        idx = torch.argmin(torch.abs(eigenvalues.real - 1.0))
        
        # Extract corresponding eigenvector
        pi = eigenvectors[:, idx].real
        
        # Normalize to sum to 1
        pi = torch.abs(pi)  # Ensure positive
        pi = pi / pi.sum()
        
        return pi.float()
    
    def via_linear_system(self) -> torch.Tensor:
        """
        Compute stationary distribution by solving linear system.
        
        Mathematical setup:
        πP = π  →  π(P - I) = 0  →  (P^T - I)π^T = 0
        Plus normalization: Σπ_i = 1
        
        We replace one equation with the normalization constraint.
        
        Returns:
            Stationary distribution π
        """
        # Set up system (P^T - I)π^T = 0
        A = self.P.T - torch.eye(self.n_states, dtype=self.P.dtype)
        
        # Replace last row with normalization constraint
        A[-1, :] = torch.ones(self.n_states, dtype=self.P.dtype)
        
        # Right-hand side
        b = torch.zeros(self.n_states, dtype=self.P.dtype)
        b[-1] = 1.0
        
        # Solve linear system
        pi = LA.solve(A, b)
        
        return pi.float()
    
    def via_power_iteration(
        self,
        max_iter: int = 1000,
        tol: float = 1e-10
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute stationary distribution via matrix power iteration.
        
        For ergodic chains, P^n converges to a matrix where all rows
        equal the stationary distribution π.
        
        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            (stationary distribution, number of iterations)
        """
        P_n = self.P.clone()
        
        for n in range(1, max_iter + 1):
            P_next = P_n @ self.P
            
            # Check convergence (all rows should be nearly identical)
            max_diff = torch.max(torch.abs(P_next - P_n))
            
            if max_diff < tol:
                # Extract stationary distribution (any row)
                return P_next[0].float(), n
            
            P_n = P_next
        
        # Return best estimate if not converged
        return P_n[0].float(), max_iter
    
    def via_simulation(
        self,
        n_steps: int = 100000,
        initial_state: int = 0,
        burn_in: int = 1000
    ) -> torch.Tensor:
        """
        Estimate stationary distribution via long-run simulation.
        
        By the ergodic theorem:
        lim_{T→∞} (1/T) Σ_{t=0}^{T} 1{X_t = j} = π_j
        
        Args:
            n_steps: Number of simulation steps
            initial_state: Starting state
            burn_in: Steps to discard (allow mixing)
            
        Returns:
            Estimated stationary distribution
        """
        state_counts = torch.zeros(self.n_states)
        current_state = initial_state
        
        for step in range(n_steps + burn_in):
            # Count after burn-in period
            if step >= burn_in:
                state_counts[current_state] += 1
            
            # Transition to next state
            probs = self.P[current_state].float()
            current_state = torch.multinomial(probs, num_samples=1).item()
        
        # Normalize to get probabilities
        return state_counts / state_counts.sum()
    
    def compare_all_methods(
        self,
        n_simulation_steps: int = 100000
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stationary distribution using all four methods.
        
        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        # Method 1: Eigenvector
        results['eigenvector'] = self.via_eigenvector()
        
        # Method 2: Linear system
        results['linear_system'] = self.via_linear_system()
        
        # Method 3: Power iteration
        pi_power, iterations = self.via_power_iteration()
        results['power_iteration'] = pi_power
        results['power_iterations_count'] = iterations
        
        # Method 4: Simulation
        results['simulation'] = self.via_simulation(n_steps=n_simulation_steps)
        
        return results
    
    def verify_stationary(
        self,
        pi: torch.Tensor,
        tol: float = 1e-6
    ) -> Dict:
        """
        Verify that π is indeed a stationary distribution.
        
        Checks:
        1. π is a valid probability distribution (non-negative, sums to 1)
        2. πP = π (fixed point equation)
        
        Args:
            pi: Candidate stationary distribution
            tol: Tolerance for numerical checks
            
        Returns:
            Verification results
        """
        results = {}
        
        # Check valid probability distribution
        results['is_non_negative'] = torch.all(pi >= -tol).item()
        results['sums_to_one'] = torch.abs(pi.sum() - 1.0).item() < tol
        
        # Check fixed point equation: πP should equal π
        pi_P = pi.double() @ self.P
        fixed_point_error = torch.max(torch.abs(pi_P - pi.double())).item()
        results['fixed_point_error'] = fixed_point_error
        results['is_stationary'] = fixed_point_error < tol
        
        return results


def demonstrate_stationary_distribution():
    """
    Demonstrate computation and verification of stationary distributions.
    """
    # Example: Three-state weather model
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = torch.tensor([
        [0.7, 0.25, 0.05],
        [0.3, 0.40, 0.30],
        [0.1, 0.40, 0.50]
    ])
    
    analyzer = StationaryDistributionAnalyzer(P, state_names=states)
    
    print("Stationary Distribution Analysis")
    print("=" * 70)
    print("\nTransition Matrix:")
    print(P)
    
    # Compare all methods
    results = analyzer.compare_all_methods(n_simulation_steps=100000)
    
    print("\n" + "-" * 70)
    print("Method Comparison:")
    print("-" * 70)
    
    header = f"{'State':<12}" + "".join(f"{method:<15}" 
                                        for method in ['Eigenvector', 'Linear Sys', 
                                                       'Power Iter', 'Simulation'])
    print(header)
    
    for i, state in enumerate(states):
        row = f"{state:<12}"
        row += f"{results['eigenvector'][i]:<15.8f}"
        row += f"{results['linear_system'][i]:<15.8f}"
        row += f"{results['power_iteration'][i]:<15.8f}"
        row += f"{results['simulation'][i]:<15.8f}"
        print(row)
    
    print(f"\nPower iteration converged in {results['power_iterations_count']} iterations")
    
    # Verify
    print("\n" + "-" * 70)
    print("Verification (using eigenvector result):")
    verification = analyzer.verify_stationary(results['eigenvector'])
    print(f"  Non-negative: {verification['is_non_negative']}")
    print(f"  Sums to one: {verification['sums_to_one']}")
    print(f"  Fixed point error ||πP - π||: {verification['fixed_point_error']:.2e}")
    print(f"  Is stationary: {verification['is_stationary']}")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("Interpretation:")
    pi = results['eigenvector']
    print(f"  In the long run:")
    for i, state in enumerate(states):
        pct = pi[i].item() * 100
        print(f"    {state}: {pct:.2f}% of the time")


# Run demonstration
demonstrate_stationary_distribution()
```

## Convergence to Stationary Distribution

### Rate of Convergence

The convergence rate is controlled by the **spectral gap**:

$$\gamma = 1 - |\lambda_2|$$

where $\lambda_2$ is the second-largest eigenvalue of $P$ (in absolute value).

**Theorem**: For ergodic chains:

$$\|P^n_{i,\cdot} - \pi\|_{TV} \leq C \cdot |\lambda_2|^n$$

Larger spectral gap → faster convergence.

### Visualization

```python
def visualize_convergence_to_stationary(
    P: torch.Tensor,
    state_names: list = None,
    max_steps: int = 50
):
    """
    Visualize how distributions converge to stationary distribution.
    """
    import matplotlib.pyplot as plt
    
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]
    
    # Compute stationary distribution
    analyzer = StationaryDistributionAnalyzer(P, state_names)
    pi_stationary = analyzer.via_eigenvector()
    
    # Track convergence from different starting points
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Distance to stationary distribution
    ax1 = axes[0]
    
    for start_state in range(n_states):
        # Initial distribution: 100% in start_state
        dist = torch.zeros(n_states)
        dist[start_state] = 1.0
        
        distances = []
        for step in range(max_steps):
            # Total variation distance to stationary
            tv_dist = 0.5 * torch.sum(torch.abs(dist - pi_stationary)).item()
            distances.append(tv_dist)
            
            # Update distribution: π^{(k+1)} = π^{(k)} P
            dist = dist @ P
        
        ax1.semilogy(distances, marker='o', markersize=3, 
                    label=f'Start: {state_names[start_state]}',
                    linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Time Step n', fontsize=12)
    ax1.set_ylabel('Total Variation Distance (log scale)', fontsize=12)
    ax1.set_title('Convergence to Stationary Distribution', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Component-wise evolution
    ax2 = axes[1]
    
    # Start from state 0
    dist = torch.zeros(n_states)
    dist[0] = 1.0
    
    distributions = [dist.clone()]
    for step in range(max_steps):
        dist = dist @ P
        distributions.append(dist.clone())
    
    distributions = torch.stack(distributions)
    
    for i in range(n_states):
        ax2.plot(distributions[:, i].numpy(), marker='o', markersize=3,
                label=state_names[i], linewidth=2, alpha=0.7)
        # Add horizontal line for stationary value
        ax2.axhline(y=pi_stationary[i].item(), linestyle='--', 
                   color=f'C{i}', alpha=0.4)
    
    ax2.set_xlabel('Time Step n', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f'Component Evolution (Start: {state_names[0]})', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Example visualization
P = torch.tensor([
    [0.5, 0.3, 0.2],
    [0.2, 0.6, 0.2],
    [0.3, 0.3, 0.4]
])

fig = visualize_convergence_to_stationary(P, ['A', 'B', 'C'], max_steps=30)
plt.savefig('stationary_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
```

## Ergodicity Requirements

### When Does a Unique Stationary Distribution Exist?

| Condition | Unique $\pi$? | Convergence? |
|-----------|---------------|--------------|
| Irreducible only | ✓ Yes | ✗ No (may oscillate) |
| Aperiodic only | ✗ May have multiple | Depends |
| **Ergodic** (both) | ✓ Yes | ✓ Yes |
| Reducible | Multiple possible | No (depends on start) |

### Checking Ergodicity

```python
def check_ergodicity_for_stationary(P: torch.Tensor) -> Dict:
    """
    Check if chain is ergodic (guarantees unique stationary distribution
    and convergence).
    """
    n_states = P.shape[0]
    
    # Check irreducibility: sum of P^k for k=1 to n should have all positive entries
    P_sum = torch.zeros_like(P)
    P_k = P.clone()
    
    for k in range(n_states):
        P_sum += P_k
        P_k = P_k @ P
    
    is_irreducible = torch.all(P_sum > 0).item()
    
    # Check aperiodicity: sufficient if any diagonal entry > 0
    has_self_loop = torch.any(torch.diag(P) > 0).item()
    
    return {
        'is_irreducible': is_irreducible,
        'is_aperiodic': has_self_loop,
        'is_ergodic': is_irreducible and has_self_loop,
        'unique_stationary_guaranteed': is_irreducible,
        'convergence_guaranteed': is_irreducible and has_self_loop
    }
```

## Summary

| Method | Computation | Pros | Cons |
|--------|-------------|------|------|
| **Eigenvector** | $P^T v = v$, normalize | Exact, fast | Numerical issues for large matrices |
| **Linear System** | $(P^T - I)\pi^T = 0$ | Exact, stable | Requires matrix inversion |
| **Power Iteration** | $\pi^{(k+1)} = \pi^{(k)}P$ | Simple, intuitive | Slow for small spectral gap |
| **Simulation** | Count state visits | Works for large state spaces | Only approximate, needs long runs |

## Exercises

1. **Multiple Stationary**: Construct a reducible Markov chain with two distinct stationary distributions.

2. **Convergence Rate**: For a given transition matrix, compute the spectral gap and verify that convergence rate matches the theory.

3. **Application**: Model customer loyalty with states {Loyal, Neutral, Churned}. Compute the long-run proportion of customers in each state.

4. **Numerical Stability**: Compare the four methods on a nearly-reducible chain (some transition probabilities close to 0).

## References

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times*, Chapters 1-4. AMS, 2017.
2. Norris, J.R. *Markov Chains*, Chapter 1. Cambridge University Press, 1997.
3. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, Chapter 4. Springer-Verlag, 1976.
