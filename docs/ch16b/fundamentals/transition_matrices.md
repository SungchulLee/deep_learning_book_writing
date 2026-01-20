# Transition Matrices

## Introduction

The transition matrix is the mathematical heart of a Markov chain, encoding all information about how the system evolves. This section explores the properties of transition matrices, the fundamental Chapman-Kolmogorov equations, and techniques for computing multi-step transition probabilities.

## Transition Matrix Properties

### Definition

For a Markov chain with finite state space $S = \{0, 1, \ldots, N-1\}$, the **transition matrix** $P$ is an $N \times N$ matrix where:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

### Stochastic Matrix Properties

A matrix $P$ is called **(row) stochastic** if:

1. **Non-negativity**: $P_{ij} \geq 0$ for all $i, j$
2. **Row-sum unity**: $\sum_{j=0}^{N-1} P_{ij} = 1$ for all $i$

**Important**: Each row of $P$ is a probability distribution over the next state.

### Matrix Representation

The transition matrix can be visualized as:

$$P = \begin{pmatrix}
P_{00} & P_{01} & \cdots & P_{0,N-1} \\
P_{10} & P_{11} & \cdots & P_{1,N-1} \\
\vdots & \vdots & \ddots & \vdots \\
P_{N-1,0} & P_{N-1,1} & \cdots & P_{N-1,N-1}
\end{pmatrix}$$

- **Row $i$**: Transition probabilities when currently in state $i$
- **Column $j$**: Probabilities of entering state $j$ from each state
- **Diagonal $P_{ii}$**: Probability of staying in state $i$

## n-Step Transition Probabilities

### Definition

The **n-step transition probability** is the probability of being in state $j$ after exactly $n$ steps, starting from state $i$:

$$P^{(n)}_{ij} = P(X_{n+m} = j \mid X_m = i)$$

For time-homogeneous chains, this does not depend on $m$.

### Fundamental Result: Matrix Powers

**Theorem**: The n-step transition probabilities are given by the n-th power of the transition matrix:

$$P^{(n)}_{ij} = (P^n)_{ij}$$

where $P^n$ denotes ordinary matrix multiplication: $P^n = \underbrace{P \cdot P \cdots P}_{n \text{ times}}$.

**Proof Sketch**:
By the law of total probability:
$$P^{(n)}_{ij} = \sum_{k \in S} P^{(n-1)}_{ik} P_{kj}$$

This is exactly the formula for matrix multiplication, so $P^{(n)} = P^{(n-1)} \cdot P$. By induction, $P^{(n)} = P^n$.

## Chapman-Kolmogorov Equations

### Statement

For any integers $m, n \geq 0$:

$$P^{(m+n)}_{ij} = \sum_{k \in S} P^{(m)}_{ik} P^{(n)}_{kj}$$

In matrix form:

$$P^{m+n} = P^m \cdot P^n$$

### Interpretation

To go from state $i$ to state $j$ in $m+n$ steps:
1. First go from $i$ to some intermediate state $k$ in $m$ steps
2. Then go from $k$ to $j$ in $n$ steps
3. Sum over all possible intermediate states $k$

### Visual Example

```
State i → (m steps) → State k → (n steps) → State j

P^{(m+n)}_{ij} = Σ_k [probability of i→k in m steps] × [probability of k→j in n steps]
```

## PyTorch Implementation

### Computing n-Step Transitions

```python
import torch
import torch.linalg as LA
from typing import Tuple, Dict

class TransitionMatrixAnalyzer:
    """
    Tools for analyzing transition matrices and computing
    multi-step transition probabilities.
    """
    
    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: list = None
    ):
        """
        Initialize analyzer.
        
        Args:
            transition_matrix: N×N stochastic matrix P
            state_names: Optional names for states
        """
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"State_{i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
        
        self._validate()
    
    def _validate(self):
        """Validate transition matrix properties."""
        # Check square
        assert self.P.shape[0] == self.P.shape[1], "Matrix must be square"
        
        # Check non-negative
        assert torch.all(self.P >= 0), "All entries must be non-negative"
        
        # Check row sums
        row_sums = self.P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "All rows must sum to 1"
    
    def n_step_matrix(self, n: int) -> torch.Tensor:
        """
        Compute n-step transition matrix P^n.
        
        Uses efficient matrix exponentiation.
        
        Args:
            n: Number of steps
            
        Returns:
            P^n where P^n[i,j] = P(X_n = j | X_0 = i)
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return torch.eye(self.n_states, dtype=self.P.dtype)
        
        return torch.linalg.matrix_power(self.P, n)
    
    def n_step_probability(
        self,
        from_state: int,
        to_state: int,
        n: int
    ) -> float:
        """
        Compute P^{(n)}_{ij} = P(X_n = j | X_0 = i).
        
        Args:
            from_state: Initial state i
            to_state: Target state j
            n: Number of steps
            
        Returns:
            n-step transition probability
        """
        P_n = self.n_step_matrix(n)
        return P_n[from_state, to_state].item()
    
    def distribution_evolution(
        self,
        initial_dist: torch.Tensor,
        n_steps: int
    ) -> torch.Tensor:
        """
        Compute state distribution at each time step.
        
        Given π^{(0)}, computes π^{(1)}, π^{(2)}, ..., π^{(n)}
        where π^{(k)} = π^{(0)} P^k
        
        Args:
            initial_dist: Initial distribution π^{(0)}
            n_steps: Number of steps to compute
            
        Returns:
            Tensor of shape (n_steps+1, n_states) with distributions
        """
        distributions = torch.zeros(n_steps + 1, self.n_states)
        distributions[0] = initial_dist
        
        current_dist = initial_dist.clone()
        for k in range(1, n_steps + 1):
            current_dist = current_dist @ self.P
            distributions[k] = current_dist
        
        return distributions
    
    def chapman_kolmogorov_verify(
        self,
        m: int,
        n: int,
        tol: float = 1e-6
    ) -> bool:
        """
        Verify Chapman-Kolmogorov equation: P^{m+n} = P^m × P^n
        
        Args:
            m, n: Number of steps
            tol: Tolerance for numerical comparison
            
        Returns:
            True if equation holds within tolerance
        """
        P_m = self.n_step_matrix(m)
        P_n = self.n_step_matrix(n)
        P_mn = self.n_step_matrix(m + n)
        
        # Chapman-Kolmogorov: P^{m+n} should equal P^m × P^n
        P_m_times_n = P_m @ P_n
        
        return torch.allclose(P_mn, P_m_times_n, atol=tol)
```

### Manual Chapman-Kolmogorov Computation

```python
def chapman_kolmogorov_manual(
    P: torch.Tensor,
    i: int,
    j: int,
    m: int,
    n: int
) -> Tuple[float, float]:
    """
    Compute P^{(m+n)}_{ij} two ways:
    1. Direct: [P^{m+n}]_{ij}
    2. Chapman-Kolmogorov: Σ_k P^{(m)}_{ik} P^{(n)}_{kj}
    
    Args:
        P: Transition matrix
        i, j: States
        m, n: Number of steps
        
    Returns:
        (direct_result, ck_result) - should be equal
    """
    n_states = P.shape[0]
    
    # Method 1: Direct computation
    P_m_plus_n = torch.linalg.matrix_power(P, m + n)
    direct = P_m_plus_n[i, j].item()
    
    # Method 2: Chapman-Kolmogorov summation
    P_m = torch.linalg.matrix_power(P, m)
    P_n = torch.linalg.matrix_power(P, n)
    
    ck_sum = 0.0
    for k in range(n_states):
        ck_sum += P_m[i, k].item() * P_n[k, j].item()
    
    return direct, ck_sum


# Example verification
P = torch.tensor([
    [0.7, 0.3],
    [0.4, 0.6]
])

print("Verifying Chapman-Kolmogorov: P^{(m+n)}_{ij} = Σ_k P^{(m)}_{ik} P^{(n)}_{kj}")
print("=" * 60)

for m, n in [(1, 1), (2, 3), (5, 5)]:
    direct, ck = chapman_kolmogorov_manual(P, i=0, j=1, m=m, n=n)
    print(f"m={m}, n={n}: Direct = {direct:.8f}, C-K = {ck:.8f}, "
          f"Match: {abs(direct - ck) < 1e-10}")
```

## Convergence Analysis

### Matrix Powers and Convergence

For ergodic Markov chains, the matrix powers $P^n$ converge as $n \to \infty$:

$$\lim_{n \to \infty} P^n = \mathbf{1} \pi$$

where $\mathbf{1}$ is a column vector of ones and $\pi$ is the stationary distribution. This means every row of the limiting matrix equals $\pi$.

### Implementation

```python
def analyze_convergence(
    P: torch.Tensor,
    max_steps: int = 100,
    tol: float = 1e-8
) -> Dict:
    """
    Analyze convergence of P^n as n → ∞.
    
    For ergodic chains, P^n converges to a matrix where
    all rows are identical (the stationary distribution).
    
    Args:
        P: Transition matrix
        max_steps: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Dictionary with convergence information
    """
    results = {
        'converged': False,
        'convergence_step': None,
        'limit_matrix': None,
        'differences': []
    }
    
    P_prev = P.clone()
    
    for step in range(1, max_steps + 1):
        P_current = P_prev @ P
        
        # Maximum absolute difference between consecutive powers
        diff = torch.max(torch.abs(P_current - P_prev)).item()
        results['differences'].append(diff)
        
        if diff < tol:
            results['converged'] = True
            results['convergence_step'] = step
            results['limit_matrix'] = P_current
            
            # Extract stationary distribution (any row of limit)
            results['stationary_distribution'] = P_current[0].clone()
            break
        
        P_prev = P_current
    
    return results


# Example: Convergence analysis
P = torch.tensor([
    [0.5, 0.3, 0.2],
    [0.2, 0.6, 0.2],
    [0.3, 0.3, 0.4]
])

results = analyze_convergence(P, max_steps=100, tol=1e-10)

print("Convergence Analysis")
print("=" * 50)
print(f"Converged: {results['converged']}")
print(f"Steps to converge: {results['convergence_step']}")
print(f"\nStationary distribution π:")
pi = results['stationary_distribution']
for i, p in enumerate(pi):
    print(f"  π[{i}] = {p:.8f}")

print(f"\nLimit matrix (all rows ≈ π):")
print(results['limit_matrix'])
```

## Visualization

### Transition Matrix Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_transition_matrix(
    P: torch.Tensor,
    state_names: list = None,
    title: str = "Transition Matrix"
):
    """
    Create heatmap visualization of transition matrix.
    
    Args:
        P: Transition matrix
        state_names: State labels
        title: Plot title
    """
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(P.numpy(), cmap='YlOrRd', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_names, fontsize=11)
    ax.set_yticklabels(state_names, fontsize=11)
    
    # Labels
    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            value = P[i, j].item()
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_distribution_evolution(
    distributions: torch.Tensor,
    state_names: list = None,
    title: str = "Distribution Evolution"
):
    """
    Plot evolution of state distribution over time.
    
    Args:
        distributions: (n_steps+1, n_states) tensor
        state_names: State labels
        title: Plot title
    """
    n_steps, n_states = distributions.shape
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = range(n_steps)
    
    for i in range(n_states):
        ax.plot(steps, distributions[:, i].numpy(),
               marker='o', markersize=4, linewidth=2,
               label=state_names[i], alpha=0.8)
    
    ax.set_xlabel('Time Step n', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig
```

## Example: Computing Multi-Step Probabilities

```python
# Three-state weather model
states = ['Sunny', 'Cloudy', 'Rainy']
P = torch.tensor([
    [0.7, 0.25, 0.05],  # From Sunny
    [0.3, 0.40, 0.30],  # From Cloudy
    [0.1, 0.40, 0.50]   # From Rainy
])

analyzer = TransitionMatrixAnalyzer(P, state_names=states)

print("Weather Model: Multi-Step Transition Probabilities")
print("=" * 60)

# Display P, P^2, P^5, P^10
for n in [1, 2, 5, 10, 50]:
    P_n = analyzer.n_step_matrix(n)
    print(f"\n{n}-Step Transition Matrix P^{n}:")
    print("-" * 40)
    
    header = "         " + "  ".join(f"{s:>8}" for s in states)
    print(header)
    
    for i, state_i in enumerate(states):
        row = f"{state_i:8s} " + "  ".join(f"{P_n[i,j]:.6f}" for j in range(3))
        print(row)

# Specific probability queries
print("\n" + "=" * 60)
print("Specific Probability Queries:")
print("-" * 40)

queries = [
    (0, 2, 1),   # P(Rainy after 1 day | start Sunny)
    (0, 2, 5),   # P(Rainy after 5 days | start Sunny)  
    (2, 0, 3),   # P(Sunny after 3 days | start Rainy)
]

for from_idx, to_idx, n in queries:
    prob = analyzer.n_step_probability(from_idx, to_idx, n)
    print(f"P({states[to_idx]} after {n} days | {states[from_idx]}) = {prob:.6f}")
```

## Summary

| Concept | Formula | Description |
|---------|---------|-------------|
| **Transition Matrix** | $P_{ij} = P(X_{n+1}=j \mid X_n=i)$ | One-step transitions |
| **n-Step Matrix** | $P^n = P \cdot P \cdots P$ | n-fold matrix multiplication |
| **Chapman-Kolmogorov** | $P^{m+n} = P^m \cdot P^n$ | Composition of transitions |
| **Distribution Evolution** | $\pi^{(n)} = \pi^{(0)} P^n$ | State probabilities over time |
| **Convergence** | $\lim_{n\to\infty} P^n = \mathbf{1}\pi$ | Convergence to stationary |

## Exercises

1. **Matrix Powers**: For a 2×2 transition matrix, derive a closed-form expression for $P^n$ using eigendecomposition.

2. **Verification**: Implement a function to verify the Chapman-Kolmogorov equations numerically for random transition matrices.

3. **Convergence Rate**: Investigate how the convergence rate of $P^n$ depends on the second-largest eigenvalue of $P$.

4. **Application**: Model a stock price that moves up, stays same, or moves down with given probabilities. Compute the probability distribution after 10 trading days.

## References

1. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, Chapter 3. Springer-Verlag, 1976.
2. Norris, J.R. *Markov Chains*, Chapter 1. Cambridge University Press, 1997.
3. Horn, R.A. & Johnson, C.R. *Matrix Analysis*, Chapter 8. Cambridge University Press, 2012.
