# Markov Chain Fundamentals

## Introduction

Markov chains are the theoretical backbone of all Markov Chain Monte Carlo (MCMC) methods. Before constructing samplers that converge to target distributions, we must understand the mechanics of how Markov chains evolve, how transition probabilities compose over multiple steps, and how the dynamics are fully encoded in a single matrix. This section establishes the core definitions and computational tools that underpin the rest of Chapter 18.

## The Markov Property

### Formal Definition

A discrete-time stochastic process $\{X_n\}_{n \geq 0}$ with state space $S$ is a **Markov chain** if for all $n \geq 0$ and all states $i_0, i_1, \ldots, i_{n-1}, i, j \in S$:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

This is the **Markov property** (or memorylessness):

> *Given the present state, the future is conditionally independent of the past.*

The property asserts that the current state $X_n = i$ encodes all information relevant for predicting $X_{n+1}$. The entire trajectory $(X_0, X_1, \ldots, X_{n-1})$ provides no additional predictive power beyond what $X_n$ already supplies.

### Time-Homogeneity

A Markov chain is **time-homogeneous** if transition probabilities do not depend on the time index:

$$P(X_{n+1} = j \mid X_n = i) = P(X_1 = j \mid X_0 = i) \quad \text{for all } n \geq 0$$

Throughout this chapter, we assume time-homogeneity unless stated otherwise, writing:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

### State Space

The **state space** $S$ is the set of all values the chain can take:

| Type | State Space | Example |
|------|------------|---------|
| Finite | $S = \{0, 1, 2, \ldots, N-1\}$ | Credit ratings, weather states |
| Countably infinite | $S = \mathbb{Z}^+ = \{0, 1, 2, \ldots\}$ | Queue lengths, random walks |
| Continuous | $S = \mathbb{R}^d$ | MCMC on continuous parameter spaces |

Sections 18.1–18.2 focus primarily on finite state spaces, while Section 18.3 (MCMC) extends to continuous spaces.

## Transition Probabilities and Matrices

### One-Step Transition Probabilities

The **one-step transition probability** from state $i$ to state $j$ is:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

These must satisfy two constraints:

1. **Non-negativity**: $P_{ij} \geq 0$ for all $i, j \in S$
2. **Normalization**: $\sum_{j \in S} P_{ij} = 1$ for all $i \in S$

Each row of transition probabilities forms a valid probability distribution over the next state.

### The Transition Matrix

For a finite state space with $N$ states, we arrange transition probabilities into an $N \times N$ **transition matrix** (or stochastic matrix):

$$P = \begin{pmatrix}
P_{00} & P_{01} & \cdots & P_{0,N-1} \\
P_{10} & P_{11} & \cdots & P_{1,N-1} \\
\vdots & \vdots & \ddots & \vdots \\
P_{N-1,0} & P_{N-1,1} & \cdots & P_{N-1,N-1}
\end{pmatrix}$$

The transition matrix is **row-stochastic**: all entries are non-negative and each row sums to 1. The entry at position $(i, j)$ gives the probability of transitioning from state $i$ to state $j$, and the diagonal entry $P_{ii}$ gives the probability of remaining in state $i$.

### Stochastic Matrix Properties

A matrix $P$ is (row) stochastic if and only if $P_{ij} \geq 0$ for all $i,j$ and $P \mathbf{1} = \mathbf{1}$, where $\mathbf{1}$ is the all-ones column vector. Key consequences:

- The product of two stochastic matrices is stochastic (closure under multiplication).
- All eigenvalues of $P$ satisfy $|\lambda| \leq 1$.
- $\lambda_1 = 1$ is always an eigenvalue, with right eigenvector $\mathbf{1}$.

## $n$-Step Transition Probabilities

### Definition

The **$n$-step transition probability** is the probability of reaching state $j$ from state $i$ in exactly $n$ steps:

$$P^{(n)}_{ij} = P(X_{n+m} = j \mid X_m = i)$$

For time-homogeneous chains, this does not depend on $m$.

### Matrix Powers

**Theorem.** The $n$-step transition probabilities are given by the $n$-th power of the transition matrix:

$$P^{(n)}_{ij} = (P^n)_{ij}$$

where $P^n = \underbrace{P \cdot P \cdots P}_{n \text{ times}}$ is ordinary matrix multiplication.

*Proof sketch.* By the law of total probability:

$$P^{(n)}_{ij} = \sum_{k \in S} P^{(n-1)}_{ik} P_{kj}$$

This is exactly the definition of matrix multiplication applied to $P^{(n-1)}$ and $P$. By induction, $P^{(n)} = P^n$. $\square$

This result is profoundly useful: all multi-step dynamics are encoded in matrix powers of $P$.

## Chapman-Kolmogorov Equations

### Statement

For any non-negative integers $m, n$:

$$P^{(m+n)}_{ij} = \sum_{k \in S} P^{(m)}_{ik} P^{(n)}_{kj}$$

In matrix form:

$$P^{m+n} = P^m \cdot P^n$$

### Interpretation

To go from state $i$ to state $j$ in $m + n$ steps, the chain must pass through some intermediate state $k$ at time $m$. The Chapman-Kolmogorov equations decompose the $(m+n)$-step transition as:

$$\underbrace{i \xrightarrow{m \text{ steps}} k}_{\text{probability } P^{(m)}_{ik}} \xrightarrow{n \text{ steps}} \underbrace{j}_{\text{probability } P^{(n)}_{kj}}$$

summed over all possible intermediaries $k$.

## Distribution Evolution

### Initial Distribution

The **initial distribution** $\pi^{(0)}$ specifies the probability of starting in each state:

$$\pi^{(0)}_i = P(X_0 = i)$$

### Propagation

Given initial distribution $\pi^{(0)}$ (as a row vector), the distribution at time $n$ is:

$$\pi^{(n)} = \pi^{(0)} P^n$$

Component-wise:

$$\pi^{(n)}_j = P(X_n = j) = \sum_{i \in S} \pi^{(0)}_i P^{(n)}_{ij}$$

This is the fundamental equation connecting the transition matrix to distributional evolution. For MCMC, the critical question becomes: *does $\pi^{(n)}$ converge to the target distribution $\pi$ as $n \to \infty$, regardless of the starting distribution $\pi^{(0)}$?*

## PyTorch Implementation

### Markov Chain Class

```python
import torch
import torch.linalg as LA
from typing import List, Optional, Union, Dict, Tuple

class MarkovChain:
    """
    Discrete-time Markov chain implementation in PyTorch.

    The Markov property states that:
    P(X_{n+1} = j | X_n = i, X_{n-1}, ..., X_0) = P(X_{n+1} = j | X_n = i)

    Attributes:
        P: Transition probability matrix (row-stochastic)
        n_states: Number of states
        state_names: Optional names for states
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[List[str]] = None,
        validate: bool = True
    ):
        """
        Initialize Markov chain.

        Args:
            transition_matrix: N×N transition probability matrix
                P[i,j] = P(X_{n+1} = j | X_n = i)
            state_names: Optional list of state names
            validate: Whether to validate the transition matrix
        """
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]

        if state_names is None:
            self.state_names = [f"State_{i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names

        if validate:
            self._validate_transition_matrix()

    def _validate_transition_matrix(self):
        """
        Validate that P is a proper stochastic matrix.

        Requirements:
        1. Square matrix
        2. All entries in [0, 1]
        3. Each row sums to 1
        """
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError(
                f"Transition matrix must be square, got shape {self.P.shape}"
            )
        if torch.any(self.P < 0):
            raise ValueError("All transition probabilities must be non-negative")
        if torch.any(self.P > 1):
            raise ValueError("All transition probabilities must be ≤ 1")

        row_sums = self.P.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
            raise ValueError(
                f"Each row must sum to 1. Got row sums: {row_sums.tolist()}"
            )

    def step(self, current_state: int) -> int:
        """
        Perform one step: sample X_{n+1} given X_n = current_state.

        Args:
            current_state: Index of current state

        Returns:
            Index of next state
        """
        probs = self.P[current_state]
        return torch.multinomial(probs, num_samples=1).item()

    def simulate(
        self,
        n_steps: int,
        initial_state: Optional[int] = None,
        initial_distribution: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Simulate the Markov chain for n steps.

        Args:
            n_steps: Number of transitions
            initial_state: Starting state (if specified)
            initial_distribution: Distribution to sample initial state from

        Returns:
            List of states visited (length n_steps + 1)
        """
        if initial_state is not None:
            state = initial_state
        elif initial_distribution is not None:
            state = torch.multinomial(initial_distribution, num_samples=1).item()
        else:
            state = torch.randint(0, self.n_states, (1,)).item()

        trajectory = [state]
        for _ in range(n_steps):
            state = self.step(state)
            trajectory.append(state)

        return trajectory

    def get_transition_probability(
        self,
        from_state: Union[int, str],
        to_state: Union[int, str]
    ) -> float:
        """Get P(from_state → to_state)."""
        if isinstance(from_state, str):
            from_state = self.state_names.index(from_state)
        if isinstance(to_state, str):
            to_state = self.state_names.index(to_state)
        return self.P[from_state, to_state].item()


def create_stochastic_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert any non-negative matrix to a row-stochastic matrix
    by normalizing each row to sum to 1.
    """
    matrix = torch.relu(matrix)
    row_sums = matrix.sum(dim=1, keepdim=True)

    # Handle zero rows: assign uniform distribution
    zero_rows = (row_sums == 0).squeeze()
    if zero_rows.any():
        n = matrix.shape[1]
        matrix[zero_rows] = 1.0 / n
        row_sums[zero_rows] = 1.0

    return matrix / row_sums
```

### Transition Matrix Analyzer

```python
class TransitionMatrixAnalyzer:
    """
    Tools for analyzing transition matrices and computing
    multi-step transition probabilities.
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[List[str]] = None
    ):
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]
        self.state_names = state_names or [
            f"State_{i}" for i in range(self.n_states)
        ]
        self._validate()

    def _validate(self):
        """Validate stochastic matrix properties."""
        assert self.P.shape[0] == self.P.shape[1], "Matrix must be square"
        assert torch.all(self.P >= 0), "All entries must be non-negative"
        row_sums = self.P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "All rows must sum to 1"

    def n_step_matrix(self, n: int) -> torch.Tensor:
        """
        Compute n-step transition matrix P^n.

        P^n[i,j] = P(X_n = j | X_0 = i)
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return torch.eye(self.n_states, dtype=self.P.dtype)
        return torch.linalg.matrix_power(self.P, n)

    def n_step_probability(
        self, from_state: int, to_state: int, n: int
    ) -> float:
        """Compute P^{(n)}_{ij}."""
        P_n = self.n_step_matrix(n)
        return P_n[from_state, to_state].item()

    def distribution_evolution(
        self,
        initial_dist: torch.Tensor,
        n_steps: int
    ) -> torch.Tensor:
        """
        Compute π^{(0)}, π^{(1)}, ..., π^{(n)} where π^{(k)} = π^{(0)} P^k.

        Returns:
            Tensor of shape (n_steps+1, n_states)
        """
        distributions = torch.zeros(n_steps + 1, self.n_states)
        distributions[0] = initial_dist

        current_dist = initial_dist.clone()
        for k in range(1, n_steps + 1):
            current_dist = current_dist @ self.P
            distributions[k] = current_dist

        return distributions

    def chapman_kolmogorov_verify(
        self, m: int, n: int, tol: float = 1e-6
    ) -> bool:
        """Verify Chapman-Kolmogorov: P^{m+n} = P^m × P^n."""
        P_m = self.n_step_matrix(m)
        P_n = self.n_step_matrix(n)
        P_mn = self.n_step_matrix(m + n)
        return torch.allclose(P_mn, P_m @ P_n, atol=tol)
```

### Convergence Analysis

```python
def analyze_convergence(
    P: torch.Tensor,
    max_steps: int = 100,
    tol: float = 1e-8
) -> Dict:
    """
    Analyze convergence of P^n as n → ∞.

    For ergodic chains, all rows of P^n converge to the
    stationary distribution π.
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
        diff = torch.max(torch.abs(P_current - P_prev)).item()
        results['differences'].append(diff)

        if diff < tol:
            results['converged'] = True
            results['convergence_step'] = step
            results['limit_matrix'] = P_current
            results['stationary_distribution'] = P_current[0].clone()
            break

        P_prev = P_current

    return results
```

## Example: Weather Model

A three-state weather model illustrates the core concepts:

```python
# States: Sunny, Cloudy, Rainy
states = ["Sunny", "Cloudy", "Rainy"]

P = torch.tensor([
    [0.70, 0.25, 0.05],  # From Sunny
    [0.30, 0.40, 0.30],  # From Cloudy
    [0.10, 0.40, 0.50]   # From Rainy
])

analyzer = TransitionMatrixAnalyzer(P, state_names=states)

# Multi-step transition probabilities
print("Weather Model: Multi-Step Transition Probabilities")
print("=" * 60)

for n in [1, 2, 5, 10, 50]:
    P_n = analyzer.n_step_matrix(n)
    print(f"\n{n}-Step Transition Matrix P^{n}:")
    print("-" * 40)

    header = "         " + "  ".join(f"{s:>8}" for s in states)
    print(header)

    for i, state_i in enumerate(states):
        row = f"{state_i:8s} " + "  ".join(
            f"{P_n[i,j]:.6f}" for j in range(3)
        )
        print(row)

# Distribution evolution from a deterministic start
pi_0 = torch.tensor([1.0, 0.0, 0.0])  # Start Sunny

print("\nDistribution evolution starting from Sunny:")
for n in [0, 1, 2, 5, 10, 50]:
    P_n = analyzer.n_step_matrix(n)
    pi_n = pi_0 @ P_n
    print(f"  n={n:2d}: π = [{pi_n[0]:.6f}, {pi_n[1]:.6f}, {pi_n[2]:.6f}]")

# Verify Chapman-Kolmogorov
for m, n in [(2, 3), (5, 5), (10, 10)]:
    holds = analyzer.chapman_kolmogorov_verify(m, n)
    print(f"Chapman-Kolmogorov P^{m+n} = P^{m} · P^{n}: {holds}")
```

As $n$ grows, every row of $P^n$ converges to the same vector—this is the **stationary distribution**, developed in the next section.

## Visualization

### Transition Matrix Heatmap

```python
import matplotlib.pyplot as plt

def plot_transition_matrix(
    P: torch.Tensor,
    state_names: List[str] = None,
    title: str = "Transition Matrix"
):
    """Create heatmap visualization of transition matrix."""
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(P.numpy(), cmap='YlOrRd', vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=12)

    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_names, fontsize=11)
    ax.set_yticklabels(state_names, fontsize=11)
    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title(title, fontsize=14)

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
    state_names: List[str] = None,
    title: str = "Distribution Evolution"
):
    """Plot evolution of state distribution over time."""
    n_steps, n_states = distributions.shape
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_states):
        ax.plot(range(n_steps), distributions[:, i].numpy(),
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

## Why This Matters for MCMC

The transition matrix framework established here directly enables MCMC:

| Markov Chain Concept | MCMC Application |
|---------------------|------------------|
| Transition matrix $P$ | The MCMC kernel (proposal + accept/reject) |
| $n$-step distribution $\pi^{(n)}$ | Distribution of the $n$-th MCMC sample |
| Convergence $\pi^{(n)} \to \pi$ | MCMC samples approximate the target distribution |
| Chapman-Kolmogorov | Justifies running chains for multiple steps |

The key questions remaining—*when* does $\pi^{(n)}$ converge, to *what*, and *how fast*—are addressed by the stationary distribution theory and ergodicity results in the following sections.

## Summary

| Concept | Mathematical Form | Description |
|---------|------------------|-------------|
| **Markov Property** | $P(X_{n+1}=j \mid X_n=i, \ldots) = P(X_{n+1}=j \mid X_n=i)$ | Future depends only on present |
| **Transition Matrix** | $P$ where $P_{ij} \geq 0$, $\sum_j P_{ij} = 1$ | Row-stochastic matrix encoding all one-step dynamics |
| **$n$-Step Probability** | $P^{(n)}_{ij} = (P^n)_{ij}$ | Multi-step transitions via matrix powers |
| **Chapman-Kolmogorov** | $P^{m+n} = P^m \cdot P^n$ | Decomposition over intermediate states |
| **Distribution Evolution** | $\pi^{(n)} = \pi^{(0)} P^n$ | How state probabilities evolve over time |

## Exercises

1. **Stochastic Closure.** Prove that if $P$ and $Q$ are both row-stochastic matrices of compatible dimensions, then $PQ$ is also row-stochastic.

2. **Eigenvalue Bound.** Show that all eigenvalues $\lambda$ of a row-stochastic matrix satisfy $|\lambda| \leq 1$. (*Hint*: use the Gershgorin circle theorem.)

3. **Board Game.** Create a Markov chain for a simplified board game where a player on a circular board of 6 positions moves forward 1–3 spaces with equal probability. Compute the 10-step transition matrix and verify that each row approaches a uniform distribution.

4. **Chapman-Kolmogorov Verification.** For the weather model above, manually compute $P^{(3)}_{0,2}$ (probability of Rainy after 3 days starting Sunny) both directly via $P^3$ and via the Chapman-Kolmogorov equation with $m=1, n=2$. Verify they agree.

5. **Financial Application.** Model daily stock price movements as a Markov chain with states $\{$Down, Flat, Up$\}$. Estimate transition probabilities from historical data, and compute the distribution of the stock's state after 20 trading days.

## References

1. Lawler, G.F. *Introduction to Stochastic Processes*, Chapter 1. Chapman & Hall/CRC, 2006.
2. Norris, J.R. *Markov Chains*, Chapter 1. Cambridge University Press, 1997.
3. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, Chapter 3. Springer-Verlag, 1976.
4. Horn, R.A. & Johnson, C.R. *Matrix Analysis*, Chapter 8. Cambridge University Press, 2012.
5. Durrett, R. *Essentials of Stochastic Processes*, Chapter 1. Springer, 2016.
