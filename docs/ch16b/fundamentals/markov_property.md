# Markov Property and Basic Definitions

## Introduction

The Markov property is the cornerstone of Markov chain theory. It captures a profound simplification: the future evolution of a system depends only on its current state, not on how it arrived there. This "memorylessness" enables tractable mathematical analysis while still capturing essential dynamics of many real-world systems.

## The Markov Property

### Formal Definition

A discrete-time stochastic process $\{X_n\}_{n \geq 0}$ with state space $S$ is a **Markov chain** if for all $n \geq 0$ and all states $i_0, i_1, \ldots, i_{n-1}, i, j \in S$:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

### Intuitive Interpretation

The Markov property states that:

> *"Given the present, the future is conditionally independent of the past."*

This means:
- History contains no additional information for predicting the future beyond what the current state provides
- All relevant information about the system is encoded in the current state
- The system has "no memory" of how it reached its current state

### Time-Homogeneity

A Markov chain is **time-homogeneous** (or stationary) if the transition probabilities do not depend on time:

$$P(X_{n+1} = j \mid X_n = i) = P(X_1 = j \mid X_0 = i) \quad \text{for all } n \geq 0$$

Throughout this chapter, we assume time-homogeneity unless otherwise stated. This allows us to write:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

## State Space

### Definition

The **state space** $S$ is the set of all possible values the chain can take. Common examples include:

| Type | State Space | Example |
|------|------------|---------|
| Finite | $S = \{0, 1, 2, \ldots, N\}$ | Weather states, credit ratings |
| Countably Infinite | $S = \mathbb{Z}^+ = \{0, 1, 2, \ldots\}$ | Queue lengths, population counts |
| Continuous | $S = \mathbb{R}$ or $\mathbb{R}^d$ | Position in space (for Markov processes) |

This chapter focuses primarily on **finite state spaces**, though the concepts extend naturally.

### State Indexing

For a finite state space with $N$ states, we typically index states as $S = \{0, 1, 2, \ldots, N-1\}$ or use meaningful labels like $S = \{\text{Sunny}, \text{Cloudy}, \text{Rainy}\}$.

## Transition Probabilities

### One-Step Transitions

The **one-step transition probability** from state $i$ to state $j$ is:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

These probabilities must satisfy:

1. **Non-negativity**: $P_{ij} \geq 0$ for all $i, j \in S$
2. **Normalization**: $\sum_{j \in S} P_{ij} = 1$ for all $i \in S$

### Transition Matrix

For a finite state space with $N$ states, we arrange the transition probabilities into an $N \times N$ **transition matrix** (or stochastic matrix) $P$:

$$P = \begin{pmatrix}
P_{00} & P_{01} & \cdots & P_{0,N-1} \\
P_{10} & P_{11} & \cdots & P_{1,N-1} \\
\vdots & \vdots & \ddots & \vdots \\
P_{N-1,0} & P_{N-1,1} & \cdots & P_{N-1,N-1}
\end{pmatrix}$$

**Properties of the Transition Matrix:**

- Each row represents a probability distribution over next states
- Each row sums to 1 (row-stochastic)
- All entries are non-negative
- Entry $(i, j)$ gives the probability of transitioning from state $i$ to state $j$

## PyTorch Implementation

### Basic Markov Chain Class

```python
import torch
import torch.nn.functional as F
from typing import List, Optional, Union

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
        # Check square
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError(
                f"Transition matrix must be square, got shape {self.P.shape}"
            )
        
        # Check non-negativity
        if torch.any(self.P < 0):
            raise ValueError("All transition probabilities must be non-negative")
        
        # Check entries ≤ 1
        if torch.any(self.P > 1):
            raise ValueError("All transition probabilities must be ≤ 1")
        
        # Check row sums
        row_sums = self.P.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
            raise ValueError(
                f"Each row must sum to 1. Got row sums: {row_sums.tolist()}"
            )
    
    def step(self, current_state: int) -> int:
        """
        Perform one step of the Markov chain.
        
        Given current state i, sample next state j with probability P[i,j].
        This implements the Markov property: next state depends only on
        current state.
        
        Args:
            current_state: Index of current state
            
        Returns:
            Index of next state
        """
        # Get transition probabilities from current state
        probs = self.P[current_state]
        
        # Sample from categorical distribution
        next_state = torch.multinomial(probs, num_samples=1).item()
        
        return next_state
    
    def simulate(
        self,
        n_steps: int,
        initial_state: Optional[int] = None,
        initial_distribution: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Simulate the Markov chain for n steps.
        
        Args:
            n_steps: Number of transitions to simulate
            initial_state: Starting state (if specified)
            initial_distribution: Distribution to sample initial state from
            
        Returns:
            List of states visited (length n_steps + 1)
        """
        # Determine initial state
        if initial_state is not None:
            state = initial_state
        elif initial_distribution is not None:
            state = torch.multinomial(initial_distribution, num_samples=1).item()
        else:
            # Uniform distribution over states
            state = torch.randint(0, self.n_states, (1,)).item()
        
        # Record trajectory
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
        """
        Get transition probability P(from_state → to_state).
        
        Args:
            from_state: Source state (index or name)
            to_state: Target state (index or name)
            
        Returns:
            Transition probability
        """
        # Convert names to indices if needed
        if isinstance(from_state, str):
            from_state = self.state_names.index(from_state)
        if isinstance(to_state, str):
            to_state = self.state_names.index(to_state)
        
        return self.P[from_state, to_state].item()


def create_stochastic_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert any non-negative matrix to a row-stochastic matrix.
    
    Each row is normalized to sum to 1, creating valid transition
    probabilities.
    
    Args:
        matrix: Non-negative matrix
        
    Returns:
        Row-stochastic matrix
    """
    # Ensure non-negative
    matrix = torch.relu(matrix)
    
    # Normalize each row
    row_sums = matrix.sum(dim=1, keepdim=True)
    
    # Handle zero rows (assign uniform distribution)
    zero_rows = (row_sums == 0).squeeze()
    if zero_rows.any():
        n = matrix.shape[1]
        matrix[zero_rows] = 1.0 / n
        row_sums[zero_rows] = 1.0
    
    return matrix / row_sums
```

### Example: Two-State Chain

```python
# Example: Simple two-state Markov chain
# States: {0: "Off", 1: "On"}
# Transitions:
#   - If Off: 70% stay Off, 30% turn On
#   - If On: 40% turn Off, 60% stay On

P = torch.tensor([
    [0.7, 0.3],  # From Off
    [0.4, 0.6]   # From On
])

mc = MarkovChain(
    transition_matrix=P,
    state_names=["Off", "On"]
)

# Simulate 20 steps starting from "Off"
trajectory = mc.simulate(n_steps=20, initial_state=0)
state_names = [mc.state_names[s] for s in trajectory]

print(f"Trajectory: {state_names}")
print(f"P(Off → On) = {mc.get_transition_probability('Off', 'On'):.2f}")
```

### Example: Three-State Weather Model

```python
# Weather model with states: Sunny, Cloudy, Rainy
states = ["Sunny", "Cloudy", "Rainy"]

P = torch.tensor([
    [0.70, 0.25, 0.05],  # From Sunny
    [0.30, 0.40, 0.30],  # From Cloudy
    [0.10, 0.40, 0.50]   # From Rainy
])

weather_chain = MarkovChain(P, state_names=states)

# Simulate 30 days starting from Sunny
trajectory = weather_chain.simulate(n_steps=30, initial_state=0)
weather_sequence = [states[s] for s in trajectory]

# Count frequencies
from collections import Counter
freq = Counter(weather_sequence)
print(f"Weather frequencies over 31 days:")
for state in states:
    print(f"  {state}: {freq[state]}/31 = {freq[state]/31:.3f}")
```

## Distribution Evolution

### Initial Distribution

The **initial distribution** $\pi^{(0)}$ specifies the probability of starting in each state:

$$\pi^{(0)}_i = P(X_0 = i)$$

### Distribution at Time n

Given initial distribution $\pi^{(0)}$, the distribution at time $n$ is:

$$\pi^{(n)} = \pi^{(0)} P^n$$

where $P^n$ is the $n$-th power of the transition matrix.

Component-wise:

$$\pi^{(n)}_j = P(X_n = j) = \sum_{i \in S} \pi^{(0)}_i P^{(n)}_{ij}$$

### PyTorch Implementation

```python
def compute_distribution_at_time_n(
    P: torch.Tensor,
    initial_dist: torch.Tensor,
    n: int
) -> torch.Tensor:
    """
    Compute state distribution at time n.
    
    Given initial distribution π⁽⁰⁾ and transition matrix P,
    computes π⁽ⁿ⁾ = π⁽⁰⁾ P^n
    
    Args:
        P: Transition matrix
        initial_dist: Initial distribution (row vector)
        n: Number of time steps
        
    Returns:
        Distribution at time n
    """
    # Compute P^n
    P_n = torch.linalg.matrix_power(P, n)
    
    # Multiply: π⁽ⁿ⁾ = π⁽⁰⁾ × P^n
    return initial_dist @ P_n


# Example: Distribution evolution
P = torch.tensor([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Start certainly in state 0
pi_0 = torch.tensor([1.0, 0.0])

print("Distribution evolution:")
for n in [0, 1, 2, 5, 10, 50]:
    pi_n = compute_distribution_at_time_n(P, pi_0, n)
    print(f"  n={n:2d}: π = [{pi_n[0]:.6f}, {pi_n[1]:.6f}]")
```

## Summary

| Concept | Mathematical Form | Description |
|---------|------------------|-------------|
| **Markov Property** | $P(X_{n+1}=j \mid X_n=i, \ldots) = P(X_{n+1}=j \mid X_n=i)$ | Future depends only on present |
| **Transition Probability** | $P_{ij} = P(X_{n+1}=j \mid X_n=i)$ | One-step transition from $i$ to $j$ |
| **Transition Matrix** | $P$ where $\sum_j P_{ij} = 1$ | Row-stochastic matrix of all transitions |
| **Time-Homogeneity** | $P_{ij}^{(n)} = P_{ij}$ for all $n$ | Transition probabilities don't change over time |
| **Distribution Evolution** | $\pi^{(n)} = \pi^{(0)} P^n$ | How state probabilities evolve |

## Exercises

1. **Validation**: Prove that if $P$ is a valid transition matrix, then $P^2$ is also a valid transition matrix.

2. **Implementation**: Create a Markov chain for a simple board game where a player moves forward 1-3 spaces with equal probability.

3. **Simulation**: Generate 1000 simulations of the weather model and compare empirical frequencies to the theoretical stationary distribution (computed in later sections).

4. **Theory**: Show that the Markov property implies the Chapman-Kolmogorov equations (covered in next section).

## References

1. Lawler, G.F. *Introduction to Stochastic Processes*, Chapter 1. Chapman & Hall/CRC, 2006.
2. Durrett, R. *Essentials of Stochastic Processes*, Chapter 1. Springer, 2016.
3. Ross, S.M. *Introduction to Probability Models*, Chapter 4. Academic Press, 2014.
