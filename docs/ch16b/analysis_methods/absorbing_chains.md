# Absorbing Markov Chains

## Introduction

Absorbing Markov chains model systems that eventually "settle" into final states from which they cannot escape. These chains are ubiquitous in applications: the gambler who eventually goes broke or wins, the patient who recovers or dies, the project that succeeds or fails. This section develops the theory for analyzing such chains.

## Definitions

### Absorbing State

A state $i$ is **absorbing** if once entered, the chain cannot leave:

$$P_{ii} = 1$$

Equivalently, $P_{ij} = 0$ for all $j \neq i$.

### Absorbing Markov Chain

A Markov chain is **absorbing** if:
1. It has at least one absorbing state
2. From every state, it is possible to reach some absorbing state

The second condition ensures that absorption eventually occurs with probability 1.

### Transient States

In an absorbing chain, non-absorbing states are called **transient** because the chain will eventually leave them and never return.

## Canonical Form

### Reordering States

For an absorbing chain with $r$ absorbing states and $t$ transient states, we reorder the states to put transient states first:

$$P = \begin{pmatrix}
Q & R \\
\mathbf{0} & I_r
\end{pmatrix}$$

where:
- $Q$: $t \times t$ matrix of transitions among transient states
- $R$: $t \times r$ matrix of transitions from transient to absorbing states
- $\mathbf{0}$: $r \times t$ zero matrix (can't leave absorbing states to transient)
- $I_r$: $r \times r$ identity matrix (stay in absorbing states)

### Properties of Q

The matrix $Q$ has important properties:
- All rows of $Q$ sum to less than 1 (some probability "leaks" to absorbing states)
- All eigenvalues of $Q$ have absolute value less than 1
- $Q^n \to \mathbf{0}$ as $n \to \infty$

## The Fundamental Matrix

### Definition

The **fundamental matrix** $N$ is defined as:

$$N = (I - Q)^{-1} = I + Q + Q^2 + Q^3 + \cdots$$

The infinite series converges because all eigenvalues of $Q$ are less than 1 in absolute value.

### Interpretation

The entry $N_{ij}$ represents the **expected number of visits** to transient state $j$ when starting from transient state $i$, before absorption.

**Derivation**:
Let $n_{ij}$ = expected visits to state $j$ starting from state $i$.

By conditioning on the first step:
$$n_{ij} = \delta_{ij} + \sum_{k \text{ transient}} P_{ik} n_{kj}$$

In matrix form: $N = I + QN$, which gives $N = (I-Q)^{-1}$.

## Key Quantities

### Expected Time to Absorption

The expected number of steps until absorption, starting from transient state $i$:

$$t_i = \sum_{j \text{ transient}} N_{ij}$$

In vector form: $\mathbf{t} = N \mathbf{1}$, where $\mathbf{1}$ is a column of ones.

### Absorption Probabilities

The probability of being absorbed into absorbing state $j$, starting from transient state $i$:

$$B_{ij} = (NR)_{ij}$$

In matrix form: $B = NR$

**Verification**: Each row of $B$ should sum to 1 (absorption is certain).

### Variance of Absorption Time

The variance of the time to absorption from state $i$:

$$\text{Var}(T_i) = [(2N - I) \mathbf{t}]_i - t_i^2$$

where $\mathbf{t}$ is the vector of expected absorption times.

## PyTorch Implementation

```python
import torch
import torch.linalg as LA
from typing import Dict, List, Tuple, Optional

class AbsorbingMarkovChain:
    """
    Analysis tools for absorbing Markov chains.
    
    An absorbing chain has at least one absorbing state (P[i,i] = 1)
    and every state can reach an absorbing state.
    
    Key quantities:
    - Fundamental matrix N = (I - Q)^{-1}
    - Expected absorption time t = N × 1
    - Absorption probabilities B = N × R
    """
    
    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[List[str]] = None
    ):
        """
        Initialize absorbing chain analyzer.
        
        Args:
            transition_matrix: N×N transition matrix
            state_names: Optional state names
        """
        self.P = transition_matrix.clone().double()
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"State_{i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
        
        # Identify absorbing and transient states
        self._classify_states()
        
        # Build canonical form
        self._build_canonical_form()
    
    def _classify_states(self):
        """
        Identify which states are absorbing vs transient.
        
        A state i is absorbing if P[i,i] = 1 (and all other P[i,j] = 0).
        """
        self.absorbing_indices = []
        self.transient_indices = []
        
        for i in range(self.n_states):
            # Check if state i is absorbing
            if torch.isclose(self.P[i, i], torch.tensor(1.0, dtype=self.P.dtype)):
                # Verify all other transitions are zero
                other_probs = self.P[i, :i].sum() + self.P[i, i+1:].sum()
                if torch.isclose(other_probs, torch.tensor(0.0, dtype=self.P.dtype)):
                    self.absorbing_indices.append(i)
                else:
                    self.transient_indices.append(i)
            else:
                self.transient_indices.append(i)
        
        self.n_transient = len(self.transient_indices)
        self.n_absorbing = len(self.absorbing_indices)
        
        if self.n_absorbing == 0:
            raise ValueError("No absorbing states found. Chain is not absorbing.")
        
        # Store state name mappings
        self.transient_names = [self.state_names[i] for i in self.transient_indices]
        self.absorbing_names = [self.state_names[i] for i in self.absorbing_indices]
    
    def _build_canonical_form(self):
        """
        Reorder states into canonical form: transient first, absorbing last.
        
        Canonical form:
        P = [[Q, R],
             [0, I]]
        """
        # New ordering: transient states first, then absorbing
        reordered = self.transient_indices + self.absorbing_indices
        
        # Reorder transition matrix
        P_canonical = self.P[torch.tensor(reordered)][:, torch.tensor(reordered)]
        
        # Extract Q (transient → transient) and R (transient → absorbing)
        t = self.n_transient
        self.Q = P_canonical[:t, :t]
        self.R = P_canonical[:t, t:]
    
    def fundamental_matrix(self) -> torch.Tensor:
        """
        Compute the fundamental matrix N = (I - Q)^{-1}.
        
        N[i,j] = expected number of times in transient state j,
                 starting from transient state i, before absorption.
        
        Returns:
            Fundamental matrix N (t × t)
        """
        I = torch.eye(self.n_transient, dtype=self.Q.dtype)
        self.N = LA.inv(I - self.Q)
        return self.N
    
    def expected_absorption_time(self) -> Dict[str, float]:
        """
        Compute expected steps until absorption from each transient state.
        
        Formula: t = N × 1
        
        Returns:
            Dictionary: state_name → expected time
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        # t = N × 1 (column vector of ones)
        ones = torch.ones(self.n_transient, 1, dtype=self.N.dtype)
        t = self.N @ ones
        
        return {name: t[i, 0].item() for i, name in enumerate(self.transient_names)}
    
    def absorption_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Compute probability of absorption into each absorbing state.
        
        Formula: B = N × R
        B[i,j] = P(absorb into state j | start in state i)
        
        Returns:
            Nested dictionary: start_state → {absorbing_state → probability}
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        self.B = self.N @ self.R
        
        result = {}
        for i, trans_name in enumerate(self.transient_names):
            result[trans_name] = {}
            for j, abs_name in enumerate(self.absorbing_names):
                result[trans_name][abs_name] = self.B[i, j].item()
        
        return result
    
    def variance_absorption_time(self) -> Dict[str, float]:
        """
        Compute variance of time to absorption.
        
        Formula: Var[T_i] = [(2N - I) × t]_i - t_i^2
        
        Returns:
            Dictionary: state_name → variance
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        I = torch.eye(self.n_transient, dtype=self.N.dtype)
        ones = torch.ones(self.n_transient, 1, dtype=self.N.dtype)
        
        t = self.N @ ones  # Expected times
        
        # Variance formula
        var_vec = (2 * self.N - I) @ t - t ** 2
        
        return {name: var_vec[i, 0].item() for i, name in enumerate(self.transient_names)}
    
    def expected_visits(self) -> Dict[str, Dict[str, float]]:
        """
        Get expected number of visits to each transient state.
        
        Returns:
            Nested dictionary: start_state → {visit_state → expected_visits}
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        result = {}
        for i, start_name in enumerate(self.transient_names):
            result[start_name] = {}
            for j, visit_name in enumerate(self.transient_names):
                result[start_name][visit_name] = self.N[i, j].item()
        
        return result
    
    def simulate_until_absorption(
        self,
        initial_state: int,
        max_steps: int = 10000
    ) -> Tuple[List[int], int, int]:
        """
        Simulate the chain until absorption occurs.
        
        Args:
            initial_state: Starting state index (in original numbering)
            max_steps: Maximum simulation steps
            
        Returns:
            (path, final_absorbing_state, num_steps)
        """
        path = [initial_state]
        current = initial_state
        
        for step in range(max_steps):
            # Check if absorbed
            if current in self.absorbing_indices:
                return path, current, step
            
            # Transition
            probs = self.P[current].float()
            current = torch.multinomial(probs, num_samples=1).item()
            path.append(current)
        
        return path, current, max_steps


def demonstrate_absorbing_chain():
    """
    Demonstrate analysis of an absorbing Markov chain: Gambler's Ruin.
    """
    print("Absorbing Markov Chain: Gambler's Ruin")
    print("=" * 70)
    
    # Gambler starts with $k, bets $1 each round
    # Win (p=0.5): gain $1, Lose (p=0.5): lose $1
    # Game ends at $0 (broke) or $N (win)
    
    N_target = 4  # Target amount
    
    # States: $0, $1, $2, $3, $4
    # Absorbing: $0 (broke), $4 (win)
    # Transient: $1, $2, $3
    
    states = ['$0 (Broke)', '$1', '$2', '$3', '$4 (Win)']
    
    P = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # $0: absorbing (broke)
        [0.5, 0.0, 0.5, 0.0, 0.0],  # $1: lose → $0, win → $2
        [0.0, 0.5, 0.0, 0.5, 0.0],  # $2: lose → $1, win → $3
        [0.0, 0.0, 0.5, 0.0, 0.5],  # $3: lose → $2, win → $4
        [0.0, 0.0, 0.0, 0.0, 1.0]   # $4: absorbing (win)
    ])
    
    print("\nTransition Matrix:")
    print(P)
    
    chain = AbsorbingMarkovChain(P, state_names=states)
    
    print(f"\nTransient states: {chain.transient_names}")
    print(f"Absorbing states: {chain.absorbing_names}")
    
    # Fundamental matrix
    print("\n" + "-" * 70)
    print("Fundamental Matrix N (expected visits to transient states):")
    N = chain.fundamental_matrix()
    
    header = f"{'':10s}" + "".join(f"{s:>10s}" for s in chain.transient_names)
    print(header)
    for i, name in enumerate(chain.transient_names):
        row = f"{name:10s}" + "".join(f"{N[i,j]:10.4f}" for j in range(chain.n_transient))
        print(row)
    
    # Expected absorption times
    print("\n" + "-" * 70)
    print("Expected Steps to Absorption:")
    times = chain.expected_absorption_time()
    for state, time in times.items():
        print(f"  From {state}: {time:.4f} steps")
    
    # Absorption probabilities
    print("\n" + "-" * 70)
    print("Absorption Probabilities:")
    probs = chain.absorption_probabilities()
    for trans_state in chain.transient_names:
        print(f"\n  Starting from {trans_state}:")
        for abs_state in chain.absorbing_names:
            prob = probs[trans_state][abs_state]
            print(f"    P(absorb at {abs_state}) = {prob:.6f}")
    
    # Variance
    print("\n" + "-" * 70)
    print("Variance of Absorption Time:")
    variances = chain.variance_absorption_time()
    for state, var in variances.items():
        std = var ** 0.5 if var > 0 else 0
        print(f"  From {state}: Var = {var:.4f}, Std = {std:.4f}")
    
    # Simulation verification
    print("\n" + "-" * 70)
    print("Simulation Verification (10,000 trials from $2):")
    
    results = {'$0 (Broke)': 0, '$4 (Win)': 0}
    total_steps = []
    
    for _ in range(10000):
        path, final, steps = chain.simulate_until_absorption(initial_state=2)
        results[states[final]] += 1
        total_steps.append(steps)
    
    print(f"  Empirical P(Broke) = {results['$0 (Broke)']/10000:.4f} "
          f"(Theory: {probs['$2']['$0 (Broke)']:.4f})")
    print(f"  Empirical P(Win)   = {results['$4 (Win)']/10000:.4f} "
          f"(Theory: {probs['$2']['$4 (Win)']:.4f})")
    print(f"  Empirical E[T]     = {sum(total_steps)/len(total_steps):.4f} "
          f"(Theory: {times['$2']:.4f})")


# Run demonstration
demonstrate_absorbing_chain()
```

## Example: Disease Progression Model

```python
def disease_progression_example():
    """
    Model disease progression with absorbing states.
    
    States: Healthy, Infected, Recovered (absorbing), Deceased (absorbing)
    """
    print("\n" + "=" * 70)
    print("Disease Progression Model")
    print("=" * 70)
    
    states = ['Healthy', 'Infected', 'Recovered', 'Deceased']
    
    P = torch.tensor([
        [0.70, 0.30, 0.00, 0.00],  # Healthy: may get infected
        [0.00, 0.40, 0.50, 0.10],  # Infected: may recover or die
        [0.00, 0.00, 1.00, 0.00],  # Recovered: absorbing
        [0.00, 0.00, 0.00, 1.00]   # Deceased: absorbing
    ])
    
    print("\nTransition Matrix:")
    header = f"{'':12s}" + "".join(f"{s:>12s}" for s in states)
    print(header)
    for i, state in enumerate(states):
        row = f"{state:12s}" + "".join(f"{P[i,j]:12.4f}" for j in range(4))
        print(row)
    
    chain = AbsorbingMarkovChain(P, state_names=states)
    
    # Key results
    times = chain.expected_absorption_time()
    probs = chain.absorption_probabilities()
    
    print("\n" + "-" * 70)
    print("Key Results:")
    
    print("\nExpected time until recovery or death:")
    for state, time in times.items():
        print(f"  From {state}: {time:.2f} time periods")
    
    print("\nFinal outcome probabilities:")
    for trans_state in chain.transient_names:
        print(f"\n  If currently {trans_state}:")
        for abs_state in chain.absorbing_names:
            prob = probs[trans_state][abs_state]
            print(f"    {abs_state}: {prob:.4f} ({prob*100:.1f}%)")


disease_progression_example()
```

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_absorbing_chain(chain: AbsorbingMarkovChain, n_simulations: int = 1000):
    """
    Visualize properties of an absorbing Markov chain.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get starting transient state index (in original numbering)
    start_idx = chain.transient_indices[len(chain.transient_indices) // 2]
    
    # Plot 1: Sample paths
    ax = axes[0, 0]
    for _ in range(20):
        path, final, steps = chain.simulate_until_absorption(start_idx, max_steps=200)
        ax.plot(path, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('State', fontsize=11)
    ax.set_yticks(range(chain.n_states))
    ax.set_yticklabels(chain.state_names, fontsize=9)
    ax.set_title('Sample Absorption Paths', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Highlight absorbing states
    for idx in chain.absorbing_indices:
        ax.axhline(y=idx, color='red', linestyle='--', alpha=0.3)
    
    # Plot 2: Absorption time distribution
    ax = axes[0, 1]
    
    absorption_times = []
    for _ in range(n_simulations):
        _, _, steps = chain.simulate_until_absorption(start_idx, max_steps=1000)
        absorption_times.append(steps)
    
    ax.hist(absorption_times, bins=50, density=True, alpha=0.7, edgecolor='black')
    
    # Add theoretical mean
    times = chain.expected_absorption_time()
    trans_name = chain.state_names[start_idx]
    if trans_name in times:
        mean_time = times[trans_name]
        ax.axvline(x=mean_time, color='red', linestyle='--', linewidth=2,
                  label=f'E[T] = {mean_time:.2f}')
    
    emp_mean = np.mean(absorption_times)
    ax.axvline(x=emp_mean, color='blue', linestyle='--', linewidth=2,
              label=f'Empirical = {emp_mean:.2f}')
    
    ax.set_xlabel('Steps to Absorption', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of Absorption Time', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Absorption probabilities
    ax = axes[1, 0]
    
    # Count outcomes
    outcomes = {name: 0 for name in chain.absorbing_names}
    for _ in range(n_simulations):
        _, final, _ = chain.simulate_until_absorption(start_idx)
        outcomes[chain.state_names[final]] += 1
    
    probs = chain.absorption_probabilities()
    
    x = np.arange(len(chain.absorbing_names))
    width = 0.35
    
    empirical = [outcomes[name] / n_simulations for name in chain.absorbing_names]
    theoretical = [probs[trans_name][name] for name in chain.absorbing_names]
    
    ax.bar(x - width/2, empirical, width, label='Empirical', alpha=0.7)
    ax.bar(x + width/2, theoretical, width, label='Theoretical', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(chain.absorbing_names)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title(f'Absorption Probabilities (from {trans_name})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Fundamental matrix heatmap
    ax = axes[1, 1]
    
    N = chain.fundamental_matrix()
    im = ax.imshow(N.numpy(), cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(chain.n_transient))
    ax.set_yticks(range(chain.n_transient))
    ax.set_xticklabels(chain.transient_names, fontsize=9)
    ax.set_yticklabels(chain.transient_names, fontsize=9)
    ax.set_xlabel('To State', fontsize=11)
    ax.set_ylabel('From State', fontsize=11)
    ax.set_title('Fundamental Matrix N (Expected Visits)', fontsize=12)
    
    # Add text annotations
    for i in range(chain.n_transient):
        for j in range(chain.n_transient):
            ax.text(j, i, f'{N[i,j]:.2f}', ha='center', va='center',
                   fontsize=10, color='black')
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig
```

## Summary

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| **Fundamental Matrix** | $N = (I - Q)^{-1}$ | $N_{ij}$ = expected visits to $j$ from $i$ |
| **Expected Absorption Time** | $\mathbf{t} = N\mathbf{1}$ | Steps until absorption |
| **Absorption Probabilities** | $B = NR$ | $B_{ij}$ = P(absorb into $j$ from $i$) |
| **Variance of Time** | $(2N-I)\mathbf{t} - \mathbf{t}^2$ | Variance of absorption time |

## Exercises

1. **Drunkard's Walk**: A drunkard starts at position 2 on a line from 0 to 4. At each step, they move left or right with equal probability. Compute the probability of reaching 0 vs 4.

2. **Multiple Absorbing States**: Modify the disease model to include "Chronic" as a third absorbing state.

3. **Sensitivity Analysis**: How do the absorption probabilities change as the recovery probability varies?

4. **Expected Visits**: In the gambler's ruin, starting with $2, compute the expected number of times the gambler will have exactly $1.

## References

1. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, Chapter 3. Springer-Verlag, 1976.
2. Grinstead, C.M. & Snell, J.L. *Introduction to Probability*, Chapter 11. AMS, 1997.
3. Norris, J.R. *Markov Chains*, Chapter 1. Cambridge University Press, 1997.
