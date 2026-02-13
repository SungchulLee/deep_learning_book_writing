# Stationary Distribution

## Introduction

The stationary distribution is the central object connecting Markov chain theory to MCMC sampling. It represents the long-run equilibrium of a chain—the distribution to which the chain converges regardless of its starting state. In the MCMC framework, the entire point is to construct a chain whose stationary distribution equals a given target $\pi$, so that running the chain long enough yields approximate samples from $\pi$.

## Definition and Properties

### Mathematical Definition

A probability distribution $\pi = (\pi_0, \pi_1, \ldots, \pi_{N-1})$ is a **stationary distribution** (or invariant distribution, equilibrium distribution) for a Markov chain with transition matrix $P$ if:

$$\pi = \pi P$$

Component-wise:

$$\pi_j = \sum_{i \in S} \pi_i P_{ij} \quad \text{for all } j \in S$$

### Why "Stationary"

If the chain starts with distribution $\pi$—that is, $P(X_0 = i) = \pi_i$—then at every subsequent time:

$$P(X_n = j) = \sum_{i \in S} \pi_i P^{(n)}_{ij} = (\pi P^n)_j = \pi_j$$

The distribution is preserved under the dynamics: once in equilibrium, the chain stays in equilibrium. The fixed-point equation $\pi = \pi P$ encodes this invariance.

### Physical Interpretations

For ergodic chains, $\pi_i$ admits three equivalent interpretations:

1. **Long-run proportion**: the fraction of time spent in state $i$ as $n \to \infty$
2. **Limiting probability**: $\lim_{n \to \infty} P(X_n = i)$ regardless of starting state
3. **Reciprocal of mean return time**: $\pi_i = 1/\mathbb{E}[T_i]$ where $T_i = \min\{n \geq 1 : X_n = i \mid X_0 = i\}$

The third interpretation is especially useful for MCMC: states with higher stationary probability are visited more frequently (shorter expected return time), which is exactly what we want when sampling from $\pi$.

## Existence and Uniqueness

### Existence

**Theorem.** Every finite Markov chain has at least one stationary distribution.

*Proof sketch.* The set of probability distributions on $S$ is compact (closed and bounded in $\mathbb{R}^N$), and the mapping $\mu \mapsto \mu P$ is continuous. By the Brouwer fixed-point theorem, a fixed point exists.

### Uniqueness

**Theorem.** An **irreducible** Markov chain has exactly one stationary distribution.

Irreducibility (all states communicate) is sufficient for uniqueness but not for convergence. For convergence we additionally need aperiodicity—this is the content of the ergodicity theorems in the next section.

### Summary of Conditions

| Condition | Unique $\pi$? | Convergence $P^n \to \mathbf{1}\pi$? |
|-----------|:---:|:---:|
| Irreducible only | Yes | Not guaranteed (may oscillate) |
| Aperiodic only | May have multiple | Depends |
| **Ergodic** (irreducible + aperiodic) | **Yes** | **Yes** |
| Reducible | Multiple possible | No (depends on start) |

## Four Methods for Computing $\pi$

### Method 1: Eigenvector Method

Since $\pi P = \pi$, transposing gives $P^T \pi^T = \pi^T$. So $\pi^T$ is a right eigenvector of $P^T$ with eigenvalue $\lambda = 1$.

**Algorithm:**

1. Compute eigenvalues and eigenvectors of $P^T$
2. Find the eigenvector corresponding to eigenvalue 1
3. Normalize to sum to 1

### Method 2: Linear System

The equation $\pi(P - I) = \mathbf{0}$ defines a homogeneous linear system. We replace one equation with the normalization constraint $\sum_i \pi_i = 1$ to obtain a unique solution.

**Algorithm:**

1. Form the matrix $A = P^T - I$
2. Replace the last row of $A$ with $[1, 1, \ldots, 1]$
3. Set right-hand side $b = [0, 0, \ldots, 0, 1]^T$
4. Solve $A \pi^T = b$

### Method 3: Power Iteration

For ergodic chains, repeatedly multiplying any initial distribution by $P$ converges to $\pi$.

**Algorithm:**

1. Start with any distribution $\pi^{(0)}$
2. Iterate: $\pi^{(k+1)} = \pi^{(k)} P$
3. Stop when $\|\pi^{(k+1)} - \pi^{(k)}\| < \epsilon$

This is conceptually the simplest method and directly mirrors what MCMC does: run the chain and wait for convergence.

### Method 4: Simulation (Ergodic Theorem)

For ergodic chains, the time-average of indicator functions converges to the space-average.

**Algorithm:**

1. Simulate the chain for $T$ steps (with burn-in)
2. Count visits to each state: $N_i = \sum_{t=0}^{T} \mathbf{1}\{X_t = i\}$
3. Estimate: $\hat{\pi}_i = N_i / T$

This is essentially what MCMC does in practice—the method foreshadows the ergodic theorem that justifies all MCMC estimators.

## PyTorch Implementation

```python
import torch
import torch.linalg as LA
from typing import Dict, Tuple, Optional

class StationaryDistributionAnalyzer:
    """
    Compute stationary distributions using multiple methods.

    A stationary distribution π satisfies: π = πP
    Equivalently, π^T is a right eigenvector of P^T with eigenvalue 1.
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[list] = None
    ):
        self.P = transition_matrix.clone().double()
        self.n_states = self.P.shape[0]
        self.state_names = state_names or [
            f"State_{i}" for i in range(self.n_states)
        ]

    def via_eigenvector(self) -> torch.Tensor:
        """
        Method 1: Eigenvector of P^T with eigenvalue 1.

        πP = π  ⟹  P^T π^T = π^T
        """
        eigenvalues, eigenvectors = LA.eig(self.P.T)
        idx = torch.argmin(torch.abs(eigenvalues.real - 1.0))
        pi = eigenvectors[:, idx].real
        pi = torch.abs(pi)
        pi = pi / pi.sum()
        return pi.float()

    def via_linear_system(self) -> torch.Tensor:
        """
        Method 2: Solve (P^T - I)π^T = 0 with normalization.

        Replace last equation with Σπ_i = 1.
        """
        A = self.P.T - torch.eye(self.n_states, dtype=self.P.dtype)
        A[-1, :] = torch.ones(self.n_states, dtype=self.P.dtype)
        b = torch.zeros(self.n_states, dtype=self.P.dtype)
        b[-1] = 1.0
        pi = LA.solve(A, b)
        return pi.float()

    def via_power_iteration(
        self,
        max_iter: int = 1000,
        tol: float = 1e-10
    ) -> Tuple[torch.Tensor, int]:
        """
        Method 3: Iterate π^{(k+1)} = π^{(k)} P until convergence.
        """
        P_n = self.P.clone()
        for n in range(1, max_iter + 1):
            P_next = P_n @ self.P
            max_diff = torch.max(torch.abs(P_next - P_n))
            if max_diff < tol:
                return P_next[0].float(), n
            P_n = P_next
        return P_n[0].float(), max_iter

    def via_simulation(
        self,
        n_steps: int = 100000,
        initial_state: int = 0,
        burn_in: int = 1000
    ) -> torch.Tensor:
        """
        Method 4: Estimate π via long-run simulation (ergodic theorem).

        lim_{T→∞} (1/T) Σ_{t=0}^{T} 1{X_t = j} = π_j
        """
        state_counts = torch.zeros(self.n_states)
        current_state = initial_state

        for step in range(n_steps + burn_in):
            if step >= burn_in:
                state_counts[current_state] += 1
            probs = self.P[current_state].float()
            current_state = torch.multinomial(probs, num_samples=1).item()

        return state_counts / state_counts.sum()

    def compare_all_methods(
        self,
        n_simulation_steps: int = 100000
    ) -> Dict[str, torch.Tensor]:
        """Compute π using all four methods for comparison."""
        results = {}
        results['eigenvector'] = self.via_eigenvector()
        results['linear_system'] = self.via_linear_system()
        pi_power, iterations = self.via_power_iteration()
        results['power_iteration'] = pi_power
        results['power_iterations_count'] = iterations
        results['simulation'] = self.via_simulation(n_steps=n_simulation_steps)
        return results

    def verify_stationary(
        self,
        pi: torch.Tensor,
        tol: float = 1e-6
    ) -> Dict:
        """
        Verify that π is a valid stationary distribution.

        Checks: (1) valid probability distribution, (2) πP = π.
        """
        results = {}
        results['is_non_negative'] = torch.all(pi >= -tol).item()
        results['sums_to_one'] = torch.abs(pi.sum() - 1.0).item() < tol
        pi_P = pi.double() @ self.P
        fixed_point_error = torch.max(torch.abs(pi_P - pi.double())).item()
        results['fixed_point_error'] = fixed_point_error
        results['is_stationary'] = fixed_point_error < tol
        return results
```

### Demonstration: Method Comparison

```python
def demonstrate_stationary_distribution():
    """Compare all four methods on a three-state weather model."""
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = torch.tensor([
        [0.7, 0.25, 0.05],
        [0.3, 0.40, 0.30],
        [0.1, 0.40, 0.50]
    ])

    analyzer = StationaryDistributionAnalyzer(P, state_names=states)
    results = analyzer.compare_all_methods(n_simulation_steps=100000)

    print("Stationary Distribution: Method Comparison")
    print("=" * 70)

    header = f"{'State':<12}" + "".join(
        f"{m:<15}" for m in ['Eigenvector', 'Linear Sys',
                              'Power Iter', 'Simulation']
    )
    print(header)

    for i, state in enumerate(states):
        row = f"{state:<12}"
        row += f"{results['eigenvector'][i]:<15.8f}"
        row += f"{results['linear_system'][i]:<15.8f}"
        row += f"{results['power_iteration'][i]:<15.8f}"
        row += f"{results['simulation'][i]:<15.8f}"
        print(row)

    print(f"\nPower iteration converged in "
          f"{results['power_iterations_count']} iterations")

    # Verify
    verification = analyzer.verify_stationary(results['eigenvector'])
    print(f"\nVerification: ||πP - π|| = "
          f"{verification['fixed_point_error']:.2e}")

    # Interpretation
    pi = results['eigenvector']
    print("\nLong-run interpretation:")
    for i, state in enumerate(states):
        pct = pi[i].item() * 100
        mean_return = 1.0 / pi[i].item()
        print(f"  {state}: {pct:.2f}% of the time "
              f"(mean return time: {mean_return:.2f} days)")


demonstrate_stationary_distribution()
```

## Convergence to Stationary Distribution

### Rate of Convergence

For ergodic chains, the convergence rate is controlled by the **spectral gap**:

$$\gamma = 1 - |\lambda_2|$$

where $\lambda_2$ is the second-largest eigenvalue of $P$ in absolute value. The total variation distance between the distribution at time $n$ and the stationary distribution decays exponentially:

$$\|P^n_{i,\cdot} - \pi\|_{TV} \leq C \cdot |\lambda_2|^n$$

A larger spectral gap means faster convergence—the chain "forgets" its starting state more quickly.

### Visualization

```python
import matplotlib.pyplot as plt

def visualize_convergence_to_stationary(
    P: torch.Tensor,
    state_names: list = None,
    max_steps: int = 50
):
    """Visualize how distributions converge to stationary."""
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    analyzer = StationaryDistributionAnalyzer(P, state_names)
    pi_stationary = analyzer.via_eigenvector()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: TV distance to stationary (log scale)
    ax1 = axes[0]
    for start_state in range(n_states):
        dist = torch.zeros(n_states)
        dist[start_state] = 1.0

        distances = []
        for step in range(max_steps):
            tv_dist = 0.5 * torch.sum(
                torch.abs(dist - pi_stationary)
            ).item()
            distances.append(tv_dist)
            dist = dist @ P

        ax1.semilogy(distances, marker='o', markersize=3,
                    label=f'Start: {state_names[start_state]}',
                    linewidth=2, alpha=0.7)

    ax1.set_xlabel('Time Step n', fontsize=12)
    ax1.set_ylabel('TV Distance (log scale)', fontsize=12)
    ax1.set_title('Convergence to Stationary Distribution', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Component-wise evolution from state 0
    ax2 = axes[1]
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
        ax2.axhline(y=pi_stationary[i].item(), linestyle='--',
                   color=f'C{i}', alpha=0.4)

    ax2.set_xlabel('Time Step n', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f'Component Evolution (Start: {state_names[0]})',
                  fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

## Detailed Balance and Reversibility

A Markov chain satisfies **detailed balance** with respect to distribution $\pi$ if:

$$\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j$$

This says the probability flow from $i$ to $j$ equals the flow from $j$ to $i$ in equilibrium. A chain satisfying detailed balance is called **reversible**.

**Proposition.** If $\pi$ satisfies detailed balance with $P$, then $\pi$ is a stationary distribution for $P$.

*Proof.* Summing the detailed balance equation over $i$:

$$\sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j$$

which is exactly $\pi = \pi P$. $\square$

Detailed balance is *sufficient* but not *necessary* for stationarity. However, it is the primary tool for designing MCMC algorithms: the Metropolis-Hastings algorithm (Section 18.3) constructs transition kernels that satisfy detailed balance with respect to the target distribution.

## Connection to MCMC Design

The stationary distribution theory provides the blueprint for MCMC:

| Theory | MCMC Application |
|--------|-----------------|
| $\pi = \pi P$ (fixed point) | Design $P$ so that $\pi$ is its stationary distribution |
| Uniqueness (irreducibility) | Ensure the MCMC chain can reach all states |
| Convergence (ergodicity) | Guarantee samples eventually approximate $\pi$ |
| Detailed balance | Primary tool for constructing valid MCMC kernels |
| Spectral gap | Determines how long to run the chain (burn-in) |

## Summary

| Method | Computation | Pros | Cons |
|--------|-------------|------|------|
| **Eigenvector** | $P^T v = v$, normalize | Exact, fast for small matrices | Numerical issues for large $N$ |
| **Linear System** | $(P^T - I)\pi^T = 0$ + normalization | Exact, numerically stable | Requires matrix factorization |
| **Power Iteration** | $\pi^{(k+1)} = \pi^{(k)}P$ | Simple, intuitive | Slow for small spectral gap |
| **Simulation** | Count state visits | Scales to large state spaces | Approximate, needs long runs |

## Exercises

1. **Multiple Stationary Distributions.** Construct a reducible Markov chain with two distinct stationary distributions and verify that both satisfy $\pi = \pi P$.

2. **Detailed Balance.** For the transition matrix $P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$, find the stationary distribution $\pi$ and verify that detailed balance holds.

3. **Convergence Rate.** Compute the spectral gap for the weather model. Verify empirically that the TV distance decays at rate $|\lambda_2|^n$.

4. **Customer Loyalty.** Model customer behavior with states $\{$Loyal, Neutral, Churned$\}$. Compute the long-run proportion of customers in each state and the expected time for a Neutral customer to become Loyal.

5. **Numerical Stability.** Compare the four methods on a nearly-reducible chain where some transition probabilities are close to 0. Which method is most robust?

## References

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times*, Chapters 1–4. AMS, 2017.
2. Norris, J.R. *Markov Chains*, Chapter 1. Cambridge University Press, 1997.
3. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, Chapter 4. Springer-Verlag, 1976.
4. Robert, C.P. & Casella, G. *Monte Carlo Statistical Methods*, Chapter 6. Springer, 2004.
