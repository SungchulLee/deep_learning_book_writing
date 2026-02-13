# Ergodicity

## Introduction

Ergodicity is the property that guarantees a Markov chain converges to its stationary distribution—the single most important condition for MCMC correctness. This section develops the classification machinery (communicating classes, irreducibility, aperiodicity, recurrence) that determines whether a chain is ergodic, then establishes the convergence theorems and mixing time analysis that quantify *how fast* convergence occurs.

Understanding ergodicity is essential for MCMC practitioners: it tells us when our sampler is valid (correctness), how long to run the burn-in period (convergence speed), and how to diagnose poorly mixing chains (structural bottlenecks).

## State Classification

### Accessibility and Communication

State $j$ is **accessible** from state $i$ (written $i \to j$) if there exists $n \geq 0$ such that:

$$P^{(n)}_{ij} > 0$$

States $i$ and $j$ **communicate** (written $i \leftrightarrow j$) if $i \to j$ and $j \to i$.

**Theorem.** Communication is an equivalence relation:

1. **Reflexive**: $i \leftrightarrow i$ (since $P^{(0)}_{ii} = 1$)
2. **Symmetric**: $i \leftrightarrow j$ implies $j \leftrightarrow i$ (by definition)
3. **Transitive**: $i \leftrightarrow j$ and $j \leftrightarrow k$ implies $i \leftrightarrow k$ (compose paths)

### Communicating Classes

The equivalence relation of communication partitions the state space $S$ into **communicating classes**. States in the same class can reach each other; states in different classes may have only one-way accessibility or no connection at all.

### Irreducibility

A Markov chain is **irreducible** if all states communicate:

$$i \leftrightarrow j \quad \text{for all } i, j \in S$$

Equivalently, the entire state space forms a single communicating class.

**Practical test.** A finite chain with $N$ states is irreducible if and only if $\sum_{k=1}^{N-1} P^k$ has all positive entries.

**MCMC implication.** If the MCMC chain is irreducible, the sampler can eventually reach every region of the state space from any starting point. A reducible chain would leave some states permanently unvisited.

## Periodicity

### Definition

The **period** of state $i$ is:

$$d(i) = \gcd\{n \geq 1 : P^{(n)}_{ii} > 0\}$$

A state is **aperiodic** if $d(i) = 1$ and **periodic** if $d(i) > 1$.

### Interpretation

A period of $d$ means the chain can only return to state $i$ at times that are multiples of $d$. Aperiodic states can return at "irregular" intervals, unconstrained to any fixed cycle.

**Key result.** In an irreducible chain, all states have the same period. This allows us to speak of the period of the chain itself.

### Sufficient Condition for Aperiodicity

If any state $i$ has a positive self-loop probability ($P_{ii} > 0$), then $d(i) = 1$, and since the chain is irreducible, all states are aperiodic. This is the most common way aperiodicity arises in practice—and it explains why many MCMC algorithms include a "stay" probability.

### Examples

**Periodic (period 2):**

$$P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \quad \Rightarrow \quad 0 \to 1 \to 0 \to 1 \to \cdots$$

Returns to state 0 only at even times. Period $= \gcd\{2, 4, 6, \ldots\} = 2$.

**Aperiodic (self-loop):**

$$P = \begin{pmatrix} 0.5 & 0.5 \\ 1.0 & 0.0 \end{pmatrix}$$

State 0 can return at time 1 (self-loop) or time 2. Period $= \gcd\{1, 2, 3, \ldots\} = 1$.

## Recurrence and Transience

### First Return Time

The **first return time** to state $i$ is:

$$T_i = \min\{n \geq 1 : X_n = i \mid X_0 = i\}$$

### Classification

A state $i$ is:

- **Recurrent** if $P(T_i < \infty \mid X_0 = i) = 1$ (certain to return)
- **Transient** if $P(T_i < \infty \mid X_0 = i) < 1$ (may never return)

### Equivalent Characterizations

State $i$ is recurrent if and only if $\sum_{n=0}^{\infty} P^{(n)}_{ii} = \infty$ (expected visits is infinite), and transient if and only if this sum is finite.

### Positive vs. Null Recurrence

Recurrent states are further classified:

- **Positive recurrent**: $\mathbb{E}[T_i \mid X_0 = i] < \infty$ (finite mean return time)
- **Null recurrent**: $\mathbb{E}[T_i \mid X_0 = i] = \infty$ (infinite mean return time)

For **finite** state spaces, all recurrent states are positive recurrent. Null recurrence only arises in countably infinite chains (e.g., the symmetric random walk on $\mathbb{Z}$).

## Ergodicity: The Complete Picture

### Definition

A Markov chain is **ergodic** if it is:

1. **Irreducible**: all states communicate
2. **Aperiodic**: the period equals 1
3. **Positive recurrent**: all states have finite expected return time

For finite state spaces, conditions 1 and 2 imply condition 3.

### Fundamental Convergence Theorem

**Theorem (Ergodic Theorem).** For an ergodic Markov chain with transition matrix $P$ and stationary distribution $\pi$:

1. A **unique** stationary distribution $\pi$ exists
2. For any initial distribution: $\displaystyle\lim_{n \to \infty} P^n_{ij} = \pi_j$ for all $i, j$
3. $\pi_i = 1/\mathbb{E}[T_i]$, where $T_i$ is the first return time to state $i$

Moreover, the convergence is **exponential**:

$$|P^n_{ij} - \pi_j| \leq C \cdot \rho^n$$

where $\rho = |\lambda_2| < 1$ is the magnitude of the second-largest eigenvalue.

### Implications for MCMC

This theorem is the mathematical foundation of MCMC:

| Property | Guarantee |
|----------|-----------|
| Irreducibility | The sampler explores the full state space |
| Aperiodicity | The sampler doesn't cycle deterministically |
| Positive recurrence | The sampler revisits every region infinitely often |
| Convergence theorem | After sufficient burn-in, samples approximate $\pi$ |
| Exponential rate | Convergence happens in $O(1/\gamma)$ steps where $\gamma$ is the spectral gap |

## Spectral Gap and Convergence Rate

### The Spectral Gap

The **spectral gap** of transition matrix $P$ is:

$$\gamma = 1 - |\lambda_2|$$

where $\lambda_2$ is the second-largest eigenvalue in absolute value (the largest is always $\lambda_1 = 1$).

The spectral gap directly controls the convergence rate:

- **Large $\gamma$** (close to 1): fast convergence, good mixing
- **Small $\gamma$** (close to 0): slow convergence, poor mixing
- $\gamma = 0$: the chain does not converge

### Structural Interpretation

| Chain Structure | Spectral Gap | Mixing |
|----------------|:---:|--------|
| Well-connected (all states easily reachable) | Large | Fast |
| Bottleneck (weak links between clusters) | Small | Slow |
| Nearly periodic | Very small | Very slow |

For MCMC, a small spectral gap signals a poorly designed proposal distribution: the chain gets "stuck" in local regions and takes a long time to traverse the full state space.

## Mixing Time

### Definition

The **mixing time** quantifies how many steps until the chain is "close" to its stationary distribution.

**Total variation distance:**

$$\|P^n(x, \cdot) - \pi\|_{TV} = \frac{1}{2} \sum_{y \in S} |P^n_{xy} - \pi_y|$$

**$\epsilon$-mixing time:**

$$\tau_{\text{mix}}(\epsilon) = \min\{n : \max_x \|P^n(x, \cdot) - \pi\|_{TV} \leq \epsilon\}$$

The standard choice is $\epsilon = 1/4$, written simply as $\tau_{\text{mix}}$.

### Mixing Time Bounds

The mixing time is bounded by the spectral gap:

$$\frac{1}{\gamma} \leq \tau_{\text{mix}} \leq \frac{\log(1/\epsilon \cdot \pi_{\min})}{\gamma}$$

where $\pi_{\min} = \min_i \pi_i$.

### Why Mixing Time Matters for MCMC

In MCMC practice:

- **Burn-in period**: discard the first $\sim \tau_{\text{mix}}$ samples before collecting
- **Thinning interval**: keep every $k$-th sample where $k \sim \tau_{\text{mix}}$ to reduce autocorrelation
- **Efficiency**: faster mixing $=$ more effectively independent samples per unit computation

## PyTorch Implementation

### State Classifier

```python
import torch
from typing import Dict, List, Set, Tuple
from math import gcd

class StateClassifier:
    """
    Classify states of a Markov chain: communicating classes,
    irreducibility, periodicity, and ergodicity.
    """

    def __init__(self, transition_matrix: torch.Tensor):
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]

    def is_accessible(
        self, i: int, j: int, max_steps: int = None
    ) -> bool:
        """Check if state j is accessible from state i."""
        if max_steps is None:
            max_steps = self.n_states

        P_sum = torch.zeros_like(self.P)
        P_k = self.P.clone()
        for _ in range(max_steps):
            P_sum += P_k
            P_k = P_k @ self.P

        return P_sum[i, j].item() > 0

    def communicates(self, i: int, j: int) -> bool:
        """Check if states i and j communicate (i ↔ j)."""
        return self.is_accessible(i, j) and self.is_accessible(j, i)

    def find_communicating_classes(self) -> List[Set[int]]:
        """Partition state space into communicating classes."""
        visited = [False] * self.n_states
        classes = []

        for start in range(self.n_states):
            if visited[start]:
                continue
            current_class = {start}
            visited[start] = True

            for other in range(self.n_states):
                if not visited[other] and self.communicates(start, other):
                    current_class.add(other)
                    visited[other] = True

            classes.append(current_class)

        return classes

    def is_irreducible(self) -> bool:
        """Check if the chain has a single communicating class."""
        return len(self.find_communicating_classes()) == 1

    def compute_period(self, state: int) -> int:
        """
        Compute period d(i) = gcd{n ≥ 1 : P^{(n)}_{ii} > 0}.
        """
        return_times = []
        P_n = self.P.clone()

        for n in range(1, 2 * self.n_states + 1):
            if P_n[state, state].item() > 1e-10:
                return_times.append(n)
            P_n = P_n @ self.P

        if not return_times:
            return 0

        period = return_times[0]
        for t in return_times[1:]:
            period = gcd(period, t)
            if period == 1:
                break

        return period

    def is_aperiodic(self) -> bool:
        """Check aperiodicity (sufficient: any self-loop)."""
        if torch.any(torch.diag(self.P) > 0):
            return True
        return self.compute_period(0) == 1

    def is_ergodic(self) -> Dict[str, bool]:
        """
        Check ergodicity = irreducible + aperiodic.
        For finite chains, this implies positive recurrence.
        """
        is_irr = self.is_irreducible()
        is_aper = self.is_aperiodic()
        return {
            'irreducible': is_irr,
            'aperiodic': is_aper,
            'ergodic': is_irr and is_aper
        }

    def classify_all_states(self) -> Dict[int, Dict]:
        """Full classification of every state."""
        classes = self.find_communicating_classes()
        state_to_class = {}
        for idx, cls in enumerate(classes):
            for state in cls:
                state_to_class[state] = idx

        results = {}
        for state in range(self.n_states):
            results[state] = {
                'communicating_class': state_to_class[state],
                'period': self.compute_period(state),
                'has_self_loop': self.P[state, state].item() > 0
            }

        return results
```

### Mixing Time Analyzer

```python
class MixingTimeAnalyzer:
    """
    Analyze convergence and mixing properties: spectral gap,
    mixing time, total variation distance evolution.
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: list = None
    ):
        self.P = transition_matrix.clone().double()
        self.n_states = self.P.shape[0]
        self.state_names = state_names or [
            f"S{i}" for i in range(self.n_states)
        ]
        self._compute_spectrum()
        self._compute_stationary()

    def _compute_spectrum(self):
        """Compute eigenvalues sorted by absolute value."""
        eigenvalues, eigenvectors = torch.linalg.eig(self.P)
        abs_vals = torch.abs(eigenvalues.real)
        sorted_idx = torch.argsort(abs_vals, descending=True)
        self.eigenvalues = eigenvalues[sorted_idx]
        self.eigenvectors = eigenvectors[:, sorted_idx]

    def _compute_stationary(self):
        """Extract stationary distribution from eigenvector."""
        idx = torch.argmin(torch.abs(self.eigenvalues.real - 1.0))
        pi = self.eigenvectors[:, idx].real
        self.pi = torch.abs(pi) / torch.abs(pi).sum()

    def spectral_gap(self) -> float:
        """Spectral gap γ = 1 - |λ₂|."""
        lambda_2 = self.eigenvalues[1]
        return (1 - torch.abs(lambda_2)).item()

    def total_variation_distance(
        self, dist1: torch.Tensor, dist2: torch.Tensor
    ) -> float:
        """TV(μ, ν) = (1/2) Σ |μ(x) - ν(x)|."""
        return 0.5 * torch.sum(torch.abs(dist1 - dist2)).item()

    def mixing_time(
        self,
        epsilon: float = 0.25,
        max_steps: int = 10000
    ) -> Dict:
        """
        Compute τ_mix(ε): first n such that max_x TV(P^n(x,·), π) ≤ ε.
        """
        results = {
            'epsilon': epsilon,
            'mixing_time': None,
            'max_tv_over_time': []
        }

        for step in range(max_steps):
            P_n = torch.linalg.matrix_power(self.P, step)

            max_tv = 0
            for i in range(self.n_states):
                tv = self.total_variation_distance(P_n[i], self.pi)
                max_tv = max(max_tv, tv)

            results['max_tv_over_time'].append(max_tv)

            if max_tv <= epsilon and results['mixing_time'] is None:
                results['mixing_time'] = step

        return results

    def convergence_rate(self, max_steps: int = 100) -> Dict:
        """TV distance evolution from each starting state."""
        results = {
            'spectral_gap': self.spectral_gap(),
            'second_eigenvalue': self.eigenvalues[1].item(),
            'theoretical_rate': torch.abs(self.eigenvalues[1]).item(),
            'distances': {}
        }

        for start_idx in range(self.n_states):
            distances = []
            dist = torch.zeros(self.n_states, dtype=self.P.dtype)
            dist[start_idx] = 1.0

            for step in range(max_steps):
                tv = self.total_variation_distance(
                    dist.float(), self.pi.float()
                )
                distances.append(tv)
                dist = dist @ self.P

            results['distances'][self.state_names[start_idx]] = distances

        return results
```

### Demonstration: Fast vs. Slow Mixing

```python
def demonstrate_mixing_analysis():
    """Compare mixing behavior of well-connected vs. bottleneck chains."""
    print("Mixing Time Analysis")
    print("=" * 70)

    # Fast mixing: well-connected
    print("\n1. Fast Mixing Chain (Well-Connected)")
    print("-" * 50)

    P_fast = torch.tensor([
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ])

    analyzer_fast = MixingTimeAnalyzer(P_fast, ['A', 'B', 'C'])
    print(f"Spectral gap: {analyzer_fast.spectral_gap():.6f}")
    print(f"|λ₂|: {torch.abs(analyzer_fast.eigenvalues[1]).item():.6f}")

    mixing_fast = analyzer_fast.mixing_time(epsilon=0.01)
    print(f"Mixing time (ε=0.01): {mixing_fast['mixing_time']} steps")

    # Slow mixing: bottleneck
    print("\n2. Slow Mixing Chain (Bottleneck)")
    print("-" * 50)

    P_slow = torch.tensor([
        [0.45, 0.45, 0.05, 0.05],
        [0.45, 0.45, 0.05, 0.05],
        [0.05, 0.05, 0.45, 0.45],
        [0.05, 0.05, 0.45, 0.45]
    ])

    analyzer_slow = MixingTimeAnalyzer(P_slow, ['A1', 'A2', 'B1', 'B2'])
    print(f"Spectral gap: {analyzer_slow.spectral_gap():.6f}")
    print(f"|λ₂|: {torch.abs(analyzer_slow.eigenvalues[1]).item():.6f}")

    mixing_slow = analyzer_slow.mixing_time(epsilon=0.01, max_steps=500)
    print(f"Mixing time (ε=0.01): {mixing_slow['mixing_time']} steps")

    # Comparison
    print(f"\nSpectral gap ratio: "
          f"{analyzer_fast.spectral_gap() / analyzer_slow.spectral_gap():.1f}x")


demonstrate_mixing_analysis()
```

## Visualization

### Chain Structure Graph

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_chain_structure(
    P: torch.Tensor,
    state_names: List[str] = None,
    title: str = "Markov Chain Structure"
):
    """Visualize chain as directed graph colored by communicating class."""
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    classifier = StateClassifier(P)
    classes = classifier.find_communicating_classes()
    state_info = classifier.classify_all_states()

    G = nx.DiGraph()
    for i in range(n_states):
        G.add_node(i, label=state_names[i])
    for i in range(n_states):
        for j in range(n_states):
            if P[i, j].item() > 0.01:
                G.add_edge(i, j, weight=P[i, j].item())

    colors = plt.cm.Set3(range(len(classes)))
    node_colors = [colors[state_info[s]['communicating_class']]
                   for s in range(n_states)]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000,
                          node_color=node_colors, alpha=0.8)

    edges = G.edges(data=True)
    edge_weights = [e[2]['weight'] for e in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                          arrowsize=20, edge_color='gray',
                          width=[w * 3 for w in edge_weights],
                          alpha=0.6, connectionstyle="arc3,rad=0.1")

    labels = {i: state_names[i] for i in range(n_states)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=12,
                           font_weight='bold')

    edge_labels = {(i, j): f"{P[i,j]:.2f}"
                   for i, j in G.edges() if P[i, j].item() > 0.01}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    return fig


def visualize_convergence(
    P: torch.Tensor,
    state_names: list = None,
    max_steps: int = 50
):
    """Four-panel convergence visualization."""
    analyzer = MixingTimeAnalyzer(P, state_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) TV distance over time
    ax = axes[0, 0]
    conv_data = analyzer.convergence_rate(max_steps)
    for state, distances in conv_data['distances'].items():
        ax.semilogy(distances, marker='o', markersize=3,
                   label=f'Start: {state}', linewidth=2, alpha=0.7)
    rate = conv_data['theoretical_rate']
    theoretical = [rate ** n for n in range(max_steps)]
    ax.semilogy(theoretical, 'k--', linewidth=2, alpha=0.5,
               label=f'|λ₂|^n = {rate:.3f}^n')
    ax.set_xlabel('Time Step n')
    ax.set_ylabel('Total Variation Distance')
    ax.set_title('Convergence Rate (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Eigenvalue spectrum
    ax = axes[0, 1]
    eigenvalues = analyzer.eigenvalues
    ax.scatter(eigenvalues.real.numpy(), eigenvalues.imag.numpy(),
              s=200, c='blue', alpha=0.7, edgecolors='black', linewidth=2)
    ax.scatter([1], [0], s=300, c='red', marker='*', label='λ₁ = 1')
    ax.scatter([eigenvalues[1].real.item()], [eigenvalues[1].imag.item()],
              s=300, c='green', marker='*',
              label=f'λ₂ = {eigenvalues[1].real:.3f}')
    theta = torch.linspace(0, 2 * 3.14159, 100)
    ax.plot(torch.cos(theta).numpy(), torch.sin(theta).numpy(),
           'k--', alpha=0.3)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Eigenvalue Spectrum (γ = {analyzer.spectral_gap():.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # (1,0) State distribution evolution
    ax = axes[1, 0]
    n_states = P.shape[0]
    sn = state_names or [f'S{i}' for i in range(n_states)]

    dist = torch.zeros(n_states, dtype=P.dtype)
    dist[0] = 1.0
    evolution = [dist.clone()]
    for _ in range(max_steps):
        dist = dist @ P
        evolution.append(dist.clone())
    evolution = torch.stack(evolution)

    for i in range(n_states):
        ax.plot(evolution[:, i].numpy(), marker='o', markersize=3,
               label=sn[i], linewidth=2, alpha=0.7)
        ax.axhline(y=analyzer.pi[i].item(), linestyle='--',
                  color=f'C{i}', alpha=0.4)
    ax.set_xlabel('Time Step n')
    ax.set_ylabel('Probability')
    ax.set_title(f'Distribution Evolution (Start: {sn[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Mixing time
    ax = axes[1, 1]
    mixing_data = analyzer.mixing_time(epsilon=0.25, max_steps=max_steps)
    ax.semilogy(mixing_data['max_tv_over_time'], 'b-', linewidth=2,
               label='Max TV distance')
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2,
              label='ε = 0.25')
    if mixing_data['mixing_time'] is not None:
        ax.axvline(x=mixing_data['mixing_time'], color='green',
                  linestyle='--', linewidth=2,
                  label=f'τ_mix = {mixing_data["mixing_time"]}')
    ax.set_xlabel('Time Step n')
    ax.set_ylabel('Max TV Distance')
    ax.set_title('Mixing Time Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

## Examples: Ergodic, Periodic, and Reducible Chains

```python
# Example 1: Ergodic chain
print("Example 1: Ergodic Chain")
print("=" * 50)

P_ergodic = torch.tensor([
    [0.5, 0.3, 0.2],
    [0.2, 0.6, 0.2],
    [0.3, 0.3, 0.4]
])

classifier = StateClassifier(P_ergodic)
result = classifier.is_ergodic()
print(f"Irreducible: {result['irreducible']}")
print(f"Aperiodic: {result['aperiodic']}")
print(f"Ergodic: {result['ergodic']}")
# → Ergodic: True. Unique π exists and P^n converges.


# Example 2: Periodic chain (period 3)
print("\nExample 2: Periodic Chain")
print("=" * 50)

P_periodic = torch.tensor([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0]
])

classifier_periodic = StateClassifier(P_periodic)
print(f"This chain cycles: 0 → 1 → 2 → 0 → ...")
print(f"Period of state 0: {classifier_periodic.compute_period(0)}")
result = classifier_periodic.is_ergodic()
print(f"Ergodic: {result['ergodic']} (irreducible but periodic)")
# → Not ergodic. π exists and is unique (irreducible), but P^n oscillates.


# Example 3: Reducible chain
print("\nExample 3: Reducible Chain")
print("=" * 50)

P_reducible = torch.tensor([
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.7, 0.3],
    [0.0, 0.0, 0.4, 0.6]
])

classifier_reducible = StateClassifier(P_reducible)
classes = classifier_reducible.find_communicating_classes()
print(f"Communicating classes: {classes}")
print(f"Irreducible: {classifier_reducible.is_irreducible()}")
# → Not irreducible. Multiple stationary distributions exist.
```

## Key Theorems Summary

| Theorem | Statement | MCMC Relevance |
|---------|-----------|----------------|
| **Existence** | Every finite chain has $\geq 1$ stationary distribution | Baseline guarantee |
| **Uniqueness** | Irreducible $\Rightarrow$ exactly one $\pi$ | MCMC converges to the right target |
| **Convergence** | Ergodic $\Rightarrow$ $P^n_{ij} \to \pi_j$ for all $i,j$ | Samples approximate $\pi$ after burn-in |
| **Exponential Rate** | $\|P^n(x,\cdot) - \pi\|_{TV} \leq C \cdot |\lambda_2|^n$ | Quantifies burn-in length |
| **Mixing Time Bound** | $\tau_{\text{mix}} \sim 1/\gamma$ | Spectral gap determines efficiency |

## Summary

| Property | Definition | Implication |
|----------|------------|-------------|
| **Accessible** | $i \to j$: $\exists n,\, P^{(n)}_{ij} > 0$ | Can reach $j$ from $i$ |
| **Communicate** | $i \leftrightarrow j$ | Can reach each other |
| **Irreducible** | Single communicating class | Full state space exploration |
| **Aperiodic** | Period $= 1$ | No deterministic cycles |
| **Recurrent** | $P(\text{return}) = 1$ | Certain to revisit |
| **Transient** | $P(\text{return}) < 1$ | May never revisit |
| **Ergodic** | Irreducible + Aperiodic | Unique $\pi$, guaranteed convergence |
| **Spectral Gap** | $\gamma = 1 - |\lambda_2|$ | Controls convergence speed |
| **Mixing Time** | Steps until TV $\leq \epsilon$ | Practical burn-in measure |

## Exercises

1. **Periodicity Detection.** Show that if any state has a positive self-loop probability ($P_{ii} > 0$) in an irreducible chain, all states are aperiodic.

2. **Finite Recurrence.** Prove that in a finite irreducible chain, all states are positive recurrent. (*Hint*: use the pigeonhole principle.)

3. **Bottleneck Effect.** Construct a chain with two clusters of 3 states each. Vary the inter-cluster transition probability $\alpha \in \{0.01, 0.05, 0.1, 0.3\}$ and plot the mixing time as a function of $\alpha$.

4. **Spectral Gap Computation.** For a 3×3 transition matrix of your choice, compute the spectral gap analytically and verify that the empirical convergence rate matches $|\lambda_2|^n$.

5. **MCMC Preview.** Given target distribution $\pi = (0.2, 0.3, 0.5)$, construct a transition matrix $P$ that has $\pi$ as its stationary distribution and satisfies detailed balance. Verify ergodicity and compute the mixing time.

## References

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times* (2nd ed.). AMS, 2017.
2. Norris, J.R. *Markov Chains*, Chapters 1–2. Cambridge University Press, 1997.
3. Montenegro, R. & Tetali, P. "Mathematical Aspects of Mixing Times in Markov Chains." *Foundations and Trends in TCS*, 2006.
4. Diaconis, P. & Stroock, D. "Geometric Bounds for Eigenvalues of Markov Chains." *Annals of Applied Probability*, 1991.
5. Durrett, R. *Essentials of Stochastic Processes*, Chapter 1. Springer, 2016.

---

## Mixing Time

**Definition**: Time for chain to get "close" to stationary distribution.

**Formal**: Total variation mixing time

$$
\tau_{\text{mix}}(\epsilon) = \min\{t : \sup_x \|P^t(x, \cdot) - \pi\|_{\text{TV}} \leq \epsilon\}
$$

### What Affects Mixing Time?

1. **Dimension**: Higher dimension → longer mixing
2. **Correlation**: Strong correlations → slower mixing
3. **Multimodality**: Separated modes → much longer mixing
4. **Curvature**: Flat regions vs steep → affects exploration

### Typical Scaling

| Method | Mixing Time |
|--------|-------------|
| Random walk MH | $\sim d^2$ |
| Langevin (MALA) | $\sim d^{5/3}$ |
| HMC | $\sim d^{5/4}$ or better |

This scaling is why HMC dominates in high dimensions.

---

## The Monte Carlo Error

Even with perfect convergence, Monte Carlo has **statistical error**.

### Variance of Estimator

$$
\text{Var}\left[\frac{1}{N}\sum_{t=1}^N f(X^{(t)})\right] = \frac{\sigma_f^2}{N_{\text{eff}}}
$$

where $\sigma_f^2 = \text{Var}_\pi[f]$ and $N_{\text{eff}}$ is the **effective sample size**:

$$
N_{\text{eff}} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

where $\rho_k$ is the autocorrelation at lag $k$.

---

## Practical Guidelines

### Starting the Chain

- **Random initialization**: Often works, but may require long burn-in
- **Informed initialization**: Use MAP estimate, prior mode, or previous fit
- **Multiple chains**: Start from different points to check convergence

### Running the Chain

- **Burn-in**: Discard initial 50% (conservative) or use diagnostics
- **Thinning**: Keep every $k$-th sample (debated — usually unnecessary)
- **Monitoring**: Use $\hat{R}$, ESS, trace plots

### Stopping the Chain

- **Minimum**: Until $\hat{R} < 1.01$ and ESS $> 100$
- **Preferred**: ESS $> 1000$ for reliable inference
- **Critical**: Even more samples for tails, quantiles

### The Art of MCMC

Using MCMC well requires understanding the target distribution, choosing the appropriate algorithm, monitoring convergence carefully, and interpreting results with suitable skepticism. The beauty of MCMC is that we can sample from arbitrarily complex distributions using only the ability to evaluate $\tilde{\pi}(x)$ (unnormalized), a clever transition kernel, and patience.
