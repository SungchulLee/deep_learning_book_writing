# State Classification

## Introduction

Not all states in a Markov chain behave alike. Some states communicate with each other, some lead to "dead ends," and some exhibit periodic behavior. Understanding the classification of states is essential for analyzing the long-term behavior of Markov chains and determining whether unique stationary distributions exist.

## Accessibility and Communication

### Accessibility

State $j$ is **accessible** from state $i$ (written $i \to j$) if there exists some $n \geq 0$ such that:

$$P^{(n)}_{ij} > 0$$

This means it is possible (with positive probability) to reach state $j$ starting from state $i$ in some finite number of steps.

### Communication

States $i$ and $j$ **communicate** (written $i \leftrightarrow j$) if:

$$i \to j \text{ and } j \to i$$

Both states can reach each other.

**Theorem**: Communication is an equivalence relation:
1. **Reflexive**: $i \leftrightarrow i$ (stay in place with $P^{(0)}_{ii} = 1$)
2. **Symmetric**: If $i \leftrightarrow j$ then $j \leftrightarrow i$ (by definition)
3. **Transitive**: If $i \leftrightarrow j$ and $j \leftrightarrow k$ then $i \leftrightarrow k$

### Communicating Classes

The equivalence relation of communication partitions the state space into **communicating classes**. States in the same class can all reach each other; states in different classes may have one-way accessibility or no connection at all.

## Irreducibility

### Definition

A Markov chain is **irreducible** if all states communicate with each other:

$$i \leftrightarrow j \quad \text{for all } i, j \in S$$

Equivalently, the entire state space forms a single communicating class.

### Practical Test

A chain is irreducible if and only if for some (equivalently, all) $n \geq N-1$:

$$P^n \text{ has all positive entries}$$

where $N$ is the number of states.

**Intuition**: In an irreducible chain, you can eventually get from anywhere to anywhere else.

## Periodicity

### Definition

The **period** of state $i$ is:

$$d(i) = \gcd\{n \geq 1 : P^{(n)}_{ii} > 0\}$$

where $\gcd$ denotes the greatest common divisor.

State $i$ is:
- **Aperiodic** if $d(i) = 1$
- **Periodic** if $d(i) > 1$

### Interpretation

- Period $d$ means returns to state $i$ can only occur at times that are multiples of $d$
- Aperiodic states can return at "irregular" intervals (not constrained to multiples)

### Key Result

**Theorem**: In an irreducible Markov chain, all states have the same period.

This allows us to speak of the period of an irreducible chain, not just individual states.

### Examples of Periodicity

**Period 2 (Alternating)**:
```
States: {0, 1}
P = [[0, 1],
     [1, 0]]
     
The chain alternates: 0 → 1 → 0 → 1 → ...
Can only return to state 0 at even times.
Period = 2
```

**Aperiodic (Self-loop)**:
```
States: {0, 1}
P = [[0.5, 0.5],
     [1.0, 0.0]]
     
From state 0: can return in 1 step (self-loop) or 2 steps
gcd{1, 2, 3, ...} = 1
Period = 1 (aperiodic)
```

## Recurrence and Transience

### First Return Time

The **first return time** to state $i$ is:

$$T_i = \min\{n \geq 1 : X_n = i \mid X_0 = i\}$$

### Classification

State $i$ is:
- **Recurrent** if $P(T_i < \infty \mid X_0 = i) = 1$ (certain to return)
- **Transient** if $P(T_i < \infty \mid X_0 = i) < 1$ (may never return)

### Equivalent Characterizations

State $i$ is **recurrent** if and only if:

$$\sum_{n=0}^{\infty} P^{(n)}_{ii} = \infty$$

State $i$ is **transient** if and only if:

$$\sum_{n=0}^{\infty} P^{(n)}_{ii} < \infty$$

**Intuition**: 
- Recurrent: expected number of returns is infinite
- Transient: expected number of returns is finite

### Positive vs Null Recurrence

Recurrent states are further classified:
- **Positive recurrent**: $E[T_i \mid X_0 = i] < \infty$ (finite expected return time)
- **Null recurrent**: $E[T_i \mid X_0 = i] = \infty$ (infinite expected return time)

For **finite** state spaces, all recurrent states are positive recurrent.

## Ergodicity

### Definition

A Markov chain is **ergodic** if it is:
1. **Irreducible**: all states communicate
2. **Aperiodic**: period equals 1
3. **Positive recurrent**: all states have finite expected return time

For finite state spaces, conditions 1 and 2 imply condition 3.

### Fundamental Convergence Theorem

**Theorem**: For an ergodic Markov chain:

1. A **unique** stationary distribution $\pi$ exists
2. For any initial distribution, $\lim_{n \to \infty} P^n_{ij} = \pi_j$ for all $i, j$
3. $\pi_i = 1/E[T_i]$ (stationary probability equals reciprocal of mean return time)

## PyTorch Implementation

### State Classification Functions

```python
import torch
from typing import Dict, List, Set, Tuple

class StateClassifier:
    """
    Tools for classifying states in a Markov chain.
    
    Determines:
    - Communicating classes
    - Irreducibility
    - Periodicity
    - Recurrence/transience
    """
    
    def __init__(self, transition_matrix: torch.Tensor):
        """
        Initialize classifier.
        
        Args:
            transition_matrix: N×N transition probability matrix
        """
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]
    
    def is_accessible(self, i: int, j: int, max_steps: int = None) -> bool:
        """
        Check if state j is accessible from state i.
        
        j is accessible from i if P^{(n)}_{ij} > 0 for some n.
        
        Args:
            i: Source state
            j: Target state
            max_steps: Maximum steps to check (default: n_states)
            
        Returns:
            True if j is accessible from i
        """
        if max_steps is None:
            max_steps = self.n_states
        
        # Compute sum of P^1 + P^2 + ... + P^{max_steps}
        P_sum = torch.zeros_like(self.P)
        P_k = self.P.clone()
        
        for _ in range(max_steps):
            P_sum += P_k
            P_k = P_k @ self.P
        
        return P_sum[i, j].item() > 0
    
    def communicates(self, i: int, j: int) -> bool:
        """
        Check if states i and j communicate (i ↔ j).
        
        Args:
            i, j: States to check
            
        Returns:
            True if i and j communicate
        """
        return self.is_accessible(i, j) and self.is_accessible(j, i)
    
    def find_communicating_classes(self) -> List[Set[int]]:
        """
        Find all communicating classes.
        
        Returns:
            List of sets, each set is a communicating class
        """
        # Union-Find approach
        visited = [False] * self.n_states
        classes = []
        
        for start in range(self.n_states):
            if visited[start]:
                continue
            
            # Find all states that communicate with 'start'
            current_class = {start}
            visited[start] = True
            
            for other in range(self.n_states):
                if not visited[other] and self.communicates(start, other):
                    current_class.add(other)
                    visited[other] = True
            
            classes.append(current_class)
        
        return classes
    
    def is_irreducible(self) -> bool:
        """
        Check if the chain is irreducible (single communicating class).
        
        A chain is irreducible iff all states communicate.
        
        Returns:
            True if irreducible
        """
        classes = self.find_communicating_classes()
        return len(classes) == 1
    
    def check_irreducibility_via_powers(self) -> Tuple[bool, int]:
        """
        Check irreducibility by examining if P^k has all positive entries.
        
        Returns:
            (is_irreducible, k) where k is smallest power with all positive
            entries, or (False, -1) if reducible
        """
        P_k = self.P.clone()
        
        for k in range(1, self.n_states + 1):
            if torch.all(P_k > 0):
                return True, k
            P_k = P_k @ self.P
        
        return False, -1
    
    def compute_period(self, state: int) -> int:
        """
        Compute the period of a state.
        
        Period d(i) = gcd{n ≥ 1 : P^{(n)}_{ii} > 0}
        
        Args:
            state: State to analyze
            
        Returns:
            Period of the state
        """
        from math import gcd
        
        # Find times when P^n[state, state] > 0
        return_times = []
        P_n = self.P.clone()
        
        for n in range(1, 2 * self.n_states + 1):
            if P_n[state, state].item() > 1e-10:
                return_times.append(n)
            P_n = P_n @ self.P
        
        if not return_times:
            return 0  # State may be transient
        
        # Compute GCD of return times
        period = return_times[0]
        for t in return_times[1:]:
            period = gcd(period, t)
            if period == 1:
                break
        
        return period
    
    def is_aperiodic(self) -> bool:
        """
        Check if the chain is aperiodic.
        
        For irreducible chains, check any single state.
        
        Returns:
            True if aperiodic
        """
        # Check if any state has a self-loop (sufficient condition)
        if torch.any(torch.diag(self.P) > 0):
            return True
        
        # Otherwise, compute period of state 0
        return self.compute_period(0) == 1
    
    def is_ergodic(self) -> Dict[str, bool]:
        """
        Check if the chain is ergodic.
        
        Ergodic = Irreducible + Aperiodic (+ Positive Recurrent)
        For finite chains, irreducible + aperiodic implies positive recurrent.
        
        Returns:
            Dictionary with component checks and overall result
        """
        is_irr = self.is_irreducible()
        is_aper = self.is_aperiodic()
        
        return {
            'irreducible': is_irr,
            'aperiodic': is_aper,
            'ergodic': is_irr and is_aper
        }
    
    def classify_all_states(self) -> Dict[int, Dict]:
        """
        Classify all states in the chain.
        
        Returns:
            Dictionary mapping state -> classification info
        """
        classes = self.find_communicating_classes()
        
        # Map each state to its class
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

### Example: Classifying States

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

is_irr, k = classifier.check_irreducibility_via_powers()
if is_irr:
    print(f"P^{k} has all positive entries")


# Example 2: Periodic chain
print("\n" + "=" * 50)
print("Example 2: Periodic Chain (Alternating)")
print("=" * 50)

P_periodic = torch.tensor([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0]
])

classifier_periodic = StateClassifier(P_periodic)

print("This chain cycles: 0 → 1 → 2 → 0 → ...")
print(f"Period of state 0: {classifier_periodic.compute_period(0)}")
result = classifier_periodic.is_ergodic()
print(f"Ergodic: {result['ergodic']} (irreducible but periodic)")


# Example 3: Reducible chain
print("\n" + "=" * 50)
print("Example 3: Reducible Chain (Two Components)")
print("=" * 50)

P_reducible = torch.tensor([
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.7, 0.3],
    [0.0, 0.0, 0.4, 0.6]
])

classifier_reducible = StateClassifier(P_reducible)

print("Two separate components: {0, 1} and {2, 3}")
classes = classifier_reducible.find_communicating_classes()
for i, cls in enumerate(classes):
    print(f"  Class {i+1}: {cls}")

print(f"Irreducible: {classifier_reducible.is_irreducible()}")
```

## Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_chain_structure(
    P: torch.Tensor,
    state_names: List[str] = None,
    title: str = "Markov Chain Structure"
):
    """
    Visualize Markov chain as directed graph with classifications.
    """
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]
    
    classifier = StateClassifier(P)
    classes = classifier.find_communicating_classes()
    state_info = classifier.classify_all_states()
    
    # Create directed graph
    G = nx.DiGraph()
    
    for i in range(n_states):
        G.add_node(i, label=state_names[i])
    
    for i in range(n_states):
        for j in range(n_states):
            if P[i, j].item() > 0.01:
                G.add_edge(i, j, weight=P[i, j].item())
    
    # Assign colors by communicating class
    colors = plt.cm.Set3(range(len(classes)))
    node_colors = []
    for state in range(n_states):
        class_idx = state_info[state]['communicating_class']
        node_colors.append(colors[class_idx])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000,
                          node_color=node_colors, alpha=0.8)
    
    # Draw edges with weights
    edges = G.edges(data=True)
    edge_weights = [e[2]['weight'] for e in edges]
    
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                          arrowsize=20, edge_color='gray',
                          width=[w * 3 for w in edge_weights],
                          alpha=0.6, connectionstyle="arc3,rad=0.1")
    
    # Labels
    labels = {i: state_names[i] for i in range(n_states)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=12,
                           font_weight='bold')
    
    # Edge labels (transition probabilities)
    edge_labels = {(i, j): f"{P[i,j]:.2f}" 
                   for i, j in G.edges() if P[i, j].item() > 0.01}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
    
    # Legend for classes
    legend_elements = []
    for i, cls in enumerate(classes):
        states_str = ", ".join(state_names[s] for s in cls)
        legend_elements.append(plt.scatter([], [], c=[colors[i]], s=100,
                              label=f"Class {i+1}: {{{states_str}}}"))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig
```

## Summary Table

| Property | Definition | Implication |
|----------|------------|-------------|
| **Accessible** | $i \to j$: $\exists n, P^{(n)}_{ij} > 0$ | Can reach $j$ from $i$ |
| **Communicate** | $i \leftrightarrow j$ | Can reach each other |
| **Irreducible** | Single communicating class | Can go anywhere from anywhere |
| **Aperiodic** | Period $= 1$ | No fixed cycle length |
| **Recurrent** | $P(\text{return}) = 1$ | Certain to revisit |
| **Transient** | $P(\text{return}) < 1$ | May never revisit |
| **Ergodic** | Irreducible + Aperiodic | Unique stationary distribution, convergence guaranteed |

## Exercises

1. **Classification**: For the matrix $P = \begin{pmatrix} 0.5 & 0.5 & 0 \\ 0.3 & 0 & 0.7 \\ 0 & 0.4 & 0.6 \end{pmatrix}$, find all communicating classes.

2. **Periodicity**: Show that if any state has a positive self-loop probability ($P_{ii} > 0$), the chain is aperiodic.

3. **Recurrence**: Prove that in a finite irreducible chain, all states are recurrent.

4. **Application**: Model a system with "working," "degraded," and "failed" states. Determine if repair is always possible (irreducibility) and classify the states.

## References

1. Norris, J.R. *Markov Chains*, Chapters 1-2. Cambridge University Press, 1997.
2. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times*, Chapter 1. AMS, 2017.
3. Durrett, R. *Essentials of Stochastic Processes*, Chapter 1. Springer, 2016.
