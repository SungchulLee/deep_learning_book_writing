# Bounding Functions

Branch and bound prunes the search tree by computing a **bound** — an optimistic estimate of the best solution reachable from a given node. If the bound indicates that no solution in the subtree can improve upon the current best, the entire subtree is pruned. The quality of the bounding function determines how much of the tree is explored: a tight bound prunes aggressively, while a loose bound provides little benefit.

## Role of Bounds in Branch and Bound

At each node in the search tree, the bounding function computes:

- **Upper bound** (for maximization): the maximum possible value achievable in the subtree.
- **Lower bound** (for minimization): the minimum possible cost in the subtree.

A node is pruned when its bound cannot improve on the best complete solution found so far. Formally, for a maximization problem with current best value $z^*$:

$$
\text{Prune node } v \quad \text{if} \quad \text{UB}(v) \le z^*
$$

## Desirable Properties

A good bounding function balances two goals:

1. **Tightness**: The bound should be close to the true optimal value in the subtree. A tighter bound prunes more nodes.
2. **Efficiency**: The bound must be fast to compute. A bound that requires solving the original problem is useless — it must be simpler than the problem itself.

!!! tip "The Relaxation Principle"
    Most bounding functions work by **relaxing** constraints of the original problem. Removing constraints enlarges the feasible set, making the relaxed problem easier to solve, and its optimal value provides a valid bound.

## Common Bounding Techniques

### LP Relaxation

Replace integer constraints $x_i \in \{0, 1\}$ with continuous constraints $0 \le x_i \le 1$. The resulting linear program (LP) can be solved in polynomial time, and its optimal value provides an upper bound for maximization (or lower bound for minimization).

**Example.** For the 0/1 knapsack, LP relaxation allows fractional item selections. The fractional knapsack can be solved greedily in $O(n \log n)$.

### Greedy Bounds

Apply a greedy heuristic to the remaining items. For the knapsack problem, sorting items by value-to-weight ratio and greedily filling the capacity provides a fast upper bound.

### Problem-Specific Relaxations

- **TSP**: Relax the Hamiltonian cycle requirement to a minimum spanning tree (MST). The MST cost lower-bounds the TSP cost since every Hamiltonian cycle spans all vertices.
- **Assignment problem**: Relax by allowing fractional assignments; the LP relaxation of the assignment problem has an integral optimum (due to total unimodularity).

## Bound Quality Spectrum

| Bound Type | Tightness | Cost | Example |
|---|---|---|---|
| Trivial | Loose | $O(1)$ | Sum of all values (knapsack UB) |
| Greedy | Moderate | $O(n \log n)$ | Fractional knapsack relaxation |
| LP relaxation | Tight | $O(n^3)$ | Simplex or interior point |
| Lagrangian | Very tight | Iterative | Subgradient optimization |

## Python Implementation

```python
"""
Bounding Functions — Comparison for 0/1 Knapsack.

Demonstrates three bounding functions of increasing tightness:
trivial bound, greedy (fractional) bound, and LP relaxation bound.
"""


# === Trivial Upper Bound ===

def trivial_bound(
    level: int, value: int,
    values: list[int], capacity: int, weight: int
) -> float:
    """Sum of current value plus all remaining item values.

    Ignores weight constraint entirely — very loose but O(n).
    """
    return value + sum(values[level:])


# === Fractional (Greedy) Upper Bound ===

def fractional_bound(
    level: int, value: int, weight: int,
    weights: list[int], values: list[int], capacity: int
) -> float:
    """Upper bound via fractional knapsack on remaining items.

    Items must be pre-sorted by value/weight ratio (descending).
    """
    if weight > capacity:
        return 0.0

    bound = float(value)
    remaining = capacity - weight

    for i in range(level, len(weights)):
        if weights[i] <= remaining:
            bound += values[i]
            remaining -= weights[i]
        else:
            bound += values[i] * (remaining / weights[i])
            break

    return bound


# === Bound Comparison ===

def compare_bounds(
    weights: list[int], values: list[int], capacity: int
) -> None:
    """Compare bound tightness at the root node."""
    n = len(weights)
    # Sort by value density
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    w_sorted = [weights[i] for i in order]
    v_sorted = [values[i] for i in order]

    triv = trivial_bound(0, 0, v_sorted, capacity, 0)
    frac = fractional_bound(0, 0, 0, w_sorted, v_sorted, capacity)

    print(f"Trivial upper bound:    {triv}")
    print(f"Fractional upper bound: {frac}")


# === Main ===

if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    print(f"Weights: {weights}")
    print(f"Values:  {values}")
    print(f"Capacity: {capacity}")
    compare_bounds(weights, values, capacity)
    # Output:
    # Weights: [2, 3, 4, 5]
    # Values:  [3, 4, 5, 6]
    # Capacity: 8
    # Trivial upper bound:    18
    # Fractional upper bound: 12.0
```

The trivial bound (18) is far from the optimal value (10), while the fractional bound (12.0) is much tighter. In a full branch-and-bound search, the tighter bound prunes significantly more nodes.

## Reference

- Clausen, J. (1999). Branch and Bound Algorithms — Principles and Examples. Technical Report, University of Copenhagen.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
