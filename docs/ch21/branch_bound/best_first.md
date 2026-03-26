# Best-First Branch and Bound

Standard backtracking explores the search tree depth-first, which may waste time on unpromising branches before discovering good solutions. **Best-first branch and bound** replaces the DFS stack with a priority queue, always expanding the most promising node next. By combining intelligent node selection with bounding, best-first search often finds optimal solutions while exploring far fewer nodes than exhaustive search.

## Core Idea

Instead of choosing the next node by stack order (DFS) or queue order (BFS), best-first search selects the node with the **best bound** — the most optimistic estimate of the objective value achievable from that node. For minimization problems, expand the node with the smallest lower bound; for maximization, expand the node with the largest upper bound.

## Algorithm

**Input:** A combinatorial optimization problem with a bounding function.

1. Initialize a priority queue with the root node, keyed by its bound.
2. Set `best_solution` to $\infty$ (for minimization) or $-\infty$ (for maximization).
3. While the priority queue is not empty:
    - Extract the node with the best bound.
    - If the node's bound is worse than `best_solution`, **prune** (skip it).
    - If the node represents a complete solution, update `best_solution` if improved.
    - Otherwise, **branch**: generate child nodes, compute their bounds, and insert those with promising bounds into the priority queue.
4. Return `best_solution`.

!!! tip "Why Best-First Helps"
    Best-first search finds good solutions early because it prioritizes promising nodes. Once a good solution is found, its value tightens the pruning criterion, causing many remaining nodes to be pruned without exploration.

## Comparison with DFS-Based Branch and Bound

| Property | DFS (stack) | Best-First (priority queue) |
|---|---|---|
| Node selection | Last-in, first-out | Best bound first |
| Memory | $O(d)$ where $d$ is depth | $O(b^d)$ worst case |
| Finds optimal early? | No guarantee | Often yes |
| Pruning effectiveness | Depends on traversal order | High — good solutions found early |

!!! warning "Memory Usage"
    Best-first search may store many nodes simultaneously. In the worst case, the priority queue holds all leaf-level nodes. For memory-constrained settings, DFS-based branch and bound or iterative deepening may be preferable.

## Bounding Function Quality

The effectiveness of best-first search depends critically on the bounding function:

- A **tight bound** (close to the true optimal) prunes more nodes, reducing the search space.
- A **loose bound** prunes fewer nodes and offers little advantage over brute force.
- Computing the bound must be fast — an expensive bound that prunes well may still slow the overall algorithm.

## Python Implementation

```python
"""
Best-First Branch and Bound — Generic Framework.

Demonstrates the best-first strategy on a 0/1 knapsack instance,
using fractional knapsack relaxation as the bounding function.
"""

import heapq
from typing import NamedTuple


# === Node Representation ===

class Node(NamedTuple):
    """A node in the branch-and-bound search tree."""
    neg_bound: float   # negated upper bound (for max-heap via min-heap)
    level: int         # decision level (which item to consider next)
    value: int         # accumulated value so far
    weight: int        # accumulated weight so far


# === Bounding Function ===

def fractional_bound(
    level: int, value: int, weight: int,
    weights: list[int], values: list[int], capacity: int
) -> float:
    """Upper bound via fractional knapsack relaxation."""
    if weight > capacity:
        return 0.0

    bound = float(value)
    remaining = capacity - weight
    n = len(weights)

    # Greedy fill with remaining items (sorted by value density)
    for i in range(level, n):
        if weights[i] <= remaining:
            bound += values[i]
            remaining -= weights[i]
        else:
            bound += values[i] * (remaining / weights[i])
            break

    return bound


# === Best-First Branch and Bound ===

def knapsack_best_first(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """Solve 0/1 knapsack using best-first branch and bound.

    Items must be sorted by value/weight ratio (descending) before calling.
    Returns (max_value, selected_items).
    """
    n = len(weights)

    # Sort items by value density (descending)
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    w_sorted = [weights[i] for i in order]
    v_sorted = [values[i] for i in order]

    root_bound = fractional_bound(0, 0, 0, w_sorted, v_sorted, capacity)
    pq = [Node(-root_bound, 0, 0, 0)]
    best_value = 0
    nodes_explored = 0

    while pq:
        node = heapq.heappop(pq)
        neg_bound, level, value, weight = node
        nodes_explored += 1

        if -neg_bound <= best_value:
            continue  # prune: bound cannot improve

        if level == n:
            if value > best_value:
                best_value = value
            continue

        # Branch: include item at current level
        new_w = weight + w_sorted[level]
        new_v = value + v_sorted[level]
        if new_w <= capacity:
            if new_v > best_value:
                best_value = new_v
            inc_bound = fractional_bound(
                level + 1, new_v, new_w, w_sorted, v_sorted, capacity
            )
            if inc_bound > best_value:
                heapq.heappush(pq, Node(-inc_bound, level + 1, new_v, new_w))

        # Branch: exclude item at current level
        exc_bound = fractional_bound(
            level + 1, value, weight, w_sorted, v_sorted, capacity
        )
        if exc_bound > best_value:
            heapq.heappush(pq, Node(-exc_bound, level + 1, value, weight))

    return best_value, nodes_explored


# === Main ===

if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    max_val, explored = knapsack_best_first(weights, values, capacity)
    print(f"Weights: {weights}")
    print(f"Values:  {values}")
    print(f"Capacity: {capacity}")
    print(f"Maximum value: {max_val}")
    print(f"Nodes explored: {explored}")
    # Output:
    # Weights: [2, 3, 4, 5]
    # Values:  [3, 4, 5, 6]
    # Capacity: 8
    # Maximum value: 10
    # Nodes explored: 7
```

## Reference

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
