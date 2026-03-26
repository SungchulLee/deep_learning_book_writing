# 0/1 Knapsack via Branch and Bound

The 0/1 knapsack problem can be solved by dynamic programming in $O(nW)$ time, but when $W$ is very large, this pseudo-polynomial approach becomes impractical. Branch and bound offers an alternative: it explores a binary decision tree (include or exclude each item) while using bounding functions to prune subtrees that cannot lead to better solutions. In practice, branch and bound with a good bound often solves knapsack instances much faster than exhaustive search.

## Search Tree Structure

Each node at level $i$ represents a partial decision about items $1, \ldots, i$. The two children correspond to:

- **Include** item $i+1$: add its weight and value to the running totals.
- **Exclude** item $i+1$: move to the next item without changes.

A complete solution is reached at level $n$ (all items decided). The tree has $2^n$ leaves in the worst case.

## Bounding with Fractional Relaxation

Sort items by value-to-weight ratio $r_i = v_i / w_i$ in decreasing order. At any node with accumulated value $V$ and remaining capacity $C$, compute an upper bound by greedily filling the remaining capacity with fractional items:

$$
\text{UB}(V, C, \text{level}) = V + \sum_{i=\text{level}}^{k-1} v_i + v_k \cdot \frac{C - \sum_{i=\text{level}}^{k-1} w_i}{w_k}
$$

where $k$ is the first item that does not fit entirely. This fractional relaxation upper bound is always at least as large as the best integer solution in the subtree.

## Algorithm

1. Sort items by $v_i / w_i$ (descending).
2. Initialize `best_value = 0`.
3. Use DFS (or best-first) to explore the binary tree.
4. At each node, compute the upper bound.
5. **Prune** if the upper bound $\le$ `best_value`.
6. At leaf nodes, update `best_value` if the current solution is better.

## Worked Example

**Items** (sorted by ratio): $(w, v) = \{(2, 12), (4, 20), (6, 18), (9, 18)\}$, capacity $W = 15$.

Ratios: $r = [6, 5, 3, 2]$.

- **Root**: UB $= 12 + 20 + 18 \cdot (9/6) = 12 + 20 + 27 = 59$ (fractional). Explore.
- **Include item 1** ($V=12, C=13$): UB = $12 + 20 + 18 \cdot (3/6) = 41$. Explore.
- **Include item 2** ($V=32, C=9$): UB = $32 + 18 \cdot (9/6) = 59$. Actually UB $= 32 + 18 = 50$ (item 3 fits). Continue.
- ... Eventually finds optimal value.

## Python Implementation

```python
"""
0/1 Knapsack — Branch and Bound with DFS.

Uses fractional knapsack relaxation as the upper bound to prune
branches that cannot improve upon the current best solution.
"""


# === Upper Bound via Fractional Relaxation ===

def upper_bound(
    level: int, value: int, weight: int,
    weights: list[int], values: list[int], capacity: int
) -> float:
    """Fractional knapsack upper bound from the current node."""
    if weight > capacity:
        return 0.0

    bound = float(value)
    remaining = capacity - weight
    n = len(weights)

    for i in range(level, n):
        if weights[i] <= remaining:
            bound += values[i]
            remaining -= weights[i]
        else:
            bound += values[i] * (remaining / weights[i])
            break

    return bound


# === Branch and Bound (DFS) ===

def knapsack_branch_bound(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """Solve 0/1 knapsack using DFS branch and bound.

    Returns (max_value, selected_indices_in_original_order).
    """
    n = len(weights)
    # Sort by value/weight ratio descending
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    w = [weights[i] for i in order]
    v = [values[i] for i in order]

    best_value = 0
    best_selection = [0] * n
    current = [0] * n
    nodes_explored = 0

    def dfs(level: int, curr_value: int, curr_weight: int) -> None:
        nonlocal best_value, best_selection, nodes_explored
        nodes_explored += 1

        if level == n:
            if curr_value > best_value:
                best_value = curr_value
                best_selection = current[:]
            return

        # Try including item at this level
        if curr_weight + w[level] <= capacity:
            current[level] = 1
            ub = upper_bound(
                level + 1, curr_value + v[level],
                curr_weight + w[level], w, v, capacity
            )
            if ub > best_value:
                dfs(level + 1, curr_value + v[level], curr_weight + w[level])
            current[level] = 0

        # Try excluding item at this level
        ub = upper_bound(level + 1, curr_value, curr_weight, w, v, capacity)
        if ub > best_value:
            dfs(level + 1, curr_value, curr_weight)

    dfs(0, 0, 0)

    # Map selected items back to original indices
    selected = [order[i] for i in range(n) if best_selection[i]]
    return best_value, sorted(selected)


# === Main ===

if __name__ == "__main__":
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50

    max_val, selected = knapsack_branch_bound(weights, values, capacity)
    print(f"Weights:  {weights}")
    print(f"Values:   {values}")
    print(f"Capacity: {capacity}")
    print(f"Max value: {max_val}")
    print(f"Selected items: {selected}")
    # Output:
    # Weights:  [10, 20, 30]
    # Values:   [60, 100, 120]
    # Capacity: 50
    # Max value: 220
    # Selected items: [1, 2]
```

## Comparison with Dynamic Programming

| Aspect | DP | Branch and Bound |
|---|---|---|
| Time complexity | $O(nW)$ pseudo-polynomial | Exponential worst case |
| When fast | Small to moderate $W$ | Tight bounds, few items |
| Space | $O(nW)$ or $O(W)$ | $O(n)$ stack + bound computation |
| Large $W$ | Impractical | Often practical with good bounds |

!!! warning "Worst-Case Complexity"
    Branch and bound has $O(2^n)$ worst-case time. Its practical advantage comes from pruning, which depends on the problem instance and bound quality. For some instances, it explores nearly the entire tree.

## Reference

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.
- Horowitz, E., & Sahni, S. (1974). Computing partitions with applications to the knapsack problem. *Journal of the ACM*, 21(2), 277-292.
