# Iterative Deepening

BFS finds shortest paths but stores an entire frontier in memory, which can be $O(b^d)$ where $b$ is the branching factor and $d$ is the solution depth. DFS uses only $O(d)$ memory but may dive into arbitrarily deep branches and miss shallow solutions. **Iterative deepening depth-first search (IDDFS)** combines the best of both: it runs depth-limited DFS repeatedly, increasing the depth limit by one each round, until the target is found. This yields the optimality of BFS with the memory efficiency of DFS.

## Depth-Limited DFS

The building block of IDDFS is a DFS that stops exploring once it reaches a specified depth limit $\ell$. At depth $\ell$, the search returns without expanding further. This prevents the algorithm from descending into arbitrarily deep branches.

## The IDDFS Algorithm

IDDFS calls depth-limited DFS for limits $\ell = 0, 1, 2, \ldots$ until either the target is found or the entire graph has been explored. On iteration $\ell$, all vertices at depth $\leq \ell$ are visited. Since the first iteration that reaches the target uses the smallest possible depth, IDDFS finds an optimal (shortest) path in unweighted graphs.

## Time Complexity

At first glance, the repeated work seems wasteful. However, the total number of node expansions is bounded. If the solution is at depth $d$ and every vertex has at most $b$ children, then iteration $\ell$ visits at most $O(b^\ell)$ vertices. The total work across all iterations is

$$
\sum_{\ell=0}^{d} O(b^\ell) = O(b^d)
$$

because the geometric series is dominated by its last term. This matches BFS in time while using only $O(d)$ memory for the recursion stack.

!!! tip "The overhead is small"
    The ratio of total work to the work of the final iteration is $\frac{b^{d+1} - 1}{(b-1) \cdot b^d} \approx \frac{b}{b-1}$, which is at most 2 for $b \geq 2$. The repeated work at most doubles the total computation.

## Space Complexity

IDDFS stores only the current path from the root to the deepest vertex on the recursion stack. This requires $O(d)$ memory, the same as DFS and dramatically less than the $O(b^d)$ that BFS may need.

## Properties

- **Complete:** IDDFS is complete on finite graphs (it will find a solution if one exists).
- **Optimal:** For unweighted graphs, IDDFS finds a shortest path because it explores all vertices at depth $d$ before any at depth $d + 1$.
- **Time:** $O(b^d)$ where $d$ is the depth of the shallowest solution.
- **Space:** $O(d)$ -- the depth of the shallowest solution.

## Implementation

```python
"""
Iterative Deepening Depth-First Search (IDDFS).

Combines BFS optimality with DFS memory efficiency by running
depth-limited DFS with increasing depth limits.
"""

# === Depth-limited DFS ========================================================

def depth_limited_dfs(graph, source, target, limit):
    """DFS that stops at the given depth limit.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list.
    source : int
        Starting vertex.
    target : int
        Goal vertex.
    limit : int
        Maximum depth to explore.

    Returns
    -------
    list[int] | None
        Path from source to target, or None if not found within limit.
    """
    if source == target:
        return [source]
    if limit <= 0:
        return None

    for neighbor in graph[source]:
        result = depth_limited_dfs(graph, neighbor, target, limit - 1)
        if result is not None:
            return [source] + result
    return None


# === IDDFS ====================================================================

def iddfs(graph, source, target, max_depth=100):
    """Iterative deepening depth-first search.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list.
    source : int
        Starting vertex.
    target : int
        Goal vertex.
    max_depth : int
        Upper bound on depth to prevent infinite loops.

    Returns
    -------
    list[int] | None
        Shortest path from source to target, or None if unreachable.
    """
    for depth_limit in range(max_depth + 1):
        result = depth_limited_dfs(graph, source, target, depth_limit)
        if result is not None:
            return result
    return None


# === Main =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [],
        4: [7],
        5: [],
        6: [],
        7: [],
    }

    for target in [0, 4, 7, 6]:
        path = iddfs(graph, 0, target)
        depth = len(path) - 1 if path else None
        print(f"IDDFS 0 -> {target}: path={path}, depth={depth}")
```

**Output:**
```
IDDFS 0 -> 0: path=[0], depth=0
IDDFS 0 -> 4: path=[0, 1, 4], depth=2
IDDFS 0 -> 7: path=[0, 1, 4, 7], depth=3
IDDFS 0 -> 6: path=[0, 2, 6], depth=2
```

Each path has the minimum possible depth, confirming that IDDFS finds optimal solutions in unweighted graphs.

## When to Use IDDFS

| Scenario | Preferred Algorithm |
|---|---|
| Unweighted, target at unknown depth, memory limited | IDDFS |
| Unweighted, memory is not a concern | BFS |
| Weighted edges | Dijkstra or A* |
| Very deep solutions with narrow branching | DFS may suffice |

IDDFS is the standard choice when BFS would exhaust memory but optimality is still required. It is commonly used in game-tree search (e.g., chess engines) where the branching factor is large and memory is the bottleneck.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
- Korf, R. E. (1985). Depth-first iterative-deepening: An optimal admissible tree search. *Artificial Intelligence*, 27(1), 97-109.
