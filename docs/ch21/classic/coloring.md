# Graph Coloring

Graph coloring assigns colors to vertices so that no two adjacent vertices share the same color. The **$m$-coloring problem** asks: can a given graph be colored with at most $m$ colors? This NP-complete problem is solvable via backtracking, which systematically tries color assignments and prunes branches that violate the adjacency constraint.

## Problem Definition

Given an undirected graph $G = (V, E)$ with $n = |V|$ vertices and a positive integer $m$, assign each vertex a color from $\{1, 2, \ldots, m\}$ such that for every edge $(u, v) \in E$, the colors of $u$ and $v$ differ.

The minimum $m$ for which a valid coloring exists is called the **chromatic number** $\chi(G)$.

## Backtracking Approach

Process vertices in order $v_1, v_2, \ldots, v_n$. At each vertex $v_i$, try each color $c \in \{1, \ldots, m\}$:

- **Feasibility check**: Is color $c$ different from the colors of all already-colored neighbors of $v_i$?
- If feasible, assign $c$ to $v_i$ and recurse to $v_{i+1}$.
- If no color is feasible, **backtrack** to $v_{i-1}$ and try the next color.

The search tree has branching factor $m$ and depth $n$, giving $O(m^n)$ worst-case time. Pruning via the feasibility check eliminates many branches in practice.

## Pruning Strategies

Several techniques reduce the search space:

1. **Forward checking**: When coloring vertex $v_i$, remove its color from the available colors of all uncolored neighbors. If any neighbor has zero available colors, prune immediately.
2. **Vertex ordering**: Color vertices with higher degree first (largest-first heuristic). Constrained vertices are harder to color, so addressing them early prunes more branches.
3. **Symmetry breaking**: Fix the color of the first vertex to 1 (any valid coloring can be relabeled).

## Complexity

| Aspect | Value |
|---|---|
| Time (worst case) | $O(m^n)$ |
| Space | $O(n)$ for color assignments |
| Decision problem | NP-complete for $m \ge 3$ |

!!! tip "Special Cases"
    - $m = 2$: Graph is 2-colorable if and only if it is bipartite, checkable in $O(V + E)$ via BFS/DFS.
    - Planar graphs: Always 4-colorable (Four Color Theorem), and 3-colorability is still NP-complete.

## Python Implementation

```python
"""
Graph Coloring — Backtracking with Feasibility Pruning.

Finds an m-coloring of a graph or reports that none exists.
Uses adjacency list representation and simple feasibility check.
"""


# === Feasibility Check ===

def is_safe(vertex: int, color: int, colors: list[int], adj: list[list[int]]) -> bool:
    """Check if assigning color to vertex violates no constraint."""
    for neighbor in adj[vertex]:
        if colors[neighbor] == color:
            return False
    return True


# === Backtracking Solver ===

def graph_coloring(adj: list[list[int]], m: int) -> list[int] | None:
    """Find an m-coloring of the graph, or return None.

    Args:
        adj: Adjacency list for n vertices (0-indexed).
        m: Number of available colors.

    Returns:
        List of color assignments (1-indexed colors), or None.
    """
    n = len(adj)
    colors = [0] * n

    def backtrack(vertex: int) -> bool:
        if vertex == n:
            return True

        for color in range(1, m + 1):
            if is_safe(vertex, color, colors, adj):
                colors[vertex] = color
                if backtrack(vertex + 1):
                    return True
                colors[vertex] = 0

        return False

    if backtrack(0):
        return colors
    return None


# === Count All Valid Colorings ===

def count_colorings(adj: list[list[int]], m: int) -> int:
    """Count the number of valid m-colorings."""
    n = len(adj)
    colors = [0] * n
    count = 0

    def backtrack(vertex: int) -> None:
        nonlocal count
        if vertex == n:
            count += 1
            return

        for color in range(1, m + 1):
            if is_safe(vertex, color, colors, adj):
                colors[vertex] = color
                backtrack(vertex + 1)
                colors[vertex] = 0

    backtrack(0)
    return count


# === Main ===

if __name__ == "__main__":
    # Example: a cycle of 4 vertices (C4)
    adj = [
        [1, 3],  # vertex 0
        [0, 2],  # vertex 1
        [1, 3],  # vertex 2
        [2, 0],  # vertex 3
    ]

    for m in range(2, 5):
        result = graph_coloring(adj, m)
        num = count_colorings(adj, m)
        status = result if result else "No valid coloring"
        print(f"m={m}: {status}, total valid colorings: {num}")
    # Output:
    # m=2: [1, 2, 1, 2], total valid colorings: 2
    # m=3: [1, 2, 1, 2], total valid colorings: 18
    # m=4: [1, 2, 1, 2], total valid colorings: 84
```

## Worked Example

Consider a triangle graph ($K_3$) with $m = 3$ colors:

- Vertex 0: try color 1. Feasible (no colored neighbors). Assign 1.
- Vertex 1: try color 1. Not feasible (neighbor 0 has color 1). Try color 2. Feasible. Assign 2.
- Vertex 2: try color 1. Not feasible (neighbor 0 has color 1). Try color 2. Not feasible (neighbor 1 has color 2). Try color 3. Feasible. Assign 3.

Valid coloring: $[1, 2, 3]$. The chromatic number of $K_3$ is $\chi(K_3) = 3$.

## Reference

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
