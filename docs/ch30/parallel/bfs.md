# Parallel BFS

Breadth-first search (BFS) explores a graph level by level, visiting all vertices at distance $d$ before those at distance $d + 1$. In sequential execution, BFS runs in $O(V + E)$ time, but for large graphs (social networks, web graphs) this becomes a bottleneck. **Parallel BFS** exploits the fact that all vertices in the same frontier can be processed simultaneously, reducing the running time to $O(D)$ levels where $D$ is the graph diameter.

## Level-Synchronous Parallel BFS

The key insight behind parallel BFS is that the algorithm naturally decomposes into independent levels. At each level, every vertex in the current frontier can explore its neighbors independently. This leads to the **level-synchronous** approach:

1. Start with the source vertex as the initial frontier $F_0 = \{s\}$.
2. At level $d$, process all vertices in frontier $F_d$ in parallel.
3. Each processor examines the neighbors of its assigned vertices.
4. Unvisited neighbors form the next frontier $F_{d+1}$.
5. A barrier synchronization separates consecutive levels.

### Work-Span Analysis

Let $D$ denote the diameter (the maximum shortest-path distance from the source). The parallel BFS has:

- **Work**: $T_1 = O(V + E)$, since every vertex and edge is examined exactly once across all levels.
- **Span**: $T_\infty = O(D \log V)$, since there are $D$ levels, and within each level a parallel prefix sum or duplicate removal may cost $O(\log V)$.
- **Parallelism**: $P = O((V + E) / (D \log V))$.

For graphs with small diameter (e.g., $D = O(\log V)$), parallel BFS achieves high parallelism.

## Algorithm

The following pseudocode captures the level-synchronous structure:

$$
\textbf{Parallel-BFS}(G, s):
$$

1. Initialize $\text{dist}[v] \leftarrow \infty$ for all $v$; set $\text{dist}[s] \leftarrow 0$.
2. Set frontier $F \leftarrow \{s\}$.
3. While $F \neq \emptyset$:
    - **parallel for** each $u \in F$: examine all neighbors $v$ of $u$.
    - If $\text{dist}[v] = \infty$, set $\text{dist}[v] \leftarrow \text{dist}[u] + 1$ and add $v$ to $F'$.
    - Barrier synchronization; set $F \leftarrow F'$.

!!! warning "Race conditions"
    Multiple processors may try to update $\text{dist}[v]$ simultaneously. In practice, an atomic compare-and-swap (CAS) ensures that only one processor claims each unvisited vertex. This does not affect correctness since all potential parents are at the same distance.

## Implementation

```python
"""
Level-synchronous parallel BFS simulation.

Simulates the level-synchronous approach by processing each
frontier as a batch. In a true parallel system, the inner loop
over frontier vertices runs on separate processors.
"""

from collections import defaultdict

# ===================================================================
# Level-Synchronous BFS
# ===================================================================

def parallel_bfs(adj, source):
    """Simulate level-synchronous parallel BFS.

    Args:
        adj: adjacency list as dict of lists
        source: starting vertex

    Returns:
        dist: dict mapping each reachable vertex to its distance
        levels: list of frontiers (one per BFS level)
    """
    dist = {source: 0}
    frontier = [source]
    levels = [list(frontier)]

    while frontier:
        next_frontier = []
        # In a parallel system, this loop runs concurrently
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    next_frontier.append(v)
        frontier = next_frontier
        if frontier:
            levels.append(list(frontier))

    return dist, levels

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    adj = defaultdict(list)
    edges = [(0, 1), (0, 2), (1, 3), (1, 4),
             (2, 5), (3, 6), (4, 6), (5, 7)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist, levels = parallel_bfs(adj, source=0)

    print("Level-synchronous BFS from vertex 0:")
    for i, level in enumerate(levels):
        print(f"  Level {i}: {sorted(level)}")
    print()
    print("Distances:", {v: dist[v] for v in sorted(dist)})
    print(f"Diameter (from source): {max(dist.values())}")
    print(f"Levels processed: {len(levels)}")
```

**Output:**
```
Level-synchronous BFS from vertex 0:
  Level 0: [0]
  Level 1: [1, 2]
  Level 2: [3, 4, 5]
  Level 3: [6, 7]

Distances: {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3}
Diameter (from source): 3
Levels processed: 4
```

## Complexity Summary

| Metric | Value |
|---|---|
| Work $T_1$ | $O(V + E)$ |
| Span $T_\infty$ | $O(D \log V)$ |
| Parallelism | $O\!\left(\frac{V + E}{D \log V}\right)$ |
| Space | $O(V)$ |

Here $D$ is the diameter of the graph (or the maximum BFS depth from the source).

## Practical Considerations

- **Frontier size variation**: Early and late levels have small frontiers (low parallelism), while middle levels often have large frontiers (high parallelism). This uneven workload motivates hybrid approaches.
- **Direction-optimizing BFS**: For high-diameter graphs, switching between a top-down scan (expanding from frontier) and a bottom-up scan (checking unvisited vertices against the frontier) can reduce edge traversals by an order of magnitude.
- **Memory bandwidth**: On shared-memory systems, parallel BFS is often memory-bandwidth-bound rather than compute-bound, since each vertex access may cause a cache miss.

## Reference

- Leiserson, C. E. and Schardl, T. B. (2010). "A work-efficient parallel breadth-first search algorithm." *SPAA*.
- Beamer, S. et al. (2012). "Direction-optimizing breadth-first search." *SC*.
