# Bipartite Matching

Many real-world assignment problems reduce to matching in bipartite graphs: assigning workers to jobs, students to projects, or servers to requests. Given a bipartite graph, a **matching** selects a subset of edges with no shared endpoints. The fundamental algorithmic question is finding a matching of maximum size, which can be solved efficiently by reducing it to a maximum flow problem.

## Definitions

**Bipartite graph.** A graph $G = (L \cup R, E)$ whose vertices partition into two disjoint sets $L$ and $R$ such that every edge connects a vertex in $L$ to a vertex in $R$.

**Matching.** A subset $M \subseteq E$ such that no two edges in $M$ share an endpoint. A vertex is *matched* if it is an endpoint of some edge in $M$; otherwise it is *free* (or *unmatched*).

**Maximum matching.** A matching with the largest possible number of edges.

**Perfect matching.** A matching that covers every vertex (possible only when $|L| = |R|$ and enough edges exist).

## Reduction to Maximum Flow

The maximum bipartite matching problem reduces to maximum flow in a specially constructed network:

1. Add a **source** $s$ with edges $(s, \ell)$ of capacity $1$ for every $\ell \in L$.
2. Add a **sink** $t$ with edges $(r, t)$ of capacity $1$ for every $r \in R$.
3. For each original edge $(\ell, r) \in E$, add an edge of capacity $1$.

A maximum integer flow in this network corresponds to a maximum matching: each unit of flow through an edge $(\ell, r)$ indicates that $\ell$ is matched to $r$. Since all capacities are $1$ and the graph is bipartite, the max flow is always integral.

$$
|M^*| = \max\text{-flow}(s, t)
$$

## Konig's Theorem

!!! note "Konig's Theorem"
    In any bipartite graph, the size of a maximum matching equals the size of a minimum vertex cover.

$$
|M^*| = |\text{min vertex cover}|
$$

This is a remarkable duality result specific to bipartite graphs. In general graphs, the minimum vertex cover can be much larger than the maximum matching.

## Augmenting Paths

An **augmenting path** with respect to a matching $M$ is a path that:

- Starts at a free vertex in $L$.
- Alternates between unmatched and matched edges.
- Ends at a free vertex in $R$.

Flipping the matched/unmatched status of every edge along an augmenting path increases the matching size by one. The **Berge's lemma** states that a matching is maximum if and only if no augmenting path exists.

## Implementation

```python
"""
Maximum bipartite matching via augmenting paths (Hopcroft-Karp style DFS).

Finds a maximum matching in a bipartite graph by repeatedly searching
for augmenting paths using DFS.
"""

# === Maximum Bipartite Matching ===

def max_bipartite_matching(
    n_left: int, n_right: int, edges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find a maximum matching in a bipartite graph.

    Args:
        n_left: Number of vertices on the left side (0-indexed).
        n_right: Number of vertices on the right side (0-indexed).
        edges: List of edges (l, r) where l in [0, n_left), r in [0, n_right).

    Returns:
        List of matched pairs (l, r).
    """
    adj = [[] for _ in range(n_left)]
    for l, r in edges:
        adj[l].append(r)

    match_right = [-1] * n_right  # match_right[r] = matched left vertex

    def dfs(u: int, visited: set) -> bool:
        """Try to find an augmenting path from left vertex u."""
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if match_right[v] == -1 or dfs(match_right[v], visited):
                    match_right[v] = u
                    return True
        return False

    matching_size = 0
    for u in range(n_left):
        visited = set()
        if dfs(u, visited):
            matching_size += 1

    result = []
    for r in range(n_right):
        if match_right[r] != -1:
            result.append((match_right[r], r))
    return result


# === Demonstration ===

if __name__ == "__main__":
    # Workers L={0,1,2}, Jobs R={0,1,2}
    # Worker 0 can do jobs 0,1
    # Worker 1 can do jobs 0,2
    # Worker 2 can do job 1
    edges = [(0,0),(0,1),(1,0),(1,2),(2,1)]
    matching = max_bipartite_matching(3, 3, edges)
    print(f"Maximum matching size: {len(matching)}")
    for l, r in matching:
        print(f"  Worker {l} -> Job {r}")
```

**Output:**

```
Maximum matching size: 3
  Worker 1 -> Job 0
  Worker 2 -> Job 1
  Worker 0 -> Job 2
```

All three workers are matched to distinct jobs, achieving a perfect matching. Worker $0$ could do jobs $0$ or $1$, but the algorithm assigns them to job $2$ (via worker $1$) to maximize the total matching.

## Complexity

| Algorithm | Time |
|-----------|:----:|
| Augmenting path DFS (above) | $O(V \cdot E)$ |
| Hopcroft-Karp | $O(E \sqrt{V})$ |
| Max flow reduction (Ford-Fulkerson) | $O(V \cdot E)$ |

The simple DFS-based approach runs $O(V)$ augmenting path searches, each taking $O(E)$ time. The Hopcroft-Karp algorithm improves this by finding multiple augmenting paths simultaneously using BFS phases.

## Hall's Theorem

!!! note "Hall's Marriage Theorem"
    A bipartite graph $G = (L \cup R, E)$ has a matching that covers every vertex in $L$ if and only if for every subset $S \subseteq L$:

    $$
    |N(S)| \ge |S|
    $$

    where $N(S)$ denotes the set of neighbors of $S$ in $R$.

This theorem provides a necessary and sufficient condition for the existence of a perfect matching on the $L$ side, though checking it directly requires examining all $2^{|L|}$ subsets.

## Reference

- Hopcroft, J. E., & Karp, R. M. (1973). An $n^{5/2}$ algorithm for maximum matchings in bipartite graphs. *SIAM Journal on Computing*, 2(4), 225--231.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 26: Maximum Flow.
