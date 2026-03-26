# Max-Flow Min-Cut Theorem

The max-flow min-cut theorem is one of the most important results in combinatorial optimization. It establishes a deep duality: the maximum amount of flow that can be pushed through a network equals the minimum capacity that must be removed to disconnect the source from the sink. This theorem provides both a correctness proof for flow algorithms and a practical method for finding minimum cuts.

## Cut Definition

Let $G = (V, E)$ be a flow network with source $s$ and sink $t$.

**$s$-$t$ cut.** A partition $(S, T)$ of $V$ with $s \in S$ and $t \in T$.

**Capacity of a cut.** The sum of capacities of edges crossing from $S$ to $T$:

$$
c(S, T) = \sum_{\substack{u \in S,\, v \in T \\ (u,v) \in E}} c(u, v)
$$

Note that only edges from $S$ to $T$ count. Edges from $T$ to $S$ do not contribute to the cut capacity.

**Net flow across a cut.** For a flow $f$, the net flow across cut $(S, T)$ is:

$$
f(S, T) = \sum_{\substack{u \in S,\, v \in T \\ (u,v) \in E}} f(u, v) - \sum_{\substack{v \in T,\, u \in S \\ (v,u) \in E}} f(v, u)
$$

## Weak Duality

!!! note "Weak Duality Lemma"
    For any flow $f$ and any $s$-$t$ cut $(S, T)$:

    $$
    |f| = f(S, T) \le c(S, T)
    $$

The first equality follows from flow conservation: the value of the flow equals the net flow across any cut. The inequality follows because each flow value is bounded by its edge capacity.

This immediately implies $\max_f |f| \le \min_{(S,T)} c(S, T)$.

## The Theorem

!!! note "Max-Flow Min-Cut Theorem (Ford and Fulkerson, 1956)"
    In a flow network, the following three conditions are equivalent:

    1. $f$ is a maximum flow.
    2. The residual graph $G_f$ contains no augmenting path from $s$ to $t$.
    3. There exists an $s$-$t$ cut $(S, T)$ such that $|f| = c(S, T)$.

??? example "Proof"
    **(1) $\Rightarrow$ (2):** If an augmenting path existed in $G_f$, we could increase the flow, contradicting maximality.

    **(2) $\Rightarrow$ (3):** Define $S = \{v \in V : v \text{ is reachable from } s \text{ in } G_f\}$ and $T = V \setminus S$. Since there is no augmenting path, $t \notin S$, so $(S, T)$ is a valid $s$-$t$ cut. For every edge $(u, v)$ with $u \in S$ and $v \in T$, the residual capacity must be zero (otherwise $v$ would be reachable), so $f(u, v) = c(u, v)$. For every edge $(v, u)$ with $v \in T$ and $u \in S$, we must have $f(v, u) = 0$ (otherwise the reverse edge would make $v$ reachable). Therefore $|f| = f(S, T) = c(S, T)$.

    **(3) $\Rightarrow$ (1):** By weak duality, $|f| \le c(S', T')$ for every cut $(S', T')$. If $|f| = c(S, T)$ for some cut, then $f$ achieves the upper bound and must be maximum. $\square$

## Finding the Minimum Cut

After computing a maximum flow $f^*$ using any max-flow algorithm:

1. Build the residual graph $G_{f^*}$.
2. Run BFS/DFS from $s$ in $G_{f^*}$ to find all reachable vertices $S$.
3. Set $T = V \setminus S$.
4. The edges from $S$ to $T$ in the original graph form the minimum cut.

## Implementation

```python
"""
Find the minimum s-t cut by computing max flow then extracting
reachable vertices in the residual graph.
"""

from collections import deque

# === Edmonds-Karp + Min-Cut Extraction ===

def min_cut(n: int, edges: list[tuple[int, int, int]],
            source: int, sink: int) -> tuple[int, set, set]:
    """Compute max flow and extract the minimum cut.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of (u, v, capacity) tuples.
        source: Source vertex.
        sink: Sink vertex.

    Returns:
        Tuple (max_flow_value, S, T) where (S, T) is the min cut.
    """
    graph = [[] for _ in range(n)]

    def add_edge(u: int, v: int, cap: int) -> None:
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])

    for u, v, cap in edges:
        add_edge(u, v, cap)

    # Edmonds-Karp max flow
    total_flow = 0
    while True:
        parent = [None] * n
        parent[source] = (source, -1)
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for i, (v, cap, _) in enumerate(graph[u]):
                if parent[v] is None and cap > 0:
                    parent[v] = (u, i)
                    if v == sink:
                        break
                    queue.append(v)
            else:
                continue
            break

        if parent[sink] is None:
            break

        bottleneck = float('inf')
        v = sink
        while v != source:
            u, idx = parent[v]
            bottleneck = min(bottleneck, graph[u][idx][1])
            v = u

        v = sink
        while v != source:
            u, idx = parent[v]
            graph[u][idx][1] -= bottleneck
            graph[v][graph[u][idx][2]][1] += bottleneck
            v = u

        total_flow += bottleneck

    # Extract min cut: BFS on residual graph from source
    visited = set()
    queue = deque([source])
    visited.add(source)
    while queue:
        u = queue.popleft()
        for v, cap, _ in graph[u]:
            if v not in visited and cap > 0:
                visited.add(v)
                queue.append(v)

    s_side = visited
    t_side = set(range(n)) - visited
    return total_flow, s_side, t_side


# === Demonstration ===

if __name__ == "__main__":
    edges = [
        (0, 1, 3),  # s -> a
        (0, 2, 2),  # s -> b
        (1, 2, 1),  # a -> b
        (1, 3, 2),  # a -> t
        (2, 3, 3),  # b -> t
    ]
    flow_val, S, T = min_cut(4, edges, 0, 3)
    print(f"Max flow = Min cut capacity = {flow_val}")
    print(f"S = {sorted(S)}")
    print(f"T = {sorted(T)}")

    # Show cut edges
    for u, v, cap in edges:
        if u in S and v in T:
            print(f"  Cut edge: ({u}, {v}), capacity {cap}")
```

**Output:**

```
Max flow = Min cut capacity = 5
S = [0]
T = [1, 2, 3]
Cut edges: (0, 1), capacity 3
Cut edges: (0, 2), capacity 2
```

The minimum cut separates the source from everything else, with total capacity $3 + 2 = 5$, confirming the max-flow min-cut theorem.

## Applications

- **Network reliability.** The min cut identifies the most vulnerable bottleneck in a network.
- **Image segmentation.** Pixel graphs with source/sink connections enable foreground-background separation via min cut.
- **Connectivity.** The minimum number of edges to disconnect two vertices equals the maximum number of edge-disjoint paths between them (Menger's theorem).

## Reference

- Ford, L. R., & Fulkerson, D. R. (1956). Maximal flow through a network. *Canadian Journal of Mathematics*, 8, 399--404.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 26: Maximum Flow.
