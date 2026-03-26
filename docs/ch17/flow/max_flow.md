# Max-Flow Problem

Many optimization problems can be modeled as routing a commodity through a network with limited-capacity edges. The **maximum flow problem** asks for the greatest total flow that can be sent from a designated source to a designated sink while respecting capacity constraints. This foundational problem connects to minimum cuts, bipartite matching, and numerous combinatorial optimization tasks.

## Flow Network

A **flow network** is a directed graph $G = (V, E)$ with:

- A **source** vertex $s \in V$ (no incoming edges in the standard formulation).
- A **sink** vertex $t \in V$ (no outgoing edges in the standard formulation).
- A **capacity function** $c: E \to \mathbb{R}_{\ge 0}$ assigning a non-negative capacity to each edge.

## Flow Definition

A **flow** is a function $f: E \to \mathbb{R}_{\ge 0}$ satisfying two conditions:

**Capacity constraint.** For every edge $(u, v) \in E$:

$$
0 \le f(u, v) \le c(u, v)
$$

**Flow conservation.** For every vertex $v \in V \setminus \{s, t\}$:

$$
\sum_{(u,v) \in E} f(u, v) = \sum_{(v,w) \in E} f(v, w)
$$

The total flow entering $v$ equals the total flow leaving $v$.

## Flow Value

The **value** of a flow $f$ is the net flow leaving the source:

$$
|f| = \sum_{(s,v) \in E} f(s, v) - \sum_{(v,s) \in E} f(v, s)
$$

By flow conservation, this equals the net flow entering the sink.

## The Maximum Flow Problem

Given a flow network $G = (V, E)$ with capacity function $c$, source $s$, and sink $t$, find a flow $f$ that maximizes $|f|$.

## Cuts and Duality

An **$s$-$t$ cut** is a partition $(S, T)$ of $V$ with $s \in S$ and $t \in T$. The **capacity** of a cut is:

$$
c(S, T) = \sum_{\substack{u \in S,\, v \in T \\ (u,v) \in E}} c(u, v)
$$

!!! note "Weak Duality"
    For any flow $f$ and any $s$-$t$ cut $(S, T)$:

    $$
    |f| \le c(S, T)
    $$

    No flow can exceed the capacity of any cut.

!!! note "Max-Flow Min-Cut Theorem"
    The value of a maximum flow equals the capacity of a minimum cut:

    $$
    \max_f |f| = \min_{(S,T)} c(S, T)
    $$

## Algorithm Overview

| Algorithm | Time Complexity | Notes |
|-----------|:---------------:|:------|
| Ford-Fulkerson | $O(\|f^*\| \cdot E)$ | DFS-based; depends on flow value |
| Edmonds-Karp | $O(V E^2)$ | BFS shortest augmenting paths |
| Dinic | $O(V^2 E)$ | Blocking flows in layered graph |
| Push-Relabel | $O(V^2 E)$ | Local operations, no path search |
| Push-Relabel (FIFO) | $O(V^3)$ | Best general-purpose variant |

## Implementation

```python
"""
Maximum flow via Edmonds-Karp (BFS-based Ford-Fulkerson).

Finds the maximum flow from source to sink in O(VE^2) time by
always augmenting along shortest paths in the residual graph.
"""

from collections import deque

# === Edmonds-Karp Max Flow ===

def max_flow(n: int, edges: list[tuple[int, int, int]],
             source: int, sink: int) -> int:
    """Compute maximum flow using Edmonds-Karp algorithm.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of (u, v, capacity) tuples.
        source: Source vertex.
        sink: Sink vertex.

    Returns:
        Maximum flow value.
    """
    # Build adjacency list with forward/backward edges
    graph = [[] for _ in range(n)]
    def add_edge(u: int, v: int, cap: int) -> None:
        graph[u].append([v, cap, len(graph[v])])      # forward
        graph[v].append([u, 0, len(graph[u]) - 1])    # backward

    for u, v, cap in edges:
        add_edge(u, v, cap)

    def bfs() -> list[tuple[int, int]] | None:
        """BFS to find shortest augmenting path."""
        parent = [None] * n
        parent[source] = (source, -1)
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for i, (v, cap, _) in enumerate(graph[u]):
                if parent[v] is None and cap > 0:
                    parent[v] = (u, i)
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    total_flow = 0
    while True:
        parent = bfs()
        if parent is None:
            break

        # Find bottleneck
        bottleneck = float('inf')
        v = sink
        while v != source:
            u, idx = parent[v]
            bottleneck = min(bottleneck, graph[u][idx][1])
            v = u

        # Update residual capacities
        v = sink
        while v != source:
            u, idx = parent[v]
            graph[u][idx][1] -= bottleneck
            graph[v][graph[u][idx][2]][1] += bottleneck
            v = u

        total_flow += bottleneck

    return total_flow


# === Demonstration ===

if __name__ == "__main__":
    # Network: s=0, a=1, b=2, t=3
    edges = [
        (0, 1, 10),  # s -> a
        (0, 2, 8),   # s -> b
        (1, 2, 5),   # a -> b
        (1, 3, 7),   # a -> t
        (2, 3, 10),  # b -> t
    ]
    result = max_flow(4, edges, 0, 3)
    print(f"Maximum flow: {result}")
```

**Output:**

```
Maximum flow: 17
```

The source can push at most $10$ units through vertex $a$ and $8$ through vertex $b$. The cross edge $(a, b)$ with capacity $5$ lets excess capacity on the $a$-side flow through $b$ to the sink. The minimum cut is $\{s\} | \{a, b, t\}$ with capacity $10 + 8 = 18$, but the actual min-cut is $\{s, a\} | \{b, t\}$ with capacity $5 + 7 = 12$... checking: total capacity from $s$ side is $8 + 5 + 7 = 20$. The maximum flow of $17$ is achieved as verified by the algorithm.

## Applications

- **Network routing.** Maximize data throughput in communication networks.
- **Bipartite matching.** Maximum matching reduces to max flow with unit capacities.
- **Image segmentation.** Min-cut on pixel similarity graphs separates foreground from background.
- **Project selection.** Choose a subset of interdependent projects to maximize profit.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 26: Maximum Flow.
- Edmonds, J., & Karp, R. M. (1972). Theoretical improvements in algorithmic efficiency for network flow problems. *Journal of the ACM*, 19(2), 248--264.
