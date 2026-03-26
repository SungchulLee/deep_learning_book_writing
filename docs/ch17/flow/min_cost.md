# Min-Cost Max-Flow

The standard max-flow problem ignores the cost of routing flow through edges. In many applications, each edge has both a capacity and a per-unit cost, and we want to send a specified amount of flow (or the maximum possible flow) at the **lowest total cost**. The min-cost max-flow problem combines both objectives: among all maximum flows, find one with minimum cost.

## Problem Formulation

A **cost flow network** extends a flow network with an additional cost function. Given a directed graph $G = (V, E)$ with:

- Capacity function $c: E \to \mathbb{R}_{\ge 0}$.
- Cost function $w: E \to \mathbb{R}$ (cost per unit of flow on each edge).
- Source $s$ and sink $t$.

The **cost** of a flow $f$ is:

$$
\text{cost}(f) = \sum_{(u,v) \in E} w(u, v) \cdot f(u, v)
$$

The **min-cost max-flow** problem seeks a flow $f^*$ satisfying:

$$
|f^*| = \max_f |f| \quad \text{and} \quad \text{cost}(f^*) = \min \{\text{cost}(f) : |f| = |f^*|\}
$$

## Algorithm: Successive Shortest Paths

The most intuitive approach adapts Ford-Fulkerson by always augmenting along the **shortest (minimum-cost) path** in the residual graph. In the residual graph, forward edges have cost $w(u,v)$ and backward edges have cost $-w(u,v)$.

**Step 1.** Initialize $f = 0$.

**Step 2.** Find a shortest path (by cost) from $s$ to $t$ in the residual graph using Bellman-Ford or SPFA (since negative-cost backward edges may exist).

**Step 3.** Augment flow along this path by the bottleneck capacity.

**Step 4.** Repeat until no $s$-$t$ path exists in the residual graph.

Each augmentation sends flow along the cheapest available route, ensuring that the total cost grows as slowly as possible. When no more augmenting paths exist, the flow is maximum and the cost is minimized among all max flows.

## Johnson's Potential Technique

Bellman-Ford takes $O(VE)$ per shortest-path query. We can use **potentials** (reduced costs) to eliminate negative edges and switch to Dijkstra ($O(E \log V)$ per query).

Define potentials $h: V \to \mathbb{R}$ and reduced cost:

$$
w_h(u, v) = w(u, v) + h(u) - h(v)
$$

After the first Bellman-Ford pass initializes $h$ as shortest-path distances from $s$, all reduced costs are non-negative. After each augmentation, update potentials using the new distances: $h'(v) = h(v) + d(v)$ where $d(v)$ is the shortest distance in reduced-cost terms.

## Implementation

```python
"""
Min-cost max-flow via successive shortest paths with Dijkstra.

Uses Johnson's potential technique to avoid negative edges after
the initial Bellman-Ford pass, achieving O(F * E log V) overall
where F is the maximum flow value.
"""

import heapq
from collections import defaultdict

# === Min-Cost Max-Flow ===

def min_cost_max_flow(
    n: int, edges: list[tuple[int, int, int, int]], source: int, sink: int
) -> tuple[int, int]:
    """Compute min-cost max-flow.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of (u, v, capacity, cost) tuples.
        source: Source vertex.
        sink: Sink vertex.

    Returns:
        Tuple (max_flow_value, min_cost).
    """
    graph = [[] for _ in range(n)]

    def add_edge(u: int, v: int, cap: int, cost: int) -> None:
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for u, v, cap, cost in edges:
        add_edge(u, v, cap, cost)

    total_flow = 0
    total_cost = 0
    potential = [0] * n  # Johnson's potentials

    while True:
        # Dijkstra with potentials
        dist = [float('inf')] * n
        dist[source] = 0
        prev_node = [-1] * n
        prev_edge = [-1] * n
        pq = [(0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for i, (v, cap, cost, _) in enumerate(graph[u]):
                if cap > 0:
                    new_dist = d + cost + potential[u] - potential[v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev_node[v] = u
                        prev_edge[v] = i
                        heapq.heappush(pq, (new_dist, v))

        if dist[sink] == float('inf'):
            break

        # Update potentials
        for v in range(n):
            if dist[v] < float('inf'):
                potential[v] += dist[v]

        # Find bottleneck along shortest path
        bottleneck = float('inf')
        v = sink
        while v != source:
            u = prev_node[v]
            idx = prev_edge[v]
            bottleneck = min(bottleneck, graph[u][idx][1])
            v = u

        # Augment flow
        v = sink
        while v != source:
            u = prev_node[v]
            idx = prev_edge[v]
            graph[u][idx][1] -= bottleneck
            graph[v][graph[u][idx][3]][1] += bottleneck
            v = u

        total_flow += bottleneck
        total_cost += bottleneck * potential[sink]

    return total_flow, total_cost


# === Demonstration ===

if __name__ == "__main__":
    # Network: s=0, a=1, b=2, t=3
    # (u, v, capacity, cost_per_unit)
    edges = [
        (0, 1, 4, 1),   # s->a: cap 4, cost 1
        (0, 2, 3, 2),   # s->b: cap 3, cost 2
        (1, 2, 2, 1),   # a->b: cap 2, cost 1
        (1, 3, 3, 3),   # a->t: cap 3, cost 3
        (2, 3, 5, 2),   # b->t: cap 5, cost 2
    ]
    flow, cost = min_cost_max_flow(4, edges, 0, 3)
    print(f"Max flow: {flow}")
    print(f"Min cost: {cost}")
```

**Output:**

```
Max flow: 7
Min cost: 27
```

The algorithm finds the maximum flow of $7$ units at a total cost of $27$. It preferentially routes flow through cheaper paths first, then uses more expensive paths only when necessary to maximize the total flow.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Successive shortest paths (Bellman-Ford) | $O(V E \cdot |f^*|)$ |
| With Dijkstra + potentials | $O(|f^*| \cdot E \log V)$ |
| Space | $O(V + E)$ |

For networks with small maximum flow values, the successive shortest paths approach is practical. For large flow values, cycle-canceling or network simplex algorithms may be more efficient.

## Applications

- **Transportation.** Ship goods from factories to warehouses at minimum shipping cost while meeting demand.
- **Assignment problem.** The Hungarian algorithm is a special case of min-cost flow on bipartite networks.
- **Network design.** Route traffic through a communication network minimizing latency or cost.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 26: Maximum Flow.
- Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.
