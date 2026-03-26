# Routing Algorithms

Every packet traveling across a network must find a path from source to destination. **Routing algorithms** determine these paths by modeling the network as a weighted graph $G = (V, E)$ where vertices represent routers, edges represent links, and weights capture costs such as latency, hop count, or bandwidth. The two fundamental paradigms -- distance-vector and link-state -- correspond directly to the Bellman-Ford and Dijkstra algorithms, respectively.

## Network as a Graph

A network of $n$ routers with $m$ links is modeled as a weighted graph:

- Each router is a vertex $v \in V$, $|V| = n$.
- Each link is an edge $(u, v) \in E$ with weight $w(u, v) \ge 0$.
- The routing problem is single-source shortest path: for each router, compute the best path to every destination.

## Distance-Vector Routing

Each router maintains a **distance vector** $d(v, \cdot)$ storing its estimated distance to every destination. Routers periodically exchange vectors with neighbors and update via the Bellman-Ford relaxation:

$$
d(v, u) \leftarrow \min_{w \in \text{neighbors}(v)} \bigl(c(v, w) + d(w, u)\bigr)
$$

**Properties:**

- Converges in at most $|V| - 1$ rounds.
- Each router only needs information from its immediate neighbors (distributed computation).
- Vulnerable to count-to-infinity: when a link fails, routers may slowly propagate incorrect distances.

**Complexity per router per round**: $O(|V| \cdot \text{degree})$.

## Link-State Routing

Each router floods **link-state advertisements** (LSAs) describing its neighbors and link costs. Every router builds the complete topology graph and runs Dijkstra's algorithm independently.

**Properties:**

- Converges as soon as LSAs reach all routers (typically one flooding round).
- Requires $O(|V| + |E|)$ storage per router for the full topology.
- Immune to count-to-infinity since each router has the complete graph.

**Complexity per router**: $O((|V| + |E|) \log |V|)$ for Dijkstra with a binary heap.

## Comparison

| Property | Distance-Vector | Link-State |
|---|---|---|
| Algorithm | Bellman-Ford | Dijkstra |
| Knowledge | Local (neighbors only) | Global (full topology) |
| Message size | $O(|V|)$ per update | $O(\text{degree})$ per LSA |
| Convergence | $O(|V|)$ rounds | $O(1)$ flood + compute |
| Loop-free | No (count-to-infinity) | Yes |
| Example protocol | RIP | OSPF, IS-IS |

## Path-Vector Routing

For inter-domain routing (between autonomous systems), **path-vector** protocols like BGP store the full AS path for each route. This prevents loops (reject routes containing own AS number) and enables policy-based routing decisions beyond shortest-path.

## Implementation

```python
"""
Routing Algorithms -- comparison of distance-vector and link-state.

Implements both Bellman-Ford (distance-vector) and Dijkstra (link-state)
on the same network graph and verifies they produce identical results.
"""

import heapq
from collections import defaultdict


# === Graph ====================================================================

def build_graph(links: list[tuple[int, int, int]],
                n: int) -> dict[int, list[tuple[int, int]]]:
    """Build a bidirectional adjacency list."""
    graph: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for u, v, w in links:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph


# === Bellman-Ford (Distance-Vector) ===========================================

def bellman_ford(n: int, links: list[tuple[int, int, int]],
                 source: int) -> list[float]:
    """Compute shortest distances using Bellman-Ford."""
    dist = [float("inf")] * n
    dist[source] = 0
    # Undirected edges: relax both directions
    edges = [(u, v, w) for u, v, w in links] + [(v, u, w) for u, v, w in links]
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


# === Dijkstra (Link-State) ====================================================

def dijkstra(graph: dict[int, list[tuple[int, int]]],
             source: int) -> list[float]:
    """Compute shortest distances using Dijkstra."""
    n = len(graph)
    dist = [float("inf")] * n
    dist[source] = 0
    pq: list[tuple[float, int]] = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist


# === Main =====================================================================

if __name__ == "__main__":
    n = 6
    links = [
        (0, 1, 7), (0, 2, 9), (0, 5, 14),
        (1, 2, 10), (1, 3, 15),
        (2, 3, 11), (2, 5, 2),
        (3, 4, 6),
        (4, 5, 9),
    ]

    graph = build_graph(links, n)
    bf_dist = bellman_ford(n, links, source=0)
    dj_dist = dijkstra(graph, source=0)

    print("Source: Router 0")
    print(f"{'Dest':>6} {'Bellman-Ford':>13} {'Dijkstra':>9}")
    for i in range(n):
        print(f"{i:>6} {bf_dist[i]:>13.0f} {dj_dist[i]:>9.0f}")

    print(f"\nResults match: {bf_dist == dj_dist}")
```

**Output:**

```
Source: Router 0
  Dest  Bellman-Ford  Dijkstra
     0             0         0
     1             7         7
     2             9         9
     3            20        20
     4            20        20
     5            11        11

Results match: True
```

Both algorithms produce identical shortest-path distances, confirming that distance-vector and link-state routing converge to the same result -- they differ in how information is shared, not in the final outcome.

## Reference

- Kurose, J.F. and Ross, K.W. *Computer Networking: A Top-Down Approach*. Pearson
- Cormen, T.H., Leiserson, C.E., Rivest, R.L., and Stein, C. *Introduction to Algorithms*. MIT Press
