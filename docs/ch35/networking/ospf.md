# Dijkstra in OSPF

The **Open Shortest Path First** (OSPF) protocol is a link-state routing protocol used within an autonomous system (AS). Unlike distance-vector protocols such as RIP, which propagate distance estimates from neighbor to neighbor, OSPF floods the entire network topology to every router, allowing each router to independently compute shortest paths using Dijkstra's algorithm.

## Link-State Routing

In OSPF, each router:

1. **Discovers neighbors** by exchanging Hello packets on each interface.
2. **Measures link costs** (bandwidth, delay, or administrative weight).
3. **Floods link-state advertisements (LSAs)** describing its neighbors and link costs to all routers in the area.
4. **Builds a complete topology graph** from received LSAs.
5. **Runs Dijkstra's algorithm** on this graph to compute the shortest-path tree rooted at itself.

The result is a routing table mapping each destination to the appropriate next-hop interface.

## OSPF Areas

Large networks are divided into **areas** to limit the scope of LSA flooding. Area 0 is the backbone; all other areas connect through it. Each router runs Dijkstra only within its area, and **Area Border Routers** (ABRs) summarize inter-area routes.

## Dijkstra's Algorithm

Given a weighted graph $G = (V, E)$ with non-negative edge weights and source $s$, Dijkstra's algorithm computes shortest-path distances $d(s, v)$ for all $v \in V$.

The algorithm maintains a priority queue of vertices ordered by tentative distance:

1. Initialize $d(s) = 0$, $d(v) = \infty$ for all $v \ne s$.
2. Extract the vertex $u$ with minimum $d(u)$ from the priority queue.
3. For each neighbor $v$ of $u$, if $d(u) + w(u, v) < d(v)$, update $d(v)$ (relaxation).
4. Repeat until the queue is empty.

With a binary heap, the running time is:

$$
T(V, E) = O((|V| + |E|) \log |V|)
$$

Each OSPF router runs this computation whenever the topology changes (LSA update received).

!!! tip "OSPF vs RIP"
    OSPF converges faster than RIP because every router has the complete topology and computes paths locally. RIP relies on iterative distance-vector exchanges that can take up to $|V| - 1$ rounds to converge, and it suffers from the count-to-infinity problem.

## Implementation

```python
"""
OSPF Shortest Path -- Dijkstra's algorithm for link-state routing.

Each router builds a complete topology graph from LSAs and runs
Dijkstra to compute the shortest-path tree for forwarding decisions.
"""

import heapq
from collections import defaultdict


# === Graph Construction =======================================================

def build_graph(links: list[tuple[int, int, int]]) -> dict[int, list[tuple[int, int]]]:
    """Build adjacency list from (src, dst, cost) link descriptions."""
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, w in links:
        graph[u].append((v, w))
        graph[v].append((u, w))  # OSPF links are bidirectional
    # Ensure all nodes appear even if they have no outgoing edges
    for u, v, _ in links:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
    return dict(graph)


# === Dijkstra =================================================================

def dijkstra(graph: dict[int, list[tuple[int, int]]],
             source: int) -> tuple[dict[int, float], dict[int, int | None]]:
    """Compute shortest paths from *source*.

    Returns (distances, predecessors) for reconstructing paths.
    """
    dist: dict[int, float] = {v: float("inf") for v in graph}
    prev: dict[int, int | None] = {v: None for v in graph}
    dist[source] = 0
    pq: list[tuple[float, int]] = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev


def shortest_path(prev: dict[int, int | None], target: int) -> list[int]:
    """Reconstruct the shortest path to *target* from the predecessor map."""
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


# === Main =====================================================================

if __name__ == "__main__":
    # Simulate a small OSPF area with 5 routers
    links = [
        (0, 1, 4),   # Router 0 -- Router 1, cost 4
        (0, 2, 1),   # Router 0 -- Router 2, cost 1
        (2, 1, 2),   # Router 2 -- Router 1, cost 2
        (1, 3, 1),   # Router 1 -- Router 3, cost 1
        (2, 3, 5),   # Router 2 -- Router 3, cost 5
        (3, 4, 3),   # Router 3 -- Router 4, cost 3
    ]

    graph = build_graph(links)
    dist, prev = dijkstra(graph, source=0)

    print("OSPF routing table for Router 0:")
    print(f"{'Dest':>6} {'Cost':>6} {'Path'}")
    for dest in sorted(graph.keys()):
        path = shortest_path(prev, dest)
        print(f"{dest:>6} {dist[dest]:>6.0f}   {' -> '.join(map(str, path))}")
```

**Output:**

```
OSPF routing table for Router 0:
  Dest   Cost Path
     0      0   0
     1      3   0 -> 2 -> 1
     2      1   0 -> 2
     3      4   0 -> 2 -> 1 -> 3
     4      7   0 -> 2 -> 1 -> 3 -> 4
```

The routing table shows that Router 0 reaches Router 1 via Router 2 (cost 3) rather than the direct link (cost 4), demonstrating how Dijkstra finds the true shortest path through the topology.

## Reference

- Moy, J. "OSPF Version 2." RFC 2328, 1998
- Cormen, T.H., Leiserson, C.E., Rivest, R.L., and Stein, C. *Introduction to Algorithms*. MIT Press
