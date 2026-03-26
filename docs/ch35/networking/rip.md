# Bellman-Ford in RIP

The **Routing Information Protocol** (RIP) is a distance-vector routing protocol that uses the Bellman-Ford algorithm to compute shortest paths in a network. Each router maintains a table of distances to all destinations and periodically shares this table with its neighbors. Through iterative exchanges, all routers converge on shortest-path distances -- provided the network has no negative-weight edges (which physical networks never do).

## Distance-Vector Protocol

In a distance-vector protocol, each router $v$:

1. **Initializes** $d(v, v) = 0$ and $d(v, u) = \infty$ for all other destinations $u$.
2. **Periodically broadcasts** its distance vector $d(v, \cdot)$ to all neighbors.
3. **Updates** upon receiving a neighbor $w$'s vector: for each destination $u$,

$$
d(v, u) \leftarrow \min\bigl(d(v, u),\; c(v, w) + d(w, u)\bigr)
$$

where $c(v, w)$ is the link cost from $v$ to $w$.

This is exactly the relaxation step of Bellman-Ford, distributed across routers.

## RIP Specifics

- **Metric**: Hop count (each link has cost 1).
- **Maximum distance**: 15 hops; distance 16 means "unreachable." This limits RIP to small networks.
- **Update interval**: Every 30 seconds, each router broadcasts its full routing table.
- **Convergence**: After at most $|V| - 1$ rounds of updates, all distances are correct (in a stable network).

## Count-to-Infinity Problem

When a link fails, distance-vector protocols can enter a loop where two routers keep incrementing their distance estimates through each other. RIP mitigates this with:

- **Split horizon**: Do not advertise a route back through the neighbor it was learned from.
- **Poison reverse**: Advertise the route back to the learning neighbor with distance 16 (infinity).
- **Triggered updates**: Send immediate updates on topology changes rather than waiting for the 30-second timer.

!!! warning "Slow convergence"
    Despite these mitigations, RIP can still converge slowly after topology changes. For networks larger than about 15 hops, link-state protocols like OSPF are preferred because every router has the full topology and can recompute routes immediately.

## Bellman-Ford Algorithm

Given a graph with $|V|$ vertices and $|E|$ edges, Bellman-Ford relaxes all edges $|V| - 1$ times:

$$
T(V, E) = O(|V| \cdot |E|)
$$

Unlike Dijkstra, it handles negative edge weights and can detect negative cycles (though these do not occur in RIP's hop-count metric).

## Implementation

```python
"""
RIP Distance-Vector Routing -- Bellman-Ford applied to hop counts.

Simulates a RIP-like protocol where routers exchange distance vectors
and iteratively converge on shortest-path hop counts.
"""

from __future__ import annotations
from collections import defaultdict

INFINITY = 16  # RIP maximum distance (unreachable)


# === Bellman-Ford (centralized) ===============================================

def bellman_ford(n_routers: int, links: list[tuple[int, int, int]],
                 source: int) -> tuple[list[int], list[int | None]]:
    """Compute shortest distances from *source* using Bellman-Ford.

    Returns (distances, predecessors).
    """
    dist = [INFINITY] * n_routers
    prev: list[int | None] = [None] * n_routers
    dist[source] = 0

    for _ in range(n_routers - 1):
        for u, v, w in links:
            if dist[u] + w < dist[v]:
                dist[v] = min(dist[u] + w, INFINITY)
                prev[v] = u
            if dist[v] + w < dist[u]:
                dist[u] = min(dist[v] + w, INFINITY)
                prev[u] = v

    return dist, prev


# === Distance-Vector Simulation ===============================================

def simulate_rip(n_routers: int,
                 links: list[tuple[int, int, int]]) -> dict[int, list[int]]:
    """Simulate RIP distance-vector exchange until convergence.

    Returns a dict mapping each router to its distance vector.
    """
    # Build adjacency list
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, w in links:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Initialize distance tables
    tables: dict[int, list[int]] = {}
    for r in range(n_routers):
        tables[r] = [INFINITY] * n_routers
        tables[r][r] = 0

    # Iterate until convergence
    for _ in range(n_routers - 1):
        updated = False
        for r in range(n_routers):
            for neighbor, cost in adj[r]:
                for dest in range(n_routers):
                    new_dist = min(cost + tables[neighbor][dest], INFINITY)
                    if new_dist < tables[r][dest]:
                        tables[r][dest] = new_dist
                        updated = True
        if not updated:
            break

    return tables


# === Main =====================================================================

if __name__ == "__main__":
    # Network: 5 routers with hop-count metric (all costs = 1)
    links = [
        (0, 1, 1), (0, 2, 1),
        (1, 3, 1), (2, 3, 1),
        (3, 4, 1),
    ]
    n = 5

    # Centralized Bellman-Ford from router 0
    dist, prev = bellman_ford(n, links, source=0)
    print("Bellman-Ford from router 0:")
    print(f"  Distances: {dist}")

    # Distributed RIP simulation
    tables = simulate_rip(n, links)
    print("\nRIP routing tables (hop counts):")
    for router in range(n):
        print(f"  Router {router}: {tables[router]}")
```

**Output:**

```
Bellman-Ford from router 0:
  Distances: [0, 1, 1, 2, 3]

RIP routing tables (hop counts):
  Router 0: [0, 1, 1, 2, 3]
  Router 1: [1, 0, 2, 1, 2]
  Router 2: [1, 2, 0, 1, 2]
  Router 3: [2, 1, 1, 0, 1]
  Router 4: [3, 2, 2, 1, 0]
```

Both the centralized Bellman-Ford and the distributed RIP simulation produce identical distance vectors, confirming that the iterative distance-vector exchange converges to correct shortest-path hop counts.

## Reference

- Hedrick, C. "Routing Information Protocol." RFC 1058, 1988
- Cormen, T.H., Leiserson, C.E., Rivest, R.L., and Stein, C. *Introduction to Algorithms*. MIT Press
