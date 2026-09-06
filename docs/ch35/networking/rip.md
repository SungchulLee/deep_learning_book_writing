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

## Exercises

**Exercise 1.**
Trace the Bellman-Ford algorithm in RIP for a 4-router network: A-B(1), B-C(2), A-C(10), C-D(1). Show the distance table at each iteration until convergence.

??? success "Solution to Exercise 1"
    Iteration 0: A=[A:0, B:$\infty$, C:$\infty$, D:$\infty$], B=[A:$\infty$, B:0, C:$\infty$, D:$\infty$], C similarly, D similarly. Iteration 1 (each router shares with neighbors): A learns B=1, C=10. B learns A=1, C=2. C learns A=10, B=2, D=1. D learns C=1. Iteration 2: A learns C=min(10, 1+2)=3 via B. B learns D=min($\infty$, 2+1)=3 via C. D learns B=min($\infty$, 1+2)=3 via C, A=min($\infty$, 1+3)=4 via C. Iteration 3: A learns D=min($\infty$, 1+3)=4 via B. Converged: A's table: B=1, C=3, D=4. All via shortest paths. $\square$

---

**Exercise 2.**
Explain the "count to infinity" problem in RIP and describe two techniques used to mitigate it.

??? success "Solution to Exercise 2"
    When a link fails, routers may slowly increment their distance estimates through circular updates. Example: A-B connected, B-C connected, A-C fails. C's distance to A becomes $\infty$, but B still advertises distance 2 to A. C updates to 3, B updates to 4, and so on until both reach 16 (RIP's infinity). This takes many iterations at 30-second intervals -- potentially minutes. Mitigations: (1) **Split horizon**: a router does not advertise a route back to the neighbor from which it learned the route. This prevents B from telling C about a route to A that goes through C. (2) **Poison reverse**: instead of omitting the route, advertise it with metric $\infty$ (16). This explicitly tells the neighbor the route is unreachable via this path. Together, these techniques prevent two-node loops but may not prevent larger loops. **Triggered updates** (send updates immediately on changes rather than waiting 30 seconds) also speed convergence. $\square$

---

**Exercise 3.**
RIP uses a maximum hop count of 15 (16 = infinity). Prove that Bellman-Ford converges in at most $|V| - 1$ iterations and explain why the hop limit restricts network diameter.

??? success "Solution to Exercise 3"
    In a graph with $|V|$ vertices, any shortest path has at most $|V| - 1$ edges (a simple path visits each vertex at most once). Bellman-Ford relaxes all edges in each iteration. After iteration $k$, the algorithm has found all shortest paths using at most $k$ edges. After $|V| - 1$ iterations, all shortest paths (of any length) are found. With RIP's limit of 15, any destination more than 15 hops away is considered unreachable. This restricts the network diameter to 15. For networks with more than 15 routers in a chain, RIP cannot compute correct routes. This is a fundamental limitation of RIP and is the primary reason it was replaced by OSPF for large networks. The limit also serves as a practical safeguard against count-to-infinity: the maximum "counting" is from some value to 16, not to true infinity. $\square$

---

**Exercise 4.**
Compare the message complexity of RIP and OSPF. How many bytes per second does each protocol consume on a network with 100 routers and 200 links?

??? success "Solution to Exercise 4"
    **RIP**: each router sends its full routing table (up to 25 entries per UDP packet, 20 bytes per entry) to all neighbors every 30 seconds. With 100 destinations, each router sends $\lceil 100/25 \rceil = 4$ packets ($\sim$2 KB) every 30 seconds. With average degree 4 (200 links / 100 routers $\times$ 2): each router sends to 4 neighbors, totaling $4 \times 2 = 8$ KB per 30 seconds $= 267$ bytes/sec per router. Network-wide: $\sim$26.7 KB/sec. **OSPF**: in steady state, only Hello packets are sent (every 10 seconds, $\sim$48 bytes per link). With 200 links: $200 \times 48 / 10 = 960$ bytes/sec. LSA refreshes occur every 30 minutes (negligible). On topology changes, LSA flooding adds brief bursts. OSPF uses less bandwidth in steady state but more during topology changes. $\square$

---

**Exercise 5.**
A small office network has 5 routers connected in a ring. Compare how long RIP and OSPF take to converge after the network starts from scratch.

??? success "Solution to Exercise 5"
    **RIP**: distance vectors propagate one hop per update cycle. With a ring of 5 routers (diameter 2), convergence requires 2 iterations. At 30-second intervals: 60 seconds minimum. In practice, updates are not synchronized, so convergence takes 60--90 seconds. The ring topology has no count-to-infinity issue during initial convergence (only on failures). **OSPF**: each router floods its LSA upon startup. With 5 routers and a ring topology, flooding completes in $\sim$2 hop delays ($\sim$100 ms). Each router then runs Dijkstra on the 5-node, 5-edge graph ($< 1$ ms). Total convergence: $\sim$1 second. OSPF converges 60x faster. For this tiny network, the difference is noticeable but not critical. For larger networks, OSPF's advantage becomes essential (RIP would take minutes while OSPF converges in seconds). $\square$
