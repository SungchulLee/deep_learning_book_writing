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

## Exercises

**Exercise 1.**
Model a 6-router network as a weighted graph and compute the shortest-path routing table for one router using Dijkstra's algorithm. Show the forwarding decisions.

??? success "Solution to Exercise 1"
    Network: R1-R2(2), R1-R3(5), R2-R3(1), R2-R4(3), R3-R5(2), R4-R5(1), R4-R6(4), R5-R6(2). Dijkstra from R1: dist[R1]=0, dist[R2]=2 (via R2), dist[R3]=3 (via R2-R3), dist[R4]=5 (via R2-R4), dist[R5]=5 (via R2-R3-R5), dist[R6]=7 (via R2-R3-R5-R6). Forwarding table at R1: R2$\to$direct, R3$\to$R2, R4$\to$R2, R5$\to$R2, R6$\to$R2. All traffic leaves through the R1-R2 link because R2 is on all shortest paths. This is common in topologies where one neighbor provides the best gateway. $\square$

---

**Exercise 2.**
Compare distance-vector and link-state routing paradigms. What algorithmic property of each determines its convergence behavior?

??? success "Solution to Exercise 2"
    **Distance-vector** (Bellman-Ford): each router knows only its own distances to destinations and shares these with neighbors. Convergence requires iterative relaxation: information propagates one hop per round. Convergence time is $O(V)$ rounds, where each round can take 30 seconds (RIP). Vulnerable to count-to-infinity because routers act on potentially stale information from neighbors. **Link-state** (Dijkstra): each router knows the complete network topology via flooding. Convergence requires two phases: flooding ($O(\text{diameter})$ time, typically milliseconds) and local Dijkstra computation ($O(V \log V + E)$, microseconds). The complete-information property ensures a single computation produces correct results -- no iterative convergence needed. The key algorithmic difference: Bellman-Ford is distributed and iterative; Dijkstra is local but requires global state. $\square$

---

**Exercise 3.**
Explain how routing loops form and describe three mechanisms used in routing protocols to prevent or mitigate them.

??? success "Solution to Exercise 3"
    A routing loop occurs when router A forwards packets for destination D to router B, and B forwards them back to A (directly or through intermediate routers). Packets circulate until their TTL expires. Loops form when routing tables are inconsistent during convergence. Mechanisms: (1) **TTL (Time to Live)**: each packet's TTL is decremented at every hop. When it reaches 0, the packet is dropped. This limits the damage of loops but does not prevent them. (2) **Split horizon / poison reverse**: a router does not advertise a route back to the neighbor from which it was learned (or advertises it with infinity). This prevents simple two-node loops. (3) **Hold-down timers**: after a route is withdrawn, the router refuses to accept new routes to that destination for a hold-down period, preventing stale information from creating loops during convergence. $\square$

---

**Exercise 4.**
A network has equal-cost multiple paths (ECMP) between two endpoints. Explain how ECMP routing works and what benefits it provides.

??? success "Solution to Exercise 4"
    ECMP occurs when multiple shortest paths of equal cost exist between a source and destination. Instead of choosing one path, the router distributes traffic across all equal-cost paths. Distribution methods: (1) **Per-packet**: each packet is independently assigned to a path (e.g., round-robin). This maximizes bandwidth utilization but can cause packet reordering. (2) **Per-flow**: a hash of the flow identifier (source IP, destination IP, port, protocol) determines the path. All packets in a flow follow the same path, avoiding reordering. This is the standard approach. Benefits: (1) increased aggregate bandwidth (multiple paths share the load); (2) fault tolerance (if one path fails, traffic shifts to remaining paths); (3) better utilization of network links (prevents bottlenecks on a single path). $\square$

---

**Exercise 5.**
Software-defined networking (SDN) separates the control plane from the data plane. How does this change the routing paradigm, and what advantages does it offer over distributed routing protocols?

??? success "Solution to Exercise 5"
    In traditional routing, each router independently computes routes using distributed protocols (OSPF, BGP). In SDN, a centralized controller maintains the global network view and computes routes for all routers, then installs forwarding rules in each router's flow table. Advantages: (1) **Global optimization**: the controller can compute globally optimal paths (e.g., minimizing congestion, maximizing throughput) rather than relying on shortest-path heuristics. (2) **Rapid innovation**: new routing policies are software changes in the controller, not firmware updates on every router. (3) **Traffic engineering**: the controller can reroute flows dynamically based on real-time load, which distributed protocols cannot do efficiently. (4) **Simplified routers**: data-plane devices only forward packets based on rules; they need no routing protocol software. Disadvantage: the controller is a single point of failure (mitigated by controller replication) and must scale to handle topology changes and flow requests for the entire network. $\square$
