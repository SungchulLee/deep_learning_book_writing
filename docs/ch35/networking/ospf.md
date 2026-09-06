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

## Exercises

**Exercise 1.**
Explain how OSPF routers build the link-state database and why Dijkstra's algorithm is applied locally rather than globally.

??? success "Solution to Exercise 1"
    Each OSPF router creates a Link-State Advertisement (LSA) describing its directly connected links and their costs. LSAs are flooded to all routers in the area using reliable flooding (each router forwards received LSAs to all neighbors except the sender). After flooding converges, every router has an identical link-state database containing all LSAs -- a complete graph description. Each router then independently runs Dijkstra's algorithm on this database to compute shortest paths from itself to all destinations. The algorithm is applied locally because each router needs shortest paths from its own perspective (as the source). The global topology is the same everywhere, but the shortest-path tree differs for each source. This local computation avoids centralized coordination and is robust to individual router failures. $\square$

---

**Exercise 2.**
A network has 5 routers (A-E) with links: A-B(1), A-C(4), B-C(2), B-D(5), C-D(1), C-E(6), D-E(2). Compute the shortest-path tree from router A using Dijkstra's algorithm.

??? success "Solution to Exercise 2"
    Initialize: dist[A]=0, all others $\infty$. Step 1: process A. Update: dist[B]=1, dist[C]=4. Step 2: process B (dist=1). Update: dist[C]=min(4, 1+2)=3, dist[D]=min($\infty$, 1+5)=6. Step 3: process C (dist=3). Update: dist[D]=min(6, 3+1)=4, dist[E]=min($\infty$, 3+6)=9. Step 4: process D (dist=4). Update: dist[E]=min(9, 4+2)=6. Step 5: process E (dist=6). Shortest-path tree from A: A$\to$B (cost 1), A$\to$B$\to$C (cost 3), A$\to$B$\to$C$\to$D (cost 4), A$\to$B$\to$C$\to$D$\to$E (cost 6). Router A installs these as its routing table entries. $\square$

---

**Exercise 3.**
OSPF uses areas to scale to large networks. Explain how area partitioning reduces the computational cost of Dijkstra's algorithm.

??? success "Solution to Exercise 3"
    Without areas, every router runs Dijkstra on the full network graph with $V$ vertices and $E$ edges, costing $O(V \log V + E)$ per router. With $V = 10{,}000$, this is expensive and must be repeated whenever any link changes. OSPF divides the network into areas. Each router only maintains the full topology of its own area and summary routes from other areas (advertised by area border routers). If the network is divided into $k$ areas of $\sim V/k$ routers each, Dijkstra runs on a graph of size $V/k + k$ (local nodes plus inter-area summaries). For $k = 100$ and $V = 10{,}000$: each router processes $\sim 200$ nodes instead of 10,000 -- a 50x reduction. Link changes in one area trigger Dijkstra only in that area, not network-wide. $\square$

---

**Exercise 4.**
Compare OSPF (Dijkstra-based) with RIP (Bellman-Ford-based) in terms of convergence speed, message overhead, and suitability for large networks.

??? success "Solution to Exercise 4"
    **Convergence speed**: OSPF converges in seconds (LSA flooding + one Dijkstra run). RIP converges slowly (distance vectors propagate hop-by-hop, with $O(\text{diameter})$ iterations, each separated by 30-second update intervals). The "count to infinity" problem further delays RIP convergence on link failures. **Message overhead**: OSPF floods LSAs to all routers (high initial cost but triggered only by changes). RIP sends full routing tables to neighbors every 30 seconds regardless of changes (steady overhead). **Scalability**: OSPF scales to large networks via area hierarchies. RIP is limited to 15 hops (infinity = 16) and does not support hierarchical routing. OSPF is the standard for enterprise and ISP networks; RIP is suitable only for small networks. $\square$

---

**Exercise 5.**
A network link between routers B and C fails. Describe the sequence of events in OSPF from failure detection to routing table convergence.

??? success "Solution to Exercise 5"
    (1) **Detection**: routers B and C stop receiving Hello packets from each other (default dead interval: 40 seconds, or faster with BFD at $\sim$50 ms). Each declares the link down. (2) **LSA generation**: B and C each generate new LSAs omitting the B-C link and increment their sequence numbers. (3) **Flooding**: the new LSAs are flooded throughout the area. Each router receiving a newer LSA forwards it to all neighbors, ensuring all routers update within $\sim$1 second. (4) **SPF computation**: each router detects the LSA change, schedules a Dijkstra recomputation (with a throttle delay of $\sim$200 ms to batch changes), and runs Dijkstra on the updated link-state database. (5) **Routing table update**: new shortest paths that avoid the B-C link are installed. Traffic is rerouted. Total convergence time: $\sim$1--2 seconds with tuned timers, or $\sim$40 seconds with default Hello timers. $\square$
