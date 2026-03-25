# Johnson's Algorithm

Floyd-Warshall runs in $\Theta(V^3)$ regardless of graph density.  For
**sparse** graphs where $E \ll V^2$, running Dijkstra from every vertex would
be faster — but Dijkstra requires non-negative weights.  Johnson's algorithm
bridges this gap by **reweighting** edges to eliminate negative weights, then
running Dijkstra from each vertex.  The result is an all-pairs shortest path
algorithm that runs in $O(V^2 \log V + VE)$, which is faster than
Floyd-Warshall on sparse graphs.

## The Reweighting Technique

The core insight is that adding a carefully chosen value to each edge weight
can make all weights non-negative without changing which paths are shortest.

Given a weight function $w$ and a **potential function** $h: V \to \mathbb{R}$,
define the reweighted edge weight:

$$
\hat{w}(u, v) = w(u, v) + h(u) - h(v)
$$

This reweighting preserves shortest paths because for any path
$p = \langle v_0, v_1, \dots, v_k \rangle$:

$$
\hat{w}(p) = \sum_{i=0}^{k-1} \hat{w}(v_i, v_{i+1}) = \sum_{i=0}^{k-1} \left[w(v_i, v_{i+1}) + h(v_i) - h(v_{i+1})\right] = w(p) + h(v_0) - h(v_k)
$$

The telescoping sum means the reweighted path weight differs from the original
by a constant $h(v_0) - h(v_k)$ that depends only on the endpoints.  Therefore,
a shortest path under $w$ is also shortest under $\hat{w}$.

## Choosing the Potential Function

To make all reweighted edges non-negative, Johnson's algorithm sets
$h(v) = \delta(s', v)$ where $s'$ is a new vertex connected to every existing
vertex with zero-weight edges:

1. Add a new vertex $s'$ to the graph.
2. Add edges $(s', v)$ with weight $0$ for all $v \in V$.
3. Run Bellman-Ford from $s'$ to compute $h(v) = \delta(s', v)$.

By the triangle inequality, $\delta(s', v) \le \delta(s', u) + w(u, v)$ for
every edge $(u, v)$.  Rearranging:

$$
\hat{w}(u, v) = w(u, v) + h(u) - h(v) = w(u, v) + \delta(s', u) - \delta(s', v) \ge 0
$$

If Bellman-Ford detects a negative cycle, the algorithm reports it and stops.

## Algorithm Steps

```
JOHNSON(G, w):
    1. Add vertex s' and edges (s', v, 0) for all v in V
    2. Run BELLMAN-FORD(G', w, s')
       - If negative cycle detected: return "negative cycle"
       - Otherwise: h(v) = delta(s', v)
    3. For each edge (u, v) in E:
       w_hat(u, v) = w(u, v) + h(u) - h(v)
    4. For each vertex u in V:
       Run DIJKSTRA(G, w_hat, u) to get d_hat(u, v) for all v
       For each vertex v in V:
           d(u, v) = d_hat(u, v) - h(u) + h(v)
    5. Return distance matrix d
```

## Complexity

| Step | Time |
|---|---|
| Bellman-Ford from $s'$ | $O(VE)$ |
| Reweight all edges | $O(E)$ |
| $V$ runs of Dijkstra (binary heap) | $O(V(V+E)\log V)$ |
| Un-reweight distances | $O(V^2)$ |
| **Total** | $O(V^2 \log V + VE)$ |

For sparse graphs ($E = O(V)$), this gives $O(V^2 \log V)$, which is
significantly better than Floyd-Warshall's $\Theta(V^3)$.  For dense graphs
($E = O(V^2)$), both algorithms are $\Theta(V^3)$, and Floyd-Warshall is
simpler with lower constant factors.

## When to Use Which

| Criterion | Floyd-Warshall | Johnson |
|---|---|---|
| Graph density | Dense ($E \approx V^2$) | Sparse ($E \ll V^2$) |
| Negative edges | Supported | Supported |
| Time | $\Theta(V^3)$ | $O(V^2\log V + VE)$ |
| Implementation | Simpler | More complex |

## Worked Example

Consider 4 vertices with edges including a negative weight:

| Edge | Weight |
|---|---|
| $(0, 1)$ | 1 |
| $(0, 2)$ | 4 |
| $(1, 2)$ | -3 |
| $(2, 3)$ | 2 |

**Step 1:** Add $s'$ with edges $(s', 0) = 0$, $(s', 1) = 0$, $(s', 2) = 0$,
$(s', 3) = 0$.

**Step 2:** Bellman-Ford from $s'$: $h(0) = 0$, $h(1) = 0$, $h(2) = -3$,
$h(3) = -1$.

**Step 3:** Reweight.
$\hat{w}(0,1) = 1 + 0 - 0 = 1$.
$\hat{w}(0,2) = 4 + 0 - (-3) = 7$.
$\hat{w}(1,2) = -3 + 0 - (-3) = 0$.
$\hat{w}(2,3) = 2 + (-3) - (-1) = 0$.

All reweighted edges are non-negative.

**Step 4:** Run Dijkstra from each vertex with reweighted edges, then
un-reweight: $d(u, v) = \hat{d}(u, v) - h(u) + h(v)$.

## Implementation

```python
"""
Johnson's algorithm for all-pairs shortest paths.

Combines Bellman-Ford reweighting with Dijkstra to achieve
O(V^2 log V + VE) time, which beats Floyd-Warshall on sparse graphs.
"""

import heapq
from math import inf


# === Bellman-Ford for potential computation ==================================

def bellman_ford(vertices: list, edges: list, source) -> tuple[dict, bool]:
    """Run Bellman-Ford and return distances and cycle status.

    Returns (dist, True) if no negative cycle, (dist, False) otherwise.
    """
    dist = {v: inf for v in vertices}
    dist[source] = 0

    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if dist[u] != inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Negative cycle check
    for u, v, w in edges:
        if dist[u] != inf and dist[u] + w < dist[v]:
            return dist, False

    return dist, True


# === Dijkstra ================================================================

def dijkstra(graph: dict, source) -> dict:
    """Run Dijkstra from source, returning shortest distances."""
    dist = {v: inf for v in graph}
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist


# === Johnson's algorithm =====================================================

def johnson(vertices: list, edges: list) -> tuple[dict, bool]:
    """Compute all-pairs shortest paths using Johnson's algorithm.

    Parameters
    ----------
    vertices : list
        All vertex identifiers.
    edges : list of (u, v, w)
        Directed edges with weights.

    Returns
    -------
    dist : dict of dict
        dist[u][v] = shortest path weight from u to v.
    no_negative_cycle : bool
        True if no negative cycle exists.
    """
    # Step 1: Add virtual source s' with zero-weight edges to all vertices
    s_prime = "__s_prime__"
    aug_vertices = vertices + [s_prime]
    aug_edges = edges + [(s_prime, v, 0) for v in vertices]

    # Step 2: Bellman-Ford from s' to get potentials h
    h, ok = bellman_ford(aug_vertices, aug_edges, s_prime)
    if not ok:
        return {}, False

    # Step 3: Reweight edges
    reweighted_graph = {v: [] for v in vertices}
    for u, v, w in edges:
        w_hat = w + h[u] - h[v]
        reweighted_graph[u].append((v, w_hat))

    # Step 4: Run Dijkstra from each vertex and un-reweight
    dist = {}
    for u in vertices:
        d_hat = dijkstra(reweighted_graph, u)
        dist[u] = {}
        for v in vertices:
            if d_hat[v] == inf:
                dist[u][v] = inf
            else:
                dist[u][v] = d_hat[v] - h[u] + h[v]

    return dist, True


# === Demo ====================================================================

if __name__ == "__main__":
    vertices = [0, 1, 2, 3]
    edges = [
        (0, 1, 1), (0, 2, 4),
        (1, 2, -3),
        (2, 3, 2),
    ]

    dist, ok = johnson(vertices, edges)
    print(f"No negative cycle: {ok}")
    print("\nAll-pairs shortest distances:")
    for u in vertices:
        row = {v: dist[u][v] if dist[u][v] != inf else "inf" for v in vertices}
        print(f"  From {u}: {row}")
```

**Output:**

```
No negative cycle: True

All-pairs shortest distances:
  From 0: {0: 0, 1: 1, 2: -2, 3: 0}
  From 1: {0: 'inf', 1: 0, 2: -3, 3: -1}
  From 2: {0: 'inf', 1: 'inf', 2: 0, 3: 2}
  From 3: {0: 'inf', 1: 'inf', 2: 'inf', 3: 0}
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 25.3: Johnson's Algorithm for Sparse Graphs.
- Johnson, D. B. (1977). Efficient algorithms for shortest paths in sparse
  networks. *Journal of the ACM*, 24(1), 1-13.
