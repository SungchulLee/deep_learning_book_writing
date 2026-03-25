# Metric TSP Approximation

The general Traveling Salesman Problem (TSP) cannot be approximated within
any constant factor unless P = NP. However, when distances satisfy the
**triangle inequality** — meaning that going directly is never worse than
detouring — constant-factor approximations become possible. This special case
is called **Metric TSP**.

## Problem Definition

Given a complete graph $G = (K_n, w)$ with non-negative edge weights
satisfying the triangle inequality

$$
w(u, v) \le w(u, x) + w(x, v) \quad \forall\, u, v, x \in V
$$

find a Hamiltonian cycle (tour visiting every vertex exactly once) of minimum
total weight.

## MST-Based 2-Approximation

**Intuition.** A minimum spanning tree (MST) connects all vertices with minimum
total edge weight. Walking around the MST visits every vertex (with repeats).
The triangle inequality lets us shortcut repeated vertices without increasing
cost.

**Algorithm:**

1. Compute an MST $T$ of $G$
2. Perform a DFS walk of $T$, listing vertices in the order first visited
3. Return this ordering as a Hamiltonian cycle

!!! tip "Theorem"
    The MST-doubling algorithm produces a tour of cost at most $2 \cdot \text{OPT}$.

**Proof.** Let $C^*$ be the optimal tour. Deleting any edge from $C^*$ yields
a spanning tree, so $w(T) \le w(C^*)$. The DFS walk of $T$ traverses each edge
exactly twice, giving a closed walk of cost $2 \cdot w(T) \le 2 \cdot w(C^*)$.
Shortcutting repeated vertices via the triangle inequality only decreases cost,
so the resulting Hamiltonian cycle has cost at most

$$
2 \cdot w(T) \le 2 \cdot \text{OPT} \qquad \blacksquare
$$

## Christofides-Serdyukov Algorithm

**Intuition.** The MST walk has excess cost because odd-degree vertices force
backtracking. Adding a minimum-weight perfect matching on odd-degree vertices
creates an Eulerian graph, which can be traversed without repeating edges.

**Algorithm:**

1. Compute an MST $T$ of $G$
2. Let $O$ be the set of odd-degree vertices in $T$ (always even in number)
3. Find a minimum-weight perfect matching $M$ on the vertices in $O$
4. Combine $T$ and $M$ to form a multigraph $H$; every vertex now has even
   degree
5. Find an Eulerian circuit of $H$
6. Shortcut repeated vertices to obtain a Hamiltonian cycle

!!! tip "Theorem (Christofides, 1976)"
    The algorithm produces a tour of cost at most $\frac{3}{2} \cdot \text{OPT}$.

**Proof.** The MST satisfies $w(T) \le \text{OPT}$. For the matching, the
optimal tour restricted to vertices in $O$ forms a Hamiltonian cycle on $O$.
This cycle can be split into two perfect matchings (alternate edges), each with
cost at most $\text{OPT}/2$. The minimum matching $M$ satisfies

$$
w(M) \le \frac{\text{OPT}}{2}
$$

The Eulerian circuit of $T \cup M$ has cost $w(T) + w(M)$. After shortcutting:

$$
w(\text{tour}) \le w(T) + w(M) \le \text{OPT} + \frac{\text{OPT}}{2}
= \frac{3}{2}\,\text{OPT} \qquad \blacksquare
$$

This 3/2 ratio stood as the best known for nearly 50 years until a slight
improvement by Karlin, Klein, and Oveis Gharan (2021).

## Implementation

```python
"""
Metric TSP: MST-based 2-approximation.
"""

import heapq
from collections import defaultdict


# === Prim's MST ==============================================================

def prim_mst(n, adj):
    """Compute MST using Prim's algorithm. Returns adjacency list of MST."""
    visited = [False] * n
    mst = defaultdict(list)
    # (weight, vertex, parent)
    heap = [(0, 0, -1)]
    total = 0

    while heap:
        w, u, parent = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total += w
        if parent >= 0:
            mst[parent].append(u)
            mst[u].append(parent)
        for v, wt in adj[u]:
            if not visited[v]:
                heapq.heappush(heap, (wt, v, u))

    return mst, total


# === DFS preorder tour ========================================================

def dfs_preorder(mst, n):
    """DFS preorder traversal of MST to get Hamiltonian cycle order."""
    visited = [False] * n
    order = []
    stack = [0]

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in reversed(mst[u]):
            if not visited[v]:
                stack.append(v)

    return order


# === MST-based 2-approximation ================================================

def metric_tsp_2approx(n, dist):
    """
    2-approximation for Metric TSP via MST doubling.

    dist: n x n distance matrix satisfying triangle inequality.
    Returns (tour_cost, tour_order).
    """
    # Build adjacency
    adj = defaultdict(list)
    for u in range(n):
        for v in range(u + 1, n):
            adj[u].append((v, dist[u][v]))
            adj[v].append((u, dist[u][v]))

    mst, mst_cost = prim_mst(n, adj)
    tour = dfs_preorder(mst, n)

    # Compute tour cost
    cost = sum(dist[tour[i]][tour[i + 1]] for i in range(n - 1))
    cost += dist[tour[-1]][tour[0]]

    return cost, tour


# === Demo =====================================================================

if __name__ == "__main__":
    # 4 cities with Euclidean distances (triangle inequality holds)
    import math

    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist[i][j] = math.sqrt(dx * dx + dy * dy)

    cost, tour = metric_tsp_2approx(n, dist)
    opt = 4.0  # Square perimeter
    print(f"Tour: {tour}")
    print(f"Tour cost: {cost:.4f}")
    print(f"Optimal:   {opt:.4f}")
    print(f"Ratio:     {cost / opt:.4f}")
```

## Summary

| Algorithm | Ratio | Time |
|---|---|---|
| MST-doubling | $2$ | $O(n^2)$ |
| Christofides-Serdyukov | $3/2$ | $O(n^3)$ |
| Karlin-Klein-OG (2021) | $3/2 - \delta$ | Polynomial |

The triangle inequality is essential: without it, no polynomial-time algorithm
can achieve any constant approximation ratio unless P = NP.

## Reference

- Christofides, N. "Worst-Case Analysis of a New Heuristic for the Travelling
  Salesman Problem." Technical Report 388, CMU, 1976.
- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001. Chapter 3.
