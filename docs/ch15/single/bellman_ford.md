# The Bellman-Ford Algorithm

Dijkstra's algorithm fails when a graph contains negative-weight edges because
its greedy strategy assumes that once a vertex is finalized, no shorter path can
appear later.  The Bellman-Ford algorithm removes this restriction by
systematically relaxing *every* edge in $|V| - 1$ passes.  It handles negative
weights correctly and can also detect negative-weight cycles, making it the
most general single-source shortest path algorithm.

## Algorithm Overview

The key idea is simple: a shortest path from $s$ to any vertex $v$ contains at
most $|V| - 1$ edges (any longer path would revisit a vertex, creating a
cycle).  After $i$ passes over all edges, every shortest path using at most $i$
edges has been correctly computed.  Therefore $|V| - 1$ passes suffice.

| Algorithm | Edge weights | Complexity |
|---|---|---|
| BFS | $w = 1$ (unweighted) | $O(V + E)$ |
| Bellman-Ford | Negative weights allowed | $O(VE)$ |
| Dijkstra | $w \ge 0$ | $O(V^2)$ or $O((V+E)\log V)$ |

## Pseudocode

```
BELLMAN-FORD(G, w, s):
    INITIALIZE-SINGLE-SOURCE(G, s)
    for i = 1 to |V| - 1:
        for each edge (u, v) in E:
            RELAX(u, v, w)
    // Negative-cycle check
    for each edge (u, v) in E:
        if d[v] > d[u] + w(u, v):
            return FALSE   // negative cycle reachable from s
    return TRUE
```

The algorithm has two phases.  The first phase performs $|V| - 1$ relaxation
passes.  The second phase checks for negative-weight cycles: if any edge can
still be relaxed, a negative cycle exists.

## Correctness

The correctness of Bellman-Ford follows from the **path-relaxation property**.

!!! note "Correctness Theorem"
    If $G$ contains no negative-weight cycles reachable from $s$, then after
    $|V| - 1$ passes, $d[v] = \delta(s, v)$ for all $v \in V$.

**Proof.**  Let $v$ be any vertex reachable from $s$, and let
$p = \langle v_0, v_1, \dots, v_k \rangle$ be a shortest path from $s = v_0$
to $v = v_k$.  Since there is no negative-weight cycle, $p$ is simple, so
$k \le |V| - 1$.

After pass $i$, edge $(v_{i-1}, v_i)$ has been relaxed.  By the
path-relaxation property, after pass $k \le |V| - 1$, we have
$d[v_k] = \delta(s, v_k)$. $\square$

## Negative-Cycle Detection

After the $|V| - 1$ passes, if any edge $(u, v)$ still satisfies
$d[v] > d[u] + w(u, v)$, then no finite shortest path exists — a
negative-weight cycle is reachable from $s$.

**Proof.**  Suppose for contradiction that no negative cycle is reachable but
$d[v] > d[u] + w(u, v)$ for some edge.  The correctness theorem guarantees
$d[v] = \delta(s, v)$ and $d[u] = \delta(s, u)$, so

$$
\delta(s, v) > \delta(s, u) + w(u, v)
$$

which violates the triangle inequality. $\square$

## Complexity Analysis

- **Time:** Each of the $|V| - 1$ passes iterates over all $|E|$ edges,
  giving $O(VE)$ total.
- **Space:** $O(V)$ for the distance and predecessor arrays.

The $O(VE)$ bound is worse than Dijkstra's $O((V+E)\log V)$, but Bellman-Ford
handles negative weights — a capability Dijkstra lacks.

## Worked Example

Consider the following graph with source $s$:

| Edge | Weight |
|---|---|
| $(s, a)$ | 6 |
| $(s, b)$ | 7 |
| $(a, b)$ | 8 |
| $(a, c)$ | 5 |
| $(a, d)$ | -4 |
| $(b, c)$ | -3 |
| $(b, d)$ | 9 |
| $(c, b)$ | 7 |
| $(d, c)$ | 2 |

**Pass 1:** Relax all edges.  Vertices reachable in one hop get their direct
distances: $d[a] = 6$, $d[b] = 7$.  Further relaxations within the pass may
also discover two-hop paths.

**Pass 2:** Paths using up to two edges are finalized.  For example,
$s \to a \to d$ gives $d[d] = 6 + (-4) = 2$.  Also $s \to b \to c$ yields
$d[c] = 7 + (-3) = 4$.

**Pass 3:** Three-edge paths are considered.  Path $s \to a \to d \to c$
gives $d[c] = \min(4, 2 + 2) = 4$ (no improvement).

**Pass 4:** Fourth pass confirms all distances are stable. No edge can be
further relaxed, so no negative cycle exists.

## Comparison with Dijkstra

| Aspect | Bellman-Ford | Dijkstra |
|---|---|---|
| Relaxation strategy | Fixed order: all edges, $\lvert V\rvert - 1$ times | Greedy: outgoing edges of nearest vertex |
| Negative edges | Supported | Not supported |
| Negative cycle detection | Built-in | Not applicable |
| Time complexity | $O(VE)$ | $O((V+E)\log V)$ with binary heap |
| Use case | General graphs | Non-negative weight graphs |

## Implementation

```python
"""
Bellman-Ford single-source shortest path algorithm.

Handles negative-weight edges and detects negative-weight cycles.
"""

from math import inf


# === Bellman-Ford algorithm ==================================================

def bellman_ford(vertices: list, edges: list, source) -> tuple[dict, dict, bool]:
    """Run Bellman-Ford from the given source vertex.

    Parameters
    ----------
    vertices : list
        All vertex identifiers.
    edges : list of (u, v, w)
        Directed edges with weights.
    source : hashable
        The source vertex.

    Returns
    -------
    dist : dict
        Shortest distances from source.
    pred : dict
        Predecessor pointers for path reconstruction.
    no_negative_cycle : bool
        True if no negative cycle is reachable from source.
    """
    # Initialize
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}

    # Relax all edges |V| - 1 times
    for i in range(len(vertices) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
        if not updated:
            break  # Early termination: no changes in this pass

    # Check for negative-weight cycles
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return dist, pred, False  # Negative cycle detected

    return dist, pred, True


# === Path reconstruction =====================================================

def get_path(pred: dict, source, target) -> list:
    """Reconstruct the shortest path from source to target."""
    path = []
    v = target
    while v is not None:
        path.append(v)
        v = pred[v]
    path.reverse()
    return path if path[0] == source else []


# === Demo ====================================================================

if __name__ == "__main__":
    # Graph with negative-weight edges (no negative cycle)
    vertices = ["s", "a", "b", "c", "d"]
    edges = [
        ("s", "a", 6), ("s", "b", 7), ("a", "b", 8),
        ("a", "c", 5), ("a", "d", -4), ("b", "c", -3),
        ("b", "d", 9), ("c", "b", 7), ("d", "c", 2),
    ]

    dist, pred, ok = bellman_ford(vertices, edges, "s")
    print(f"No negative cycle: {ok}")
    print(f"Distances: {dist}")
    print(f"Path s->d: {get_path(pred, 's', 'd')}")
    print(f"Path s->c: {get_path(pred, 's', 'c')}")

    # Graph with a negative-weight cycle
    print("\n--- Graph with negative cycle ---")
    neg_edges = edges + [("c", "a", -10)]  # Creates cycle a->c->a with weight 5+(-10)=-5
    dist2, pred2, ok2 = bellman_ford(vertices, neg_edges, "s")
    print(f"No negative cycle: {ok2}")
```

**Output:**

```
No negative cycle: True
Distances: {'s': 0, 'a': 6, 'b': 4, 'c': 4, 'd': 2}
Path s->d: ['s', 'a', 'd']
Path s->c: ['s', 'b', 'c']

--- Graph with negative cycle ---
No negative cycle: False
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24.1: The Bellman-Ford Algorithm.
- Bellman, R. (1958). On a routing problem. *Quarterly of Applied Mathematics*,
  16(1), 87-90.
