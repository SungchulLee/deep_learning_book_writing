# DAG Shortest Paths

When a graph has no cycles, shortest paths become dramatically simpler to
compute.  A directed acyclic graph (DAG) admits a topological ordering of its
vertices, and processing vertices in this order guarantees that every
predecessor of a vertex $v$ has already been finalized before $v$ is reached.
This eliminates the need for Dijkstra's priority queue or Bellman-Ford's
repeated passes, yielding a clean $O(V + E)$ algorithm that also handles
negative-weight edges without difficulty.

## Why Topological Order Works

In a DAG, a topological sort produces a linear ordering
$v_1, v_2, \dots, v_n$ of all vertices such that for every edge
$(v_i, v_j)$, we have $i < j$.  This means that when we process vertex
$v_j$, every vertex that could be a predecessor of $v_j$ on a shortest
path has already been processed and its distance finalized.

By the **convergence property**, if $d[u] = \delta(s, u)$ when edge $(u, v)$
is relaxed, then $d[v] = \delta(s, v)$ afterward.  Topological order
guarantees exactly this precondition for every edge.

## Algorithm

```
DAG-SHORTEST-PATHS(G, w, s):
    topological sort G
    INITIALIZE-SINGLE-SOURCE(G, s)
    for each vertex u in topological order:
        for each edge (u, v) in Adj[u]:
            RELAX(u, v, w)
```

Each vertex is processed exactly once, and each edge is relaxed exactly once.

## Correctness

!!! note "Correctness Theorem"
    After `DAG-SHORTEST-PATHS` terminates, $d[v] = \delta(s, v)$ for all
    $v \in V$.

**Proof.**  Let $p = \langle v_0, v_1, \dots, v_k \rangle$ be a shortest path
from $s = v_0$ to some vertex $v_k$.  In topological order, $v_0$ appears
before $v_1$, which appears before $v_2$, and so on.  Therefore edge
$(v_0, v_1)$ is relaxed before $(v_1, v_2)$, which is relaxed before
$(v_2, v_3)$, and so on.

By the path-relaxation property, $d[v_k] = \delta(s, v_k)$ after all edges of
$p$ have been relaxed. $\square$

## Complexity

- **Time:** Topological sort takes $O(V + E)$.  The main loop processes each
  vertex once and each edge once, also $O(V + E)$.  Total: $O(V + E)$.
- **Space:** $O(V)$ for distance and predecessor arrays, plus $O(V + E)$ for
  the graph representation.

This is asymptotically optimal — any algorithm must examine every edge at
least once.

## Comparison with Other Algorithms

| Algorithm | Handles negative weights | Requires DAG | Time |
|---|---|---|---|
| DAG shortest paths | Yes | Yes | $O(V + E)$ |
| Dijkstra | No | No | $O((V+E)\log V)$ |
| Bellman-Ford | Yes | No | $O(VE)$ |

The DAG algorithm is the fastest but applies only to acyclic graphs.

## Longest Path in a DAG

A useful variation: to find the **longest path** in a DAG, negate all edge
weights and run DAG shortest paths.  Alternatively, replace the relaxation
condition with $d[v] < d[u] + w(u, v)$ and initialize distances to $-\infty$.
This is useful in critical-path analysis for project scheduling (PERT/CPM).

## Worked Example

Consider the DAG with vertices in topological order $\langle s, a, b, c, d, e \rangle$:

| Edge | Weight |
|---|---|
| $(s, a)$ | 5 |
| $(s, b)$ | 3 |
| $(a, b)$ | 2 |
| $(a, c)$ | 6 |
| $(b, c)$ | 7 |
| $(b, d)$ | 4 |
| $(c, d)$ | -1 |
| $(c, e)$ | 1 |
| $(d, e)$ | -2 |

**Processing $s$:** Relax $(s, a)$: $d[a] = 5$.  Relax $(s, b)$: $d[b] = 3$.

**Processing $a$:** Relax $(a, b)$: $d[b] = \min(3, 5+2) = 3$ (no change).
Relax $(a, c)$: $d[c] = 11$.

**Processing $b$:** Relax $(b, c)$: $d[c] = \min(11, 3+7) = 10$.  Relax
$(b, d)$: $d[d] = 7$.

**Processing $c$:** Relax $(c, d)$: $d[d] = \min(7, 10-1) = 7$ (no change).
Relax $(c, e)$: $d[e] = 11$.

**Processing $d$:** Relax $(d, e)$: $d[e] = \min(11, 7-2) = 5$.

**Final distances:** $d[s]=0, d[a]=5, d[b]=3, d[c]=10, d[d]=7, d[e]=5$.

## Implementation

```python
"""
DAG shortest paths using topological sort.

Computes single-source shortest paths in O(V + E) time for directed
acyclic graphs, handling negative-weight edges correctly.
"""

from math import inf
from collections import defaultdict, deque


# === Topological sort (Kahn's algorithm) =====================================

def topological_sort(graph: dict, vertices: list) -> list:
    """Return vertices in topological order using Kahn's algorithm.

    Parameters
    ----------
    graph : dict
        Adjacency list mapping vertex -> list of (neighbor, weight).
    vertices : list
        All vertex identifiers.

    Returns
    -------
    list
        Vertices in topological order.

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    in_degree = defaultdict(int)
    for v in vertices:
        in_degree[v]  # ensure every vertex appears
    for u in graph:
        for v, _ in graph[u]:
            in_degree[v] += 1

    queue = deque(v for v in vertices if in_degree[v] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _ in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(vertices):
        raise ValueError("Graph contains a cycle")
    return order


# === DAG shortest paths ======================================================

def dag_shortest_paths(graph: dict, vertices: list, source) -> tuple[dict, dict]:
    """Compute shortest paths from source in a DAG.

    Parameters
    ----------
    graph : dict
        Adjacency list mapping vertex -> list of (neighbor, weight).
    vertices : list
        All vertex identifiers.
    source : hashable
        The source vertex.

    Returns
    -------
    dist : dict
        Shortest distances from source.
    pred : dict
        Predecessor pointers for path reconstruction.
    """
    order = topological_sort(graph, vertices)

    # Initialize
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}

    # Relax edges in topological order
    for u in order:
        if dist[u] == inf:
            continue  # u is unreachable; skip
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u

    return dist, pred


# === Path reconstruction =====================================================

def get_path(pred: dict, source, target) -> list:
    """Reconstruct the shortest path from source to target."""
    path = []
    v = target
    while v is not None:
        path.append(v)
        v = pred[v]
    path.reverse()
    return path if path and path[0] == source else []


# === Demo ====================================================================

if __name__ == "__main__":
    vertices = ["s", "a", "b", "c", "d", "e"]
    graph = {
        "s": [("a", 5), ("b", 3)],
        "a": [("b", 2), ("c", 6)],
        "b": [("c", 7), ("d", 4)],
        "c": [("d", -1), ("e", 1)],
        "d": [("e", -2)],
        "e": [],
    }

    dist, pred = dag_shortest_paths(graph, vertices, "s")
    print(f"Distances: {dist}")
    print(f"Path s->e: {get_path(pred, 's', 'e')}")
    print(f"Path s->c: {get_path(pred, 's', 'c')}")
    print(f"Path s->d: {get_path(pred, 's', 'd')}")
```

**Output:**

```
Distances: {'s': 0, 'a': 5, 'b': 3, 'c': 10, 'd': 7, 'e': 5}
Path s->e: ['s', 'b', 'd', 'e']
Path s->c: ['s', 'b', 'c']
Path s->d: ['s', 'b', 'd']
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24.2: Single-Source Shortest Paths in DAGs.
