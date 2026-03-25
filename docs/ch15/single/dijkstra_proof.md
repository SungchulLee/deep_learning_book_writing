# Correctness of Dijkstra's Algorithm

Dijkstra's algorithm computes shortest paths by greedily extracting the vertex
with the smallest tentative distance and relaxing its outgoing edges.  The
correctness of this greedy strategy is not obvious: why should the vertex with
the smallest estimate actually have the correct shortest-path distance?  This
page presents the formal proof, which relies on the non-negativity of edge
weights and the upper-bound property of relaxation.

## Algorithm Recap

Dijkstra's algorithm maintains a set $S$ of vertices whose shortest-path
distances have been finalized.  At each step, it extracts the vertex
$u \in V \setminus S$ with the minimum $d[u]$, adds $u$ to $S$, and relaxes
all edges leaving $u$.

```
DIJKSTRA(G, w, s):
    INITIALIZE-SINGLE-SOURCE(G, s)
    S = {}
    Q = V            // min-priority queue keyed by d[v]
    while Q is not empty:
        u = EXTRACT-MIN(Q)
        S = S ∪ {u}
        for each edge (u, v) in Adj[u]:
            RELAX(u, v, w)
```

## Correctness Theorem

!!! note "Theorem: Dijkstra Correctness"
    If all edge weights are non-negative ($w(u, v) \ge 0$ for all
    $(u, v) \in E$), then at the time each vertex $u$ is extracted from
    the priority queue, $d[u] = \delta(s, u)$.

## Proof by Contradiction

**Proof.**  Suppose for contradiction that $u$ is the **first vertex** added to
$S$ for which $d[u] \ne \delta(s, u)$.

**Step 1: $u \ne s$.**  The source $s$ is added first with $d[s] = 0 =
\delta(s, s)$, so $u$ is some other vertex.

**Step 2: $u$ is reachable.**  Since $u$ is extracted from the queue with
$d[u] < \infty$ (otherwise the condition $d[u] \ne \delta(s, u)$ implies
$\delta(s, u) < \infty$, meaning a path exists), there is a shortest path from
$s$ to $u$.

**Step 3: Identify the critical edge.**  Let $p$ be a shortest path from $s$
to $u$.  Consider the first edge $(x, y)$ on $p$ such that $x \in S$ and
$y \notin S$ at the moment $u$ is about to be added.  Such an edge must exist
because $s \in S$ and $u \notin S$.

$$
s \xrightarrow{p_1} x \to y \xrightarrow{p_2} u
$$

**Step 4: Show $d[y] = \delta(s, y)$.**  Since $x$ was added to $S$ before
$u$, and $u$ is the *first* vertex with an incorrect estimate, we have
$d[x] = \delta(s, x)$.  When $x$ was added to $S$, edge $(x, y)$ was relaxed.
By the convergence property:

$$
d[y] \le d[x] + w(x, y) = \delta(s, x) + w(x, y) = \delta(s, y)
$$

The last equality holds because $p$ is a shortest path and
$s \leadsto x \to y$ is its sub-path (optimal substructure).  Combined with the
upper-bound property $d[y] \ge \delta(s, y)$, we get $d[y] = \delta(s, y)$.

**Step 5: Derive the contradiction.**  Since all edge weights are non-negative,
the sub-path $y \leadsto u$ has non-negative weight, so:

$$
\delta(s, y) \le \delta(s, u)
$$

Therefore:

$$
d[y] = \delta(s, y) \le \delta(s, u) \le d[u]
$$

But $u$ was chosen by `EXTRACT-MIN`, so $d[u] \le d[y]$.  Combining:

$$
d[u] \le d[y] = \delta(s, y) \le \delta(s, u) \le d[u]
$$

This forces $d[u] = \delta(s, u)$, contradicting our assumption. $\square$

## Why Non-Negative Weights Are Essential

The proof breaks at **Step 5**: if edges can have negative weights, the
sub-path $y \leadsto u$ might have negative total weight, allowing
$\delta(s, u) < \delta(s, y)$.  In that case, extracting $u$ before $y$ does
not guarantee $d[u] = \delta(s, u)$.

??? example "Counterexample with negative weights"
    Consider vertices $\{s, a, b\}$ with edges $(s, a, 3)$, $(s, b, 5)$,
    $(b, a, -4)$.  Dijkstra extracts $a$ first with $d[a] = 3$, but the
    true shortest path $s \to b \to a$ has weight $5 + (-4) = 1 < 3$.
    The algorithm produces the wrong answer because it finalizes $a$ too
    early.

## Complexity Analysis

The time complexity depends on the priority queue implementation:

| Priority queue | `EXTRACT-MIN` | `DECREASE-KEY` | Total |
|---|---|---|---|
| Array (unsorted) | $O(V)$ | $O(1)$ | $O(V^2)$ |
| Binary heap | $O(\log V)$ | $O(\log V)$ | $O((V+E)\log V)$ |
| Fibonacci heap | $O(\log V)$ amortized | $O(1)$ amortized | $O(V\log V + E)$ |

For sparse graphs ($E = O(V)$), the binary heap gives $O(V \log V)$.  For
dense graphs ($E = O(V^2)$), the simple array gives $O(V^2)$, which is optimal.

## Implementation

```python
"""
Dijkstra's algorithm with correctness verification.

Demonstrates the algorithm and verifies that the greedy extraction
order produces correct shortest-path distances.
"""

import heapq
from math import inf


# === Dijkstra's algorithm ====================================================

def dijkstra(graph: dict, source) -> tuple[dict, dict]:
    """Compute shortest paths from source using Dijkstra's algorithm.

    Parameters
    ----------
    graph : dict
        Adjacency list mapping vertex -> list of (neighbor, weight).
        All weights must be non-negative.
    source : hashable
        The source vertex.

    Returns
    -------
    dist : dict
        Shortest distances from source.
    pred : dict
        Predecessor pointers for path reconstruction.
    """
    dist = {v: inf for v in graph}
    dist[source] = 0
    pred = {v: None for v in graph}
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue  # stale entry
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                heapq.heappush(pq, (dist[v], v))

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
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: [],
    }

    dist, pred = dijkstra(graph, 0)
    print(f"Distances: {dist}")
    print(f"Path 0->3: {get_path(pred, 0, 3)}")
    print(f"Path 0->1: {get_path(pred, 0, 1)}")

    # Verify greedy correctness: extraction order
    print("\n--- Extraction order verification ---")
    finalized = {}
    dist2 = {v: inf for v in graph}
    dist2[0] = 0
    pq = [(0, 0)]
    step = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist2[u]:
            continue
        finalized[u] = dist2[u]
        step += 1
        print(f"Step {step}: extract vertex {u} with d[{u}] = {dist2[u]}")
        for v, w in graph[u]:
            if dist2[u] + w < dist2[v]:
                dist2[v] = dist2[u] + w
                heapq.heappush(pq, (dist2[v], v))
    print(f"Final: {finalized}")
```

**Output:**

```
Distances: {0: 0, 1: 3, 2: 1, 3: 4}
Path 0->3: [0, 2, 1, 3]
Path 0->1: [0, 2, 1]

--- Extraction order verification ---
Step 1: extract vertex 0 with d[0] = 0
Step 2: extract vertex 2 with d[2] = 1
Step 3: extract vertex 1 with d[1] = 3
Step 4: extract vertex 3 with d[3] = 4
Final: {0: 0, 2: 1, 1: 3, 3: 4}
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24.3: Dijkstra's Algorithm.
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
  *Numerische Mathematik*, 1, 269-271.
