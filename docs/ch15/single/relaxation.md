# Edge Relaxation

Shortest path algorithms share a common mechanism: they maintain a tentative
distance estimate for each vertex and repeatedly improve these estimates by
examining edges.  This operation, called **relaxation**, is the fundamental
building block of Dijkstra's algorithm, Bellman-Ford, and DAG shortest paths.
Understanding relaxation in isolation clarifies why each algorithm works and
what differentiates them.

## Initialization

Before any relaxation occurs, every vertex receives a tentative distance and a
predecessor pointer.  The source vertex starts at distance zero; all others
start at infinity.

For a graph $G = (V, E)$ with source $s$:

$$
d[v] =
\begin{cases}
0 & \text{if } v = s \\
\infty & \text{otherwise}
\end{cases}
\quad \text{and} \quad
\pi[v] = \text{NIL} \quad \forall\, v \in V
$$

Here $d[v]$ is the current shortest-path estimate and $\pi[v]$ records the
predecessor of $v$ on the best known path from $s$.

## The Relax Operation

Given an edge $(u, v)$ with weight $w(u, v)$, relaxation checks whether the
path through $u$ improves the current estimate for $v$.  If the path
$s \leadsto u \to v$ is shorter than the best path to $v$ found so far, the
estimate and predecessor are updated.

$$
\textsc{Relax}(u, v, w): \quad
\text{if } d[v] > d[u] + w(u, v) \text{ then }
\begin{cases}
d[v] \leftarrow d[u] + w(u, v) \\
\pi[v] \leftarrow u
\end{cases}
$$

The operation is safe because it only decreases distance estimates — it never
increases them.  After relaxation, the invariant $d[v] \le d[u] + w(u, v)$
holds for the edge $(u, v)$.

## Why Relaxation Is Correct

Relaxation preserves several key properties that guarantee shortest path
algorithms converge to the correct answer.

### Upper-Bound Property

At all times, $d[v] \ge \delta(s, v)$ where $\delta(s, v)$ is the true
shortest-path weight.  Relaxation only decreases $d[v]$, and the condition
$d[v] > d[u] + w(u, v)$ ensures the new value equals $d[u] + w(u, v)$, which
is at least $\delta(s, v)$ by the triangle inequality.

### Convergence Property

If $d[u] = \delta(s, u)$ at the time edge $(u, v)$ is relaxed, then after
relaxation $d[v] \le \delta(s, v)$.  Combined with the upper-bound property,
this means $d[v] = \delta(s, v)$ — the estimate is exact.

### Path-Relaxation Property

If $p = \langle v_0, v_1, \dots, v_k \rangle$ is a shortest path from
$s = v_0$ to $v_k$, and the edges $(v_0, v_1), (v_1, v_2), \dots,
(v_{k-1}, v_k)$ are relaxed in this order (possibly with other relaxations
interspersed), then $d[v_k] = \delta(s, v_k)$.

This property is what makes each algorithm work:

| Algorithm | How it ensures the relaxation order |
|---|---|
| **Dijkstra** | Greedy extraction from a priority queue guarantees $d[u] = \delta(s, u)$ when $u$ is processed |
| **Bellman-Ford** | $\lvert V \rvert - 1$ passes over all edges cover every possible shortest path length |
| **DAG shortest paths** | Topological order guarantees all predecessors are finalized first |

## The Triangle Inequality

For any edge $(u, v) \in E$, the shortest-path distances satisfy:

$$
\delta(s, v) \le \delta(s, u) + w(u, v)
$$

If this were violated, the path $s \leadsto u \to v$ would be shorter than
$\delta(s, v)$, contradicting the definition of shortest-path weight.
Relaxation exploits this inequality directly.

## Worked Example

Consider a graph with four vertices and the following edges:

| Edge | Weight |
|---|---|
| $(s, a)$ | 10 |
| $(s, b)$ | 5 |
| $(b, a)$ | 3 |
| $(a, c)$ | 1 |
| $(b, c)$ | 8 |

**After initialization:** $d[s] = 0$, $d[a] = \infty$, $d[b] = \infty$,
$d[c] = \infty$.

**Relax $(s, a)$:** $d[a] = \infty > 0 + 10 = 10$, so $d[a] \leftarrow 10$,
$\pi[a] \leftarrow s$.

**Relax $(s, b)$:** $d[b] = \infty > 0 + 5 = 5$, so $d[b] \leftarrow 5$,
$\pi[b] \leftarrow s$.

**Relax $(b, a)$:** $d[a] = 10 > 5 + 3 = 8$, so $d[a] \leftarrow 8$,
$\pi[a] \leftarrow b$.

**Relax $(a, c)$:** $d[c] = \infty > 8 + 1 = 9$, so $d[c] \leftarrow 9$,
$\pi[c] \leftarrow a$.

**Relax $(b, c)$:** $d[c] = 9 \not> 5 + 8 = 13$, so no update — the existing
path $s \to b \to a \to c$ with weight 9 is already better.

## Implementation

```python
"""
Edge relaxation for single-source shortest paths.

Demonstrates the RELAX operation that forms the core of Dijkstra's
algorithm, Bellman-Ford, and DAG shortest paths.
"""

from math import inf


# === Initialization =========================================================

def initialize_single_source(vertices: list, source) -> tuple[dict, dict]:
    """Set up distance estimates and predecessors for shortest path search.

    Parameters
    ----------
    vertices : list
        All vertex identifiers in the graph.
    source : hashable
        The source vertex.

    Returns
    -------
    dist : dict
        Tentative distances (0 for source, inf for all others).
    pred : dict
        Predecessor pointers (None for all vertices initially).
    """
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}
    return dist, pred


# === Relaxation ==============================================================

def relax(u, v, weight: float, dist: dict, pred: dict) -> bool:
    """Relax edge (u, v) with the given weight.

    If the path through u improves the distance estimate for v,
    update dist[v] and pred[v].

    Returns True if the estimate was improved, False otherwise.
    """
    if dist[u] + weight < dist[v]:
        dist[v] = dist[u] + weight
        pred[v] = u
        return True
    return False


# === Path reconstruction =====================================================

def reconstruct_path(pred: dict, source, target) -> list:
    """Trace the predecessor chain from target back to source."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()
    if path[0] != source:
        return []  # target not reachable
    return path


# === Demo ====================================================================

if __name__ == "__main__":
    # Graph: s -> a (10), s -> b (5), b -> a (3), a -> c (1), b -> c (8)
    vertices = ["s", "a", "b", "c"]
    edges = [("s", "a", 10), ("s", "b", 5), ("b", "a", 3),
             ("a", "c", 1), ("b", "c", 8)]

    dist, pred = initialize_single_source(vertices, "s")
    print(f"After init: {dist}")

    # Relax edges in a specific order to illustrate the operation
    for u, v, w in edges:
        changed = relax(u, v, w, dist, pred)
        status = "updated" if changed else "no change"
        print(f"Relax ({u},{v},w={w}): d[{v}]={dist[v]}, {status}")

    print(f"\nFinal distances: {dist}")
    print(f"Path s->c: {reconstruct_path(pred, 's', 'c')}")
```

**Output:**

```
After init: {'s': 0, 'a': inf, 'b': inf, 'c': inf}
Relax (s,a,w=10): d[a]=10, updated
Relax (s,b,w=5): d[b]=5, updated
Relax (b,a,w=3): d[a]=8, updated
Relax (a,c,w=1): d[c]=9, updated
Relax (b,c,w=8): d[c]=9, no change
Final distances: {'s': 0, 'a': 8, 'b': 5, 'c': 9}
Path s->c: ['s', 'b', 'a', 'c']
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24: Single-Source Shortest Paths.
