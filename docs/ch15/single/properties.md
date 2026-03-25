# Shortest Path Properties

Every shortest path algorithm relies on a small set of structural properties
that govern how shortest paths behave.  These properties explain *why*
relaxation-based algorithms converge and *when* a shortest path is guaranteed to
exist.  Internalizing them makes correctness proofs for Dijkstra, Bellman-Ford,
and DAG shortest paths nearly mechanical.

## Optimal Substructure

The most fundamental property of shortest paths is that every sub-path of a
shortest path is itself a shortest path.

!!! note "Optimal Substructure of Shortest Paths"
    Let $p = \langle v_0, v_1, \dots, v_k \rangle$ be a shortest path from
    $v_0$ to $v_k$ in a weighted directed graph $G = (V, E)$ with weight
    function $w$.  For any $0 \le i \le j \le k$, the sub-path
    $p_{ij} = \langle v_i, v_{i+1}, \dots, v_j \rangle$ is a shortest path
    from $v_i$ to $v_j$.

**Proof by contradiction.**  Suppose there exists a path $p'_{ij}$ from $v_i$
to $v_j$ with $w(p'_{ij}) < w(p_{ij})$.  Then replacing $p_{ij}$ with
$p'_{ij}$ in $p$ produces a path from $v_0$ to $v_k$ with weight strictly less
than $w(p)$, contradicting the assumption that $p$ is a shortest path.
$\square$

This property is what allows dynamic programming and greedy approaches to
compute shortest paths: the optimal solution decomposes into optimal
sub-solutions.

## Shortest-Path Weight

Let $\delta(u, v)$ denote the **shortest-path weight** from $u$ to $v$:

$$
\delta(u, v) =
\begin{cases}
\min\{w(p) : u \xrightarrow{p} v\} & \text{if a path from } u \text{ to } v \text{ exists} \\
\infty & \text{otherwise}
\end{cases}
$$

When a negative-weight cycle is reachable from $u$ on a path to $v$, the
shortest-path weight is defined as $\delta(u, v) = -\infty$ because the cycle
can be traversed arbitrarily many times.

## Triangle Inequality

For any edge $(u, v) \in E$:

$$
\delta(s, v) \le \delta(s, u) + w(u, v)
$$

The shortest path to $v$ cannot be longer than the shortest path to $u$
followed by the direct edge $(u, v)$.  If it were, the concatenation would be a
shorter path to $v$, contradicting the definition of $\delta$.

## Upper-Bound Property

After calling `INITIALIZE-SINGLE-SOURCE(G, s)`, for every vertex $v \in V$:

$$
d[v] \ge \delta(s, v)
$$

This invariant holds throughout the execution of any relaxation-based
algorithm, and once $d[v] = \delta(s, v)$, the value never changes.

**Proof.**  Initially $d[s] = 0 = \delta(s, s)$ and $d[v] = \infty \ge
\delta(s, v)$ for $v \ne s$.  Each call to $\textsc{Relax}(u, v, w)$ sets
$d[v] \leftarrow d[u] + w(u, v)$ only when the new value is smaller.  By
induction, if $d[u] \ge \delta(s, u)$ before the relaxation, then

$$
d[v] = d[u] + w(u, v) \ge \delta(s, u) + w(u, v) \ge \delta(s, v)
$$

where the last step follows from the triangle inequality. $\square$

## No-Path Property

If there is no path from $s$ to $v$, then $\delta(s, v) = \infty$.  By the
upper-bound property, $d[v] \ge \delta(s, v) = \infty$, so $d[v] = \infty$
throughout the algorithm.  This means the algorithm naturally handles
unreachable vertices without any special case.

## Convergence Property

If $s \leadsto u \to v$ is a shortest path and $d[u] = \delta(s, u)$ at any
point before edge $(u, v)$ is relaxed, then $d[v] = \delta(s, v)$ after the
relaxation, and this value does not change thereafter.

This property is the engine that drives Dijkstra's algorithm: when vertex $u$
is extracted from the priority queue with $d[u] = \delta(s, u)$, relaxing all
edges leaving $u$ correctly sets the distance for each neighbor (provided
non-negative weights).

## Path-Relaxation Property

!!! note "Path-Relaxation Property"
    Let $p = \langle v_0, v_1, \dots, v_k \rangle$ be a shortest path from
    $s = v_0$ to $v_k$.  If the edges $(v_0, v_1), (v_1, v_2), \dots,
    (v_{k-1}, v_k)$ are relaxed in this order (with possibly other
    relaxations interspersed), then $d[v_k] = \delta(s, v_k)$.

This property unifies the correctness of all three single-source algorithms:

- **Bellman-Ford:** After pass $i$, all shortest paths using at most $i$ edges
  have been correctly computed.  Since a shortest path has at most
  $|V| - 1$ edges, $|V| - 1$ passes suffice.
- **DAG shortest paths:** Topological order guarantees that edge $(u, v)$ is
  relaxed after $d[u]$ has reached $\delta(s, u)$.
- **Dijkstra:** The greedy extraction order ensures that when $u$ is processed,
  $d[u] = \delta(s, u)$, so relaxing outgoing edges from $u$ correctly
  propagates distances.

## Predecessor-Subgraph Property

The predecessor pointers $\pi[v]$ define a **predecessor subgraph**
$G_\pi = (V_\pi, E_\pi)$ where:

$$
V_\pi = \{s\} \cup \{v \in V : \pi[v] \ne \text{NIL}\}
$$

$$
E_\pi = \{(\pi[v], v) : v \in V_\pi \setminus \{s\}\}
$$

After a shortest-path algorithm terminates, $G_\pi$ is a **shortest-path
tree**: a rooted tree at $s$ such that the unique path from $s$ to any
reachable vertex $v$ in $G_\pi$ is a shortest path in $G$.

## Summary of Properties

| Property | Statement | Key consequence |
|---|---|---|
| Optimal substructure | Sub-paths of shortest paths are shortest paths | Enables DP and greedy approaches |
| Triangle inequality | $\delta(s,v) \le \delta(s,u) + w(u,v)$ | Relaxation is safe |
| Upper-bound | $d[v] \ge \delta(s,v)$ always | Estimates only improve |
| No-path | Unreachable $\Rightarrow$ $d[v] = \infty$ forever | No special handling needed |
| Convergence | Correct $d[u]$ + relax $(u,v)$ $\Rightarrow$ correct $d[v]$ | Drives Dijkstra |
| Path-relaxation | Relax edges in shortest-path order $\Rightarrow$ correct | Unifies all algorithms |
| Predecessor subgraph | $G_\pi$ is a shortest-path tree | Path reconstruction |

## Implementation

```python
"""
Shortest path properties demonstration.

Verifies the key invariants (upper-bound, triangle inequality,
convergence) by running a step-by-step relaxation and checking
the properties after each operation.
"""

from math import inf


# === Graph setup =============================================================

def build_graph():
    """Return a sample weighted directed graph as an adjacency list.

    Graph:
        s --10--> a --1--> c
        s --5---> b --3--> a
                  b --8--> c
    """
    return {
        "s": [("a", 10), ("b", 5)],
        "a": [("c", 1)],
        "b": [("a", 3), ("c", 8)],
        "c": [],
    }


# === Relaxation with property checks ========================================

def initialize(vertices, source):
    """Initialize distance estimates and predecessors."""
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}
    return dist, pred


def relax(u, v, w, dist, pred):
    """Relax edge (u, v) and return whether an update occurred."""
    if dist[u] + w < dist[v]:
        dist[v] = dist[u] + w
        pred[v] = u
        return True
    return False


def verify_upper_bound(dist, true_dist):
    """Check that d[v] >= delta(s,v) for all vertices."""
    for v in dist:
        assert dist[v] >= true_dist[v], (
            f"Upper-bound violated: d[{v}]={dist[v]} < delta={true_dist[v]}"
        )
    return True


def verify_triangle_inequality(true_dist, graph):
    """Check delta(s,v) <= delta(s,u) + w(u,v) for all edges."""
    for u in graph:
        for v, w in graph[u]:
            assert true_dist[v] <= true_dist[u] + w, (
                f"Triangle inequality violated for edge ({u},{v})"
            )
    return True


# === Main ====================================================================

if __name__ == "__main__":
    graph = build_graph()
    vertices = list(graph.keys())

    # True shortest distances (precomputed)
    true_dist = {"s": 0, "a": 8, "b": 5, "c": 9}

    # Verify triangle inequality on true distances
    assert verify_triangle_inequality(true_dist, graph)
    print("Triangle inequality: VERIFIED")

    # Run relaxation and check upper-bound after each step
    dist, pred = initialize(vertices, "s")
    edges = [("s", "a", 10), ("s", "b", 5), ("b", "a", 3),
             ("a", "c", 1), ("b", "c", 8)]

    for u, v, w in edges:
        relax(u, v, w, dist, pred)
        assert verify_upper_bound(dist, true_dist)
        print(f"After relaxing ({u},{v}): d = {dict(dist)}  "
              f"Upper-bound: VERIFIED")

    # Check convergence: final distances match true distances
    assert dist == true_dist
    print(f"\nConvergence: VERIFIED — final distances match true shortest paths")
```

**Output:**

```
Triangle inequality: VERIFIED
After relaxing (s,a): d = {'s': 0, 'a': 10, 'b': inf, 'c': inf}  Upper-bound: VERIFIED
After relaxing (s,b): d = {'s': 0, 'a': 10, 'b': 5, 'c': inf}  Upper-bound: VERIFIED
After relaxing (b,a): d = {'s': 0, 'a': 8, 'b': 5, 'c': inf}  Upper-bound: VERIFIED
After relaxing (a,c): d = {'s': 0, 'a': 8, 'b': 5, 'c': 9}  Upper-bound: VERIFIED
After relaxing (b,c): d = {'s': 0, 'a': 8, 'b': 5, 'c': 9}  Upper-bound: VERIFIED

Convergence: VERIFIED — final distances match true shortest paths
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24: Single-Source Shortest Paths.
