# Negative Cycle Detection

A negative-weight cycle is a directed cycle whose edge weights sum to a negative
value.  When such a cycle is reachable from the source on a path to some vertex
$v$, the shortest-path weight $\delta(s, v) = -\infty$ because traversing the
cycle repeatedly decreases the path weight without bound.  Detecting these
cycles is essential: without detection, a shortest path algorithm may loop
forever or return meaningless results.

## What Is a Negative Cycle

A cycle $c = \langle v_0, v_1, \dots, v_k = v_0 \rangle$ in a weighted
directed graph is a **negative-weight cycle** if:

$$
w(c) = \sum_{i=0}^{k-1} w(v_i, v_{i+1}) < 0
$$

For any vertex $v$ reachable from such a cycle, no shortest path exists because
any path can be "improved" by looping around the cycle one more time.

## Detection via Bellman-Ford

The Bellman-Ford algorithm naturally detects negative cycles.  After
$|V| - 1$ relaxation passes, all shortest paths in a graph without negative
cycles have been correctly computed (since a simple shortest path has at most
$|V| - 1$ edges).  A $|V|$-th pass checks whether any edge can still be
relaxed:

```
NEGATIVE-CYCLE-CHECK(G, w, s):
    Run BELLMAN-FORD for |V| - 1 passes
    for each edge (u, v) in E:
        if d[v] > d[u] + w(u, v):
            return TRUE   // negative cycle exists
    return FALSE
```

### Correctness

!!! note "Detection Theorem"
    After $|V| - 1$ passes of Bellman-Ford, there exists an edge
    $(u, v)$ with $d[v] > d[u] + w(u, v)$ if and only if a
    negative-weight cycle is reachable from $s$.

**Proof ($\Rightarrow$).**  If a negative cycle
$c = \langle v_0, v_1, \dots, v_k = v_0 \rangle$ is reachable from $s$,
assume for contradiction that $d[v_i] \le d[v_{i-1}] + w(v_{i-1}, v_i)$ for
all edges in $c$.  Summing around the cycle:

$$
\sum_{i=1}^{k} d[v_i] \le \sum_{i=1}^{k} d[v_{i-1}] + \sum_{i=1}^{k} w(v_{i-1}, v_i)
$$

Since $v_k = v_0$, the left and right sums of $d$ values are identical,
giving:

$$
0 \le \sum_{i=1}^{k} w(v_{i-1}, v_i) = w(c) < 0
$$

This is a contradiction, so at least one edge in the cycle must still be
relaxable.

**Proof ($\Leftarrow$).**  If no negative cycle is reachable, the correctness
of Bellman-Ford guarantees $d[v] = \delta(s, v)$ for all $v$ after $|V| - 1$
passes.  By the triangle inequality,
$\delta(s, v) \le \delta(s, u) + w(u, v)$, so no edge can be relaxed
further. $\square$

## Extracting the Cycle

Detecting a negative cycle is useful, but often we need to identify the cycle
itself.  When the $|V|$-th pass finds an edge $(u, v)$ that can still be
relaxed, vertex $v$ lies on or is reachable from a negative cycle.  To extract
the cycle:

1. From $v$, follow predecessor pointers $|V|$ times to ensure we are inside
   the cycle (not just on a path leading to it).
2. From that vertex, follow predecessors until we revisit a vertex, which
   gives us the cycle.

## Applications

Negative cycles arise in several practical contexts:

- **Currency arbitrage:** In a foreign exchange graph where edge weights are
  $-\log(\text{exchange rate})$, a negative cycle corresponds to a sequence of
  trades that yields a profit.
- **Resource optimization:** In scheduling or routing problems, negative cycles
  may indicate opportunities to reduce total cost by restructuring.
- **Verification:** Proving that a constraint system has no feasible solution
  (via the connection between shortest paths and difference constraints).

## Worked Example

Consider vertices $\{s, a, b, c\}$ with edges:

| Edge | Weight |
|---|---|
| $(s, a)$ | 4 |
| $(a, b)$ | -2 |
| $(b, c)$ | 3 |
| $(c, a)$ | -5 |

The cycle $a \to b \to c \to a$ has weight $(-2) + 3 + (-5) = -4 < 0$.

**After 3 passes of Bellman-Ford** ($|V|-1 = 3$):

- $d[s] = 0$, $d[a] = 4$, but the cycle keeps reducing $d[a]$.

**Pass 4 (detection pass):** Edge $(c, a)$ satisfies
$d[a] > d[c] + w(c, a)$, confirming the negative cycle.

## Implementation

```python
"""
Negative cycle detection and extraction using Bellman-Ford.

Detects whether a negative-weight cycle is reachable from the source
and, if so, extracts the cycle vertices.
"""

from math import inf


# === Bellman-Ford with negative cycle detection ==============================

def detect_negative_cycle(
    vertices: list, edges: list, source
) -> tuple[bool, list]:
    """Detect and extract a negative cycle reachable from source.

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
    has_cycle : bool
        True if a negative cycle is reachable from source.
    cycle : list
        Vertices forming the negative cycle (empty if none).
    """
    n = len(vertices)
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}

    # Standard |V| - 1 relaxation passes
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u

    # |V|-th pass: check for negative cycle
    cycle_vertex = None
    for u, v, w in edges:
        if dist[u] != inf and dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            pred[v] = u
            cycle_vertex = v
            break

    if cycle_vertex is None:
        return False, []

    # Trace back |V| steps to ensure we are inside the cycle
    v = cycle_vertex
    for _ in range(n):
        v = pred[v]

    # Extract the cycle
    cycle = []
    u = v
    while True:
        cycle.append(u)
        u = pred[u]
        if u == v:
            cycle.append(u)
            break
    cycle.reverse()
    return True, cycle


# === Cycle weight computation ================================================

def cycle_weight(cycle: list, edge_weights: dict) -> float:
    """Compute the total weight of a cycle.

    Parameters
    ----------
    cycle : list
        Cycle vertices where cycle[0] == cycle[-1].
    edge_weights : dict
        Mapping (u, v) -> weight.
    """
    total = 0
    for i in range(len(cycle) - 1):
        total += edge_weights[(cycle[i], cycle[i + 1])]
    return total


# === Demo ====================================================================

if __name__ == "__main__":
    # Graph with a negative cycle: a -> b -> c -> a has weight -4
    vertices = ["s", "a", "b", "c"]
    edges = [
        ("s", "a", 4),
        ("a", "b", -2),
        ("b", "c", 3),
        ("c", "a", -5),
    ]
    edge_weights = {(u, v): w for u, v, w in edges}

    has_cycle, cycle = detect_negative_cycle(vertices, edges, "s")
    print(f"Negative cycle detected: {has_cycle}")
    if has_cycle:
        print(f"Cycle: {cycle}")
        print(f"Cycle weight: {cycle_weight(cycle, edge_weights)}")

    # Graph without a negative cycle
    print("\n--- Graph without negative cycle ---")
    edges_ok = [
        ("s", "a", 4),
        ("a", "b", -2),
        ("b", "c", 3),
        ("c", "a", 1),  # cycle weight = -2 + 3 + 1 = 2 > 0
    ]
    has_cycle2, cycle2 = detect_negative_cycle(vertices, edges_ok, "s")
    print(f"Negative cycle detected: {has_cycle2}")
```

**Output:**

```
Negative cycle detected: True
Cycle: ['a', 'b', 'c', 'a']
Cycle weight: -4

--- Graph without negative cycle ---
Negative cycle detected: False
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24.1: The Bellman-Ford Algorithm.
