# Eulerian Path and Circuit

The Seven Bridges of Konigsberg puzzle asked whether one could walk through the city crossing each bridge exactly once and return to the starting point. Euler proved in 1736 that no such walk exists, founding the field of graph theory. The key insight is that the existence of such a traversal depends entirely on the **degree parity** of the vertices.

## Definitions

**Eulerian circuit.** A closed walk in a graph that visits every edge exactly once and returns to the starting vertex.

**Eulerian path.** A walk that visits every edge exactly once but may start and end at different vertices.

**Eulerian graph.** A graph that contains an Eulerian circuit. A graph that contains an Eulerian path (but not a circuit) is called *semi-Eulerian*.

## Existence Theorems

### Undirected Graphs

!!! note "Euler's Theorem (Undirected)"
    Let $G = (V, E)$ be a connected undirected graph (ignoring isolated vertices). Then:

    - $G$ has an **Eulerian circuit** if and only if every vertex has even degree.
    - $G$ has an **Eulerian path** (but no circuit) if and only if exactly two vertices have odd degree. The path must start at one odd-degree vertex and end at the other.

??? example "Proof sketch for the circuit case"
    **Necessity.** If an Eulerian circuit exists, then every time the walk enters a vertex through one edge, it must leave through a different edge. Each vertex is therefore entered and exited the same number of times, so its degree is even.

    **Sufficiency.** Start at any vertex and follow unused edges until returning to the start (this must happen since every vertex has even degree). If some edges remain unused, there exists a vertex $v$ on the current walk that is incident to an unused edge (by connectivity). Start a new walk from $v$ using only unused edges, splice it into the first walk, and repeat until all edges are covered. $\square$

### Directed Graphs

!!! note "Euler's Theorem (Directed)"
    Let $G = (V, E)$ be a directed graph in which every vertex with nonzero degree belongs to the same strongly connected component. Then:

    - $G$ has an **Eulerian circuit** if and only if $\text{in-deg}(v) = \text{out-deg}(v)$ for every vertex $v$.
    - $G$ has an **Eulerian path** if and only if there is exactly one vertex with $\text{out-deg} - \text{in-deg} = 1$ (the start) and one vertex with $\text{in-deg} - \text{out-deg} = 1$ (the end), with all other vertices balanced.

## Checking Existence

```python
"""
Check whether an undirected graph has an Eulerian path or circuit.

Uses the degree-parity theorem: count odd-degree vertices to determine
whether an Euler circuit, Euler path, or neither exists.
"""

from collections import defaultdict

# === Euler Existence Check ===

def euler_type(n: int, edges: list[tuple[int, int]]) -> str:
    """Determine if the graph has an Euler circuit, path, or neither.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of undirected edges (u, v).

    Returns:
        'circuit', 'path', or 'none'.
    """
    degree = [0] * n
    adj = defaultdict(set)

    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)

    # Check connectivity among vertices with nonzero degree
    nonzero = [v for v in range(n) if degree[v] > 0]
    if not nonzero:
        return "circuit"  # trivial: no edges

    visited = set()
    stack = [nonzero[0]]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for w in adj[v]:
            if w not in visited:
                stack.append(w)

    if visited != set(nonzero):
        return "none"  # not connected

    odd_count = sum(1 for v in range(n) if degree[v] % 2 == 1)
    if odd_count == 0:
        return "circuit"
    elif odd_count == 2:
        return "path"
    else:
        return "none"


# === Demonstration ===

if __name__ == "__main__":
    # Triangle: 0-1-2-0 (all even degrees)
    print(euler_type(3, [(0,1),(1,2),(2,0)]))  # circuit

    # Path graph: 0-1-2 (vertices 0,2 have odd degree)
    print(euler_type(3, [(0,1),(1,2)]))  # path

    # Star: center 0, leaves 1,2,3 (vertex 0 has degree 3)
    print(euler_type(4, [(0,1),(0,2),(0,3)]))  # none
```

**Output:**

```
circuit
path
none
```

The triangle has all even degrees so it admits an Eulerian circuit. The path graph $0{-}1{-}2$ has exactly two odd-degree vertices ($0$ and $2$), so it admits an Eulerian path. The star graph has vertex $0$ with degree $3$ and three vertices of degree $1$, giving four odd-degree vertices, so neither an Eulerian path nor circuit exists.

## Complexity

| Aspect | Existence check | Finding the circuit/path |
|--------|:---------------:|:-----------------------:|
| Time   | $O(V + E)$      | $O(V + E)$ (Hierholzer) |
| Space  | $O(V + E)$      | $O(E)$                  |

Checking existence requires only a single pass to compute degrees and verify connectivity. Constructing the actual Eulerian circuit or path is handled by Hierholzer's algorithm, covered on its own page.

## Degree Parity Summary

| Odd-degree vertices | Undirected graph has |
|:-------------------:|:--------------------:|
| $0$                 | Eulerian circuit     |
| $2$                 | Eulerian path only   |
| Any other count     | Neither              |

## Reference

- Euler, L. (1736). Solutio problematis ad geometriam situs pertinentis. *Commentarii academiae scientiarum Petropolitanae*, 8, 128--140.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 22: Elementary Graph Algorithms.
