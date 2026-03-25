# Paths and Cycles

Nearly every graph algorithm -- from shortest-path computations to connectivity testing -- relies on the notions of paths and cycles. A path describes how to travel between two vertices through a sequence of edges, while a cycle is a path that returns to its starting point. Distinguishing between the various types (walks, trails, paths, cycles) is critical for precise algorithmic reasoning.

## Walks, Trails, and Paths

These three terms form a hierarchy from most general to most restrictive.

### Walk

A **walk** in a graph $G = (V, E)$ is a sequence of vertices $v_0, v_1, \ldots, v_k$ such that each consecutive pair $(v_{i}, v_{i+1})$ is connected by an edge. The **length** of the walk is $k$ (the number of edges traversed). Vertices and edges may repeat.

### Trail

A **trail** is a walk in which no edge is repeated (though vertices may repeat). A **closed trail** starts and ends at the same vertex.

### Path

A **path** (or **simple path**) is a walk in which no vertex is repeated. The length of a path is the number of edges it contains.

$$
\text{Walk} \supseteq \text{Trail} \supseteq \text{Path}
$$

!!! example "Walk vs Trail vs Path"
    In a graph with edges $\{a,b\}, \{b,c\}, \{c,a\}, \{c,d\}$:

    - $a, b, c, a, b$ is a **walk** (length 4) — vertices and edges repeat.
    - $a, b, c, a$ is a **trail** (length 3) — no edge repeats, but vertex $a$ appears twice.
    - $a, b, c, d$ is a **path** (length 3) — no vertex repeats.

### Shortest Path Distance

The **distance** $d(u, v)$ between vertices $u$ and $v$ is the length of the shortest path from $u$ to $v$. If no path exists, $d(u, v) = \infty$. In unweighted graphs, [BFS](../traversals/bfs.md) computes shortest-path distances in $O(V + E)$ time.

## Cycles

A **cycle** (or **simple cycle**) is a closed walk $v_0, v_1, \ldots, v_k = v_0$ with $k \geq 3$ where all vertices $v_0, v_1, \ldots, v_{k-1}$ are distinct. The length of the cycle is $k$.

In a directed graph, a **directed cycle** follows edge directions: each $(v_i, v_{i+1 \bmod k})$ must be a directed edge.

!!! example "Cycles of Different Lengths"
    - A **triangle** is a cycle of length 3: $a, b, c, a$.
    - A **square** (4-cycle): $a, b, c, d, a$.
    - The shortest possible cycle in a simple undirected graph has length 3.

### Girth

The **girth** of a graph $G$ is the length of the shortest cycle in $G$. If $G$ is acyclic (a forest), the girth is defined as $\infty$.

## Connectivity via Paths

Paths and connectivity are intimately linked:

- An undirected graph is **connected** if there is a path between every pair of vertices.
- A directed graph is **strongly connected** if there is a directed path from $u$ to $v$ and from $v$ to $u$ for every pair $u, v$.
- A directed graph is **weakly connected** if replacing all directed edges with undirected edges yields a connected graph.

!!! tip "Theorem: Path Existence and Connectivity"
    An undirected graph $G$ is connected if and only if for every pair of vertices $u, v \in V$, there exists a path from $u$ to $v$.

## Special Path and Cycle Types

### Hamiltonian Path and Cycle

A **Hamiltonian path** visits every vertex exactly once. A **Hamiltonian cycle** is a Hamiltonian path that returns to the starting vertex. Determining whether a Hamiltonian path exists is NP-complete.

### Eulerian Trail and Circuit

An **Eulerian trail** traverses every edge exactly once. An **Eulerian circuit** is a closed Eulerian trail. By Euler's theorem, a connected undirected graph has an Eulerian circuit if and only if every vertex has even degree.

### Comparison

| Property | Hamiltonian | Eulerian |
|---|---|---|
| Visits every | vertex once | edge once |
| Existence check | NP-complete | Polynomial (degree check) |
| Condition (circuit) | No simple characterization | All degrees even |

## Path and Cycle Detection

```python
"""
Path finding and cycle detection in graphs.

Demonstrates path existence checking via BFS and simple cycle
detection via DFS in undirected graphs.
"""

from collections import deque


# === Path Finding (BFS) ===

def find_path(adj, n, start, end):
    """
    Find a path from start to end using BFS.

    Returns the path as a list of vertices, or an empty list
    if no path exists.
    """
    if start == end:
        return [start]
    visited = [False] * n
    parent = [-1] * n
    visited[start] = True
    queue = deque([start])

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                if v == end:
                    # Reconstruct path
                    path = []
                    cur = end
                    while cur != -1:
                        path.append(cur)
                        cur = parent[cur]
                    return path[::-1]
                queue.append(v)
    return []


# === Cycle Detection (Undirected) ===

def has_cycle_undirected(adj, n):
    """
    Detect whether an undirected graph contains a cycle using DFS.

    An edge to an already-visited vertex that is not the parent
    indicates a cycle.
    """
    visited = [False] * n

    def dfs(u, parent):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                if dfs(v, u):
                    return True
            elif v != parent:
                return True
        return False

    for u in range(n):
        if not visited[u]:
            if dfs(u, -1):
                return True
    return False


# === Main ===

if __name__ == "__main__":
    # Graph: 0-1-2-3, with extra edge 0-2 creating a cycle
    adj = [[1, 2], [0, 2], [0, 1, 3], [2]]

    path = find_path(adj, 4, 0, 3)
    print(f"Path from 0 to 3: {path}")

    path_none = find_path([[1], [0], [], []], 4, 0, 3)
    print(f"Path from 0 to 3 (disconnected): {path_none}")

    print(f"Has cycle: {has_cycle_undirected(adj, 4)}")

    # Tree (no cycle): 0-1, 0-2, 2-3
    adj_tree = [[1, 2], [0], [0, 3], [2]]
    print(f"Tree has cycle: {has_cycle_undirected(adj_tree, 4)}")
```

**Output:**
```
Path from 0 to 3: [0, 2, 3]
Path from 0 to 3 (disconnected): []
Has cycle: True
Tree has cycle: False
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
- West, D. B. (2001). *Introduction to Graph Theory* (2nd ed.). Prentice Hall. Section 1.2.
