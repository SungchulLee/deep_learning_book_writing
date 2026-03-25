# Detect Cycle in Directed Graph

Detecting cycles in directed graphs is essential for determining whether a dependency structure is valid. A directed graph without cycles is a DAG (directed acyclic graph), which admits topological sorting and is the foundation for scheduling, build systems, and prerequisite checking. The standard detection algorithm uses DFS with a three-color marking scheme to identify back edges, each of which signals a cycle.

## Directed Cycles

A **directed cycle** in a digraph $G = (V, E)$ is a sequence of distinct vertices $v_0, v_1, \ldots, v_{k-1}$ with $k \geq 2$ such that $(v_i, v_{i+1 \bmod k}) \in E$ for all $i$. In other words, the directed edges form a closed loop.

A digraph with no directed cycle is called a **directed acyclic graph (DAG)**.

!!! tip "Theorem: DAG Characterization"
    A directed graph $G$ is a DAG if and only if a DFS of $G$ produces no back edges.

A **back edge** in a DFS is an edge $(u, v)$ where $v$ is an ancestor of $u$ in the DFS tree -- meaning $v$ was discovered before $u$ and $u$ is discovered before $v$ finishes. Such an edge closes a cycle from $v$ down the DFS tree to $u$, then back to $v$ via the back edge.

## The Three-Color Algorithm

The standard algorithm maintains a color state for each vertex:

- **White**: undiscovered -- the vertex has not yet been visited.
- **Gray**: in progress -- the vertex is on the current DFS recursion stack (discovered but not finished).
- **Black**: finished -- all descendants of this vertex have been fully explored.

A cycle exists if and only if DFS encounters an edge from a gray vertex to another gray vertex (a back edge). When vertex $u$ is gray and we find an edge $(u, v)$ where $v$ is also gray, the path from $v$ to $u$ in the DFS tree, combined with edge $(u, v)$, forms a directed cycle.

### Correctness Argument

During DFS, gray vertices form a path from the root of the current DFS tree to the vertex being processed. If we encounter edge $(u, v)$ and $v$ is gray, then $v$ is an ancestor of $u$ on this path, so the path $v \to \cdots \to u \to v$ is a directed cycle. Conversely, if a directed cycle exists with vertices $v_0 \to v_1 \to \cdots \to v_{k-1} \to v_0$, then the first vertex in the cycle to be discovered by DFS will be gray when DFS reaches the back edge that closes the cycle.

### Complexity

The algorithm visits each vertex once and examines each edge once, giving a time complexity of $O(V + E)$ and space complexity of $O(V)$ for the color array plus $O(V)$ for the recursion stack.

$$
T(V, E) = O(V + E)
$$

## Implementation

```python
"""
Directed cycle detection using DFS with three-color marking.

Detects whether a directed graph contains a cycle by tracking
vertex states (WHITE, GRAY, BLACK) during depth-first search.
A back edge to a GRAY vertex indicates a cycle.
"""


# === Constants ===

WHITE, GRAY, BLACK = 0, 1, 2


# === Cycle Detection ===

def has_cycle_directed(adj, n):
    """
    Detect whether a directed graph contains a cycle.

    Uses DFS with three-color marking. Returns True if a cycle
    exists, False if the graph is a DAG.
    """
    color = [WHITE] * n

    def dfs(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return True  # back edge found
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for u in range(n):
        if color[u] == WHITE:
            if dfs(u):
                return True
    return False


def find_cycle_directed(adj, n):
    """
    Find and return one directed cycle, or empty list if none.

    Tracks the DFS parent to reconstruct the cycle path when
    a back edge is detected.
    """
    color = [WHITE] * n
    parent = [-1] * n
    cycle = []

    def dfs(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                # Reconstruct cycle from v to u, then back to v
                path = [v]
                cur = u
                while cur != v:
                    path.append(cur)
                    cur = parent[cur]
                path.append(v)
                path.reverse()
                cycle.extend(path)
                return True
            if color[v] == WHITE:
                parent[v] = u
                if dfs(v):
                    return True
        color[u] = BLACK
        return False

    for u in range(n):
        if color[u] == WHITE:
            if dfs(u):
                return cycle
    return cycle


# === Main ===

if __name__ == "__main__":
    # Graph with cycle: 0 -> 1 -> 2 -> 0
    adj1 = [[1], [2], [0, 3], []]
    print(f"Graph 1 has cycle: {has_cycle_directed(adj1, 4)}")
    print(f"Cycle found: {find_cycle_directed(adj1, 4)}")

    # DAG: 0 -> 1 -> 3, 0 -> 2 -> 3
    adj2 = [[1, 2], [3], [3], []]
    print(f"\nGraph 2 (DAG) has cycle: {has_cycle_directed(adj2, 4)}")
    print(f"Cycle found: {find_cycle_directed(adj2, 4)}")
```

**Output:**
```
Graph 1 has cycle: True
Cycle found: [0, 1, 2, 0]
Graph 2 (DAG) has cycle: False
Cycle found: []
```

!!! warning "Recursion Depth"
    The recursive DFS implementation may hit Python's default recursion limit on large graphs. For production use, convert to an iterative version using an explicit stack, or increase the limit with `sys.setrecursionlimit`.

## Connection to Topological Sorting

A directed graph admits a [topological ordering](../../ch17/topological/dag.md) if and only if it is a DAG. The cycle detection algorithm can be extended to produce a topological sort: when DFS finishes a vertex (coloring it black), prepend it to the output list. If no back edge is found, the resulting list is a valid topological order.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Section 22.4.
