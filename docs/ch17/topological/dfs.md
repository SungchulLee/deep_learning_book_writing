# DFS-Based Topological Sort

Depth-first search reveals a natural ordering of vertices in a directed acyclic graph. When DFS finishes processing a vertex -- meaning all its descendants have been fully explored -- that vertex can safely appear after all the vertices it depends on. By recording vertices in reverse order of their finish times, DFS produces a valid topological ordering without maintaining any explicit in-degree counts.

## Core Idea

The DFS-based topological sort exploits a key property of depth-first search on DAGs: if there is an edge $(u, v)$, then $v$ finishes before $u$. This happens because DFS explores $v$ (and all of $v$'s descendants) before returning to $u$. Reversing the finish-time order therefore places $u$ before $v$, satisfying the topological ordering constraint.

!!! tip "Finish-Time Property"
    In a DFS of a DAG $G = (V, E)$, for every edge $(u, v) \in E$, vertex $u$ has a later finish time than $v$. Sorting vertices by decreasing finish time yields a topological ordering.

## Algorithm

The algorithm performs a standard DFS and appends each vertex to a list when it finishes (all its neighbors have been fully explored). The final list, reversed, gives the topological order.

**Steps:**

1. Initialize all vertices as unvisited.
2. For each unvisited vertex $u$, run DFS from $u$.
3. In the DFS, after exploring all neighbors of $u$, append $u$ to a stack (or list).
4. The reverse of this list is the topological ordering.

## Correctness

!!! note "Why Reverse Finish Order Works"
    Consider any edge $(u, v)$ in the DAG. When DFS processes edge $(u, v)$, vertex $v$ is either:

    - **Unvisited:** DFS recurses into $v$, so $v$ finishes before $u$.
    - **Fully processed:** $v$ already finished, so $v$'s finish time is earlier than $u$'s.
    - **Currently being processed (GRAY):** This would mean $v$ is an ancestor of $u$, creating a back edge and thus a cycle. But $G$ is a DAG, so this case cannot occur.

    In all possible cases, $v$ finishes before $u$, ensuring $u$ appears before $v$ in the reverse finish-time order.

## Complexity

The algorithm visits each vertex exactly once and traverses each edge exactly once, giving:

$$
T(V, E) = O(V + E)
$$

The space complexity is $O(V)$ for the recursion stack and the color/visited array.

## Implementation

```python
"""
DFS-based topological sort.

Records vertices in reverse finish-time order to produce a valid
topological ordering of a directed acyclic graph.
"""


# === DFS Topological Sort ===
def topo_sort_dfs(graph, n):
    """
    Compute a topological ordering using depth-first search.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a DAG with vertices labeled 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[int]
        Vertices in topological order, or an empty list if a cycle
        is detected.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    order = []
    has_cycle = False

    def dfs(u):
        nonlocal has_cycle
        color[u] = GRAY
        for v in graph.get(u, []):
            if color[v] == GRAY:
                has_cycle = True
                return
            if color[v] == WHITE:
                dfs(v)
                if has_cycle:
                    return
        color[u] = BLACK
        order.append(u)  # record finish time

    for u in range(n):
        if color[u] == WHITE:
            dfs(u)
            if has_cycle:
                return []

    order.reverse()
    return order


# === Main ===
if __name__ == "__main__":
    # DAG: 0 -> 1 -> 3, 0 -> 2 -> 3 -> 4
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    result = topo_sort_dfs(dag, 5)
    print(f"Topological order: {result}")

    # Verify: every edge (u, v) has u before v
    pos = {v: i for i, v in enumerate(result)}
    valid = all(pos[u] < pos[v] for u in dag for v in dag[u])
    print(f"Valid topological order: {valid}")
```

**Output:**
```
Topological order: [0, 2, 1, 3, 4]
Valid topological order: True
```

The three-coloring scheme (WHITE, GRAY, BLACK) simultaneously detects cycles. A GRAY vertex is on the current recursion stack; encountering it again means a back edge exists, which implies a cycle. This allows the algorithm to serve as both a topological sorter and a [DAG verifier](dag.md).

## Iterative Variant

For graphs where the recursion depth may exceed the call stack limit, an iterative version using an explicit stack avoids stack overflow:

```python
"""
Iterative DFS-based topological sort.

Uses an explicit stack to avoid recursion depth limits on large graphs.
"""


# === Iterative DFS Topological Sort ===
def topo_sort_dfs_iterative(graph, n):
    """
    Compute topological ordering using iterative DFS.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a DAG with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[int]
        Vertices in topological order.
    """
    visited = [False] * n
    order = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            u, idx = stack.pop()
            neighbors = graph.get(u, [])
            if idx < len(neighbors):
                stack.append((u, idx + 1))
                v = neighbors[idx]
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)

    order.reverse()
    return order


# === Main ===
if __name__ == "__main__":
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"Iterative topological order: {topo_sort_dfs_iterative(dag, 5)}")
```

**Output:**
```
Iterative topological order: [0, 2, 1, 3, 4]
```

## Comparison with Kahn's Algorithm

Both DFS-based sort and [Kahn's algorithm](kahn.md) run in $O(V + E)$ time, but they differ in approach:

| Property | DFS-Based | Kahn's Algorithm |
|---|---|---|
| Strategy | Reverse finish-time ordering | Iterative source removal |
| Data structure | Recursion stack (or explicit stack) | Queue of zero in-degree vertices |
| Cycle detection | Back edge during DFS | Unprocessed vertices remain |
| Output order | Often reverse of insertion order | Tends to follow a BFS-like ordering |
| Parallelism | Harder to parallelize | Sources can be processed in parallel |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
