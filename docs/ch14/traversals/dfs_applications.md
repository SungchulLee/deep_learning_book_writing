# DFS Applications

Depth-first search is one of the most versatile graph algorithms. Its ability to explore as deep as possible before backtracking gives it properties that BFS cannot match: it naturally detects cycles, computes pre/post orderings that reveal graph structure, and lays the groundwork for topological sorting and strongly connected component algorithms. This page presents the core DFS applications.

## Pre-Order and Post-Order Numbering

Many graph algorithms need to know which vertices are ancestors or descendants of others. DFS timestamps provide exactly this information. DFS assigns two timestamps to each vertex: a **discovery time** (pre-order) when the vertex is first visited, and a **finish time** (post-order) when all of its descendants have been fully explored. These timestamps encode the recursive structure of the DFS tree and are the key to most DFS-based algorithms.

For a directed graph with $n$ vertices, pre and post numbers range from $1$ to $2n$, and for every pair of vertices $u$ and $v$, their intervals $[\text{pre}(u), \text{post}(u)]$ and $[\text{pre}(v), \text{post}(v)]$ are either disjoint or one contains the other. This **nesting property** follows directly from the recursive nature of DFS: if $v$ is a descendant of $u$, then $v$ is discovered after $u$ and finished before $u$.

```python
"""
DFS with pre-order and post-order numbering.

Pre/post timestamps reveal the ancestor-descendant relationships
in the DFS tree and enable cycle detection and topological sorting.
"""

# === DFS with timestamps =====================================================

def dfs_timestamps(graph):
    """Compute pre-order and post-order numbers for every vertex.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list for a directed graph.

    Returns
    -------
    pre : dict[int, int]
        Discovery time of each vertex.
    post : dict[int, int]
        Finish time of each vertex.
    """
    pre = {}
    post = {}
    clock = [0]  # mutable counter for nested function

    def explore(u):
        clock[0] += 1
        pre[u] = clock[0]
        for v in graph[u]:
            if v not in pre:
                explore(v)
        clock[0] += 1
        post[u] = clock[0]

    for vertex in graph:
        if vertex not in pre:
            explore(vertex)

    return pre, post


# === Main =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
    pre, post = dfs_timestamps(graph)
    print("Vertex | Pre | Post")
    print("-------|-----|-----")
    for v in sorted(pre):
        print(f"   {v}   |  {pre[v]}  |  {post[v]}")
```

**Output:**
```
Vertex | Pre | Post
-------|-----|-----
   0   |  1  |  8
   1   |  2  |  5
   2   |  6  |  7
   3   |  3  |  4
```

Vertex 0 has the widest interval $[1, 8]$, confirming that all other vertices are its descendants in the DFS tree.

## Cycle Detection

Detecting cycles is essential in many contexts: dependency resolution (e.g., build systems, package managers) fails when circular dependencies exist, and many algorithms assume the input is acyclic. DFS provides a natural cycle test through the structure of the edges it encounters.

A directed graph contains a cycle if and only if DFS encounters a **back edge** -- an edge from a vertex to one of its ancestors in the DFS tree. A back edge $(u, v)$ is recognized when $v$ has been discovered (has a pre-number) but has not yet finished (has no post-number). Equivalently, $v$ is currently on the recursion stack.

```python
"""
Cycle detection in a directed graph using DFS.

A back edge (to a vertex still on the recursion stack) indicates a cycle.
"""

# === Cycle detection ==========================================================

def has_cycle(graph):
    """Return True if the directed graph contains a cycle.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list for a directed graph.

    Returns
    -------
    bool
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}

    def explore(u):
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == GRAY:
                return True  # back edge found
            if color[v] == WHITE and explore(v):
                return True
        color[u] = BLACK
        return False

    return any(color[v] == WHITE and explore(v) for v in graph)


# === Main =====================================================================

if __name__ == "__main__":
    dag = {0: [1, 2], 1: [3], 2: [3], 3: []}
    cyclic = {0: [1], 1: [2], 2: [0]}

    print(f"DAG has cycle? {has_cycle(dag)}")
    print(f"Cyclic graph has cycle? {has_cycle(cyclic)}")
```

**Output:**
```
DAG has cycle? False
Cyclic graph has cycle? True
```

## Topological Sort Preview

When tasks have prerequisite relationships (courses before graduation, compilation steps before linking), we need an ordering that respects all dependencies. DFS provides an elegant solution for directed acyclic graphs.

In a DAG, a **topological ordering** arranges vertices so that every edge points from an earlier vertex to a later one. DFS produces a topological order by listing vertices in reverse post-order. If vertex $u$ has an edge to $v$, then $u$ finishes after $v$ (since the graph is acyclic), so reversing the finish order places $u$ before $v$.

!!! note "Full coverage"
    The complete topological sort algorithm, including Kahn's BFS-based variant, is covered in the topological sorting section of the next chapter.

```python
"""
Topological sort via DFS reverse post-order.

Valid only for DAGs; produces one of potentially many valid orderings.
"""

# === Topological sort =========================================================

def topological_sort(graph):
    """Return a topological ordering of a DAG.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list for a DAG.

    Returns
    -------
    list[int]
        Vertices in topological order.
    """
    visited = set()
    order = []

    def explore(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                explore(v)
        order.append(u)

    for vertex in graph:
        if vertex not in visited:
            explore(vertex)

    order.reverse()
    return order


# === Main =====================================================================

if __name__ == "__main__":
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"Topological order: {topological_sort(dag)}")
```

**Output:**
```
Topological order: [0, 2, 1, 3, 4]
```

Every edge in the DAG points from left to right in this ordering, confirming its validity.

## Connected Components in Undirected Graphs

Just as with BFS, DFS can enumerate connected components by launching a search from each unvisited vertex. Each DFS call from an unvisited vertex discovers one complete component. The DFS-based approach runs in the same $O(V + E)$ time but differs from BFS in the order vertices are visited within each component: DFS dives deep before exploring siblings, while BFS expands level by level.

## Summary

| Application | Key DFS Feature Used | Time |
|---|---|---|
| Pre/post numbering | Recursive structure of DFS | $O(V + E)$ |
| Cycle detection | Back edges to gray vertices | $O(V + E)$ |
| Topological sort | Reverse post-order on DAGs | $O(V + E)$ |
| Connected components | DFS forest partitions graph | $O(V + E)$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
- Dasgupta, S., Papadimitriou, C., & Vazirani, U. (2006). *Algorithms*, Chapters 3-4.
