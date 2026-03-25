# DFS Properties

Depth-first search has structural properties that go far beyond simply visiting every vertex. The recursive nature of DFS imposes a nesting structure on the discovery and finish times that is captured by the **parenthesis theorem**. Combined with the **white-path theorem**, these properties explain why DFS is the engine behind cycle detection, topological sorting, and strongly connected component algorithms.

## The DFS Forest

When DFS runs on a graph that is not connected (undirected) or not strongly connected (directed), a single source vertex cannot reach every other vertex. To handle this, DFS iterates over all vertices and launches a new search from each unvisited vertex. The result is a **DFS forest**: a collection of rooted trees, one for each connected component or group of mutually reachable vertices. Each tree edge $(u, v)$ corresponds to the moment DFS first discovers $v$ from $u$.

## The Parenthesis Theorem

The parenthesis theorem is the fundamental structural result about DFS. It explains why the discovery and finish times encode the entire ancestor-descendant relationship of the DFS tree.

During DFS each vertex $u$ receives a discovery time $\text{pre}(u)$ and a finish time $\text{post}(u)$. The parenthesis theorem states that for any two vertices $u$ and $v$, exactly one of the following holds:

1. The intervals $[\text{pre}(u), \text{post}(u)]$ and $[\text{pre}(v), \text{post}(v)]$ are **entirely disjoint** -- neither is an ancestor of the other in the DFS tree.
2. $[\text{pre}(u), \text{post}(u)] \subset [\text{pre}(v), \text{post}(v)]$ -- vertex $u$ is a descendant of $v$.
3. $[\text{pre}(v), \text{post}(v)] \subset [\text{pre}(u), \text{post}(u)]$ -- vertex $v$ is a descendant of $u$.

The intervals never partially overlap. This is analogous to matched parentheses in an expression: every open parenthesis has a matching close, and pairs are either nested or disjoint.

**Proof sketch.** Consider vertices $u$ and $v$ with $\text{pre}(u) < \text{pre}(v)$. If $v$ is discovered while $u$ is still being explored (i.e., before $u$ finishes), then $v$ must be a descendant of $u$. DFS will finish $v$ before returning to $u$, so $\text{post}(v) < \text{post}(u)$, giving containment. If instead $v$ is discovered after $u$ finishes, then $\text{post}(u) < \text{pre}(v)$, giving disjointness. No other case is possible. $\square$

!!! tip "Ancestor test in constant time"
    Vertex $u$ is an ancestor of $v$ in the DFS tree if and only if $\text{pre}(u) \leq \text{pre}(v)$ and $\text{post}(v) \leq \text{post}(u)$. This gives an $O(1)$ ancestor check after a single $O(V + E)$ DFS pass.

## The White-Path Theorem

The parenthesis theorem describes the timestamp structure; the white-path theorem connects the DFS tree structure to the graph's actual edges. Together they provide a complete characterization of which vertices become descendants of which.

A vertex $v$ is a descendant of $u$ in the DFS forest if and only if, at the time $u$ is discovered, there exists a path from $u$ to $v$ consisting entirely of **white** (unvisited) vertices. This theorem is essential for proving correctness of algorithms that rely on DFS ordering, such as the characterization of back edges for cycle detection.

## Time and Space Complexity

DFS visits every vertex once and examines every edge once (in a directed graph) or twice (in an undirected graph). Therefore the time complexity is

$$
O(V + E)
$$

The space complexity is $O(V)$ for the visited set. The recursion stack (or explicit stack in the iterative version) can grow to $O(V)$ in the worst case (a path graph), giving $O(V)$ total auxiliary space.

## Recursive vs. Iterative DFS

The recursive implementation mirrors the mathematical definition directly, while the iterative version uses an explicit stack to avoid hitting Python's recursion limit on large graphs. Both produce the same DFS forest, but the iterative version may visit neighbors in a different order depending on how the adjacency list is traversed.

```python
"""
DFS implementations: recursive and iterative.

Demonstrates the parenthesis theorem through pre/post timestamps
and shows both recursive and iterative approaches.
"""

# === Recursive DFS with timestamps ============================================

def dfs_recursive(graph):
    """Run recursive DFS and return pre/post timestamps.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list for a directed graph.

    Returns
    -------
    pre : dict[int, int]
        Discovery times.
    post : dict[int, int]
        Finish times.
    """
    pre = {}
    post = {}
    clock = [0]

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


# === Iterative DFS =============================================================

def dfs_iterative(graph, source):
    """Iterative DFS traversal from a single source.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list.
    source : int
        Starting vertex.

    Returns
    -------
    list[int]
        Vertices in DFS visit order.
    """
    visited = set()
    stack = [source]
    order = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


# === Main =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}

    pre, post = dfs_recursive(graph)
    print("Parenthesis theorem demonstration:")
    for v in sorted(pre):
        print(f"  Vertex {v}: [{pre[v]}, {post[v]}]")

    print(f"\nIterative DFS order: {dfs_iterative(graph, 0)}")
```

**Output:**
```
Parenthesis theorem demonstration:
  Vertex 0: [1, 8]
  Vertex 1: [2, 5]
  Vertex 2: [6, 7]
  Vertex 3: [3, 4]

Iterative DFS order: [0, 1, 3, 2]
```

The intervals confirm the parenthesis theorem: $[2, 5]$ and $[6, 7]$ are disjoint (vertices 1 and 2 are siblings), while $[3, 4] \subset [2, 5]$ (vertex 3 is a descendant of vertex 1), and all intervals nest inside $[1, 8]$ (vertex 0 is the root).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
