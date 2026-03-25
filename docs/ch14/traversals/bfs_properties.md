# BFS Properties

Understanding the formal properties of breadth-first search explains why the algorithm works correctly and what guarantees it provides. BFS is not just a traversal order; it produces a tree that encodes shortest-path information and partitions the graph into distance layers. This page states and justifies the key properties that underpin every BFS application.

## The BFS Tree

When BFS discovers a new vertex $v$ through an edge $(u, v)$, it records $u$ as the **predecessor** (or parent) of $v$. The collection of these predecessor edges forms a rooted tree called the **BFS tree** $T$. Every vertex reachable from the source $s$ appears exactly once in $T$, and $s$ is the root.

??? note "BFS tree vs. the original graph"
    The BFS tree contains exactly $|V_{\text{reachable}}| - 1$ edges. Edges of the original graph that do not appear in $T$ are called **cross edges** and always connect vertices on the same level or adjacent levels.

## Level-by-Level Exploration

BFS uses a FIFO queue, which ensures that all vertices at distance $d$ from the source are processed before any vertex at distance $d + 1$. Formally, define the **level** of vertex $v$ as

$$
\text{level}(v) = \begin{cases} 0 & \text{if } v = s \\ \min\{\text{level}(u) + 1 : (u,v) \in E,\; u \text{ already visited}\} & \text{otherwise} \end{cases}
$$

The FIFO discipline guarantees that `queue.popleft()` always returns a vertex whose level is less than or equal to the level of every other vertex in the queue.

## Shortest-Path Property

The most important property of BFS is that it computes shortest paths in unweighted graphs.

!!! tip "Shortest-path guarantee"
    For every vertex $v$ reachable from source $s$, the path from $s$ to $v$ in the BFS tree has exactly $\delta(s, v)$ edges, where $\delta(s, v)$ is the shortest-path distance in the original graph.

**Proof sketch.** Proceed by induction on $\delta(s, v)$.

- **Base case.** $\delta(s, s) = 0$, and the path from $s$ to itself in the BFS tree has zero edges.
- **Inductive step.** Assume the property holds for all vertices with $\delta(s, u) = d$. Let $v$ satisfy $\delta(s, v) = d + 1$. Then there exists an edge $(u, v)$ with $\delta(s, u) = d$. By the inductive hypothesis, $u$ is discovered at level $d$. When $u$ is processed, $v$ is added to the queue (if not already visited) and placed at level $d + 1$.

Because BFS never revisits a vertex, the first discovery of $v$ establishes the minimum number of edges from $s$ to $v$.

## Time and Space Complexity

Each vertex is enqueued and dequeued at most once, and each edge is examined at most twice (once from each endpoint in an undirected graph, or once in a directed graph). Therefore the time complexity is

$$
O(V + E)
$$

The space complexity is also $O(V)$ for the visited set and the queue, since the queue holds at most $O(V)$ vertices at any time.

## Completeness and Optimality

- **Completeness.** BFS is complete: if a path from $s$ to $v$ exists, BFS will find it. This follows because BFS visits every vertex in the connected component of $s$.
- **Optimality.** BFS is optimal for unweighted shortest paths. For weighted graphs, BFS does not minimize total weight; Dijkstra's algorithm or Bellman-Ford should be used instead.

## BFS Implementation with Properties Highlighted

```python
"""
BFS implementation that records distances and predecessors,
illustrating the shortest-path and BFS-tree properties.
"""

from collections import deque

# === BFS with distance and predecessor tracking ===============================

def bfs_with_properties(graph, source):
    """Run BFS and return distances and predecessor map.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.
    source : int
        Starting vertex.

    Returns
    -------
    dist : dict[int, int]
        Shortest distance (in edges) from source to each reachable vertex.
    pred : dict[int, int | None]
        Predecessor of each vertex in the BFS tree (None for the source).
    """
    dist = {source: 0}
    pred = {source: None}
    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                pred[v] = u
                queue.append(v)

    return dist, pred


def reconstruct_path(pred, target):
    """Trace the BFS tree from target back to the source."""
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = pred[node]
    path.reverse()
    return path


# === Main =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1, 5],
        5: [2, 4],
    }

    dist, pred = bfs_with_properties(graph, source=0)
    print("Vertex | Distance | Predecessor")
    print("-------|----------|------------")
    for v in sorted(dist):
        print(f"   {v}   |    {dist[v]}     |   {pred[v]}")

    path = reconstruct_path(pred, target=5)
    print(f"\nShortest path 0 → 5: {path} ({len(path) - 1} edges)")
```

**Output:**
```
Vertex | Distance | Predecessor
-------|----------|------------
   0   |    0     |   None
   1   |    1     |   0
   2   |    1     |   0
   3   |    2     |   1
   4   |    2     |   1
   5   |    2     |   2

Shortest path 0 → 5: [0, 2, 5] (2 edges)
```

The output confirms that BFS assigns each vertex its true shortest distance from the source and that the predecessor links form a valid BFS tree.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
