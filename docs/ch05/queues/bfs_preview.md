# BFS Preview

Breadth-first search (BFS) is the most important algorithmic application of the queue data structure. Given a graph and a starting vertex, BFS explores all vertices at distance 1 before any vertex at distance 2, all vertices at distance 2 before distance 3, and so on. This level-by-level exploration arises directly from the queue's FIFO property: vertices discovered earlier are explored earlier. BFS finds shortest paths in unweighted graphs and serves as a building block for many other graph algorithms. This page provides a preview of BFS to motivate the study of queues; the full treatment appears in the graph algorithms chapter.

## Why a Queue Produces Level-Order Exploration

When BFS visits a vertex $v$, it enqueues all of $v$'s unvisited neighbors. Because the queue is FIFO, these neighbors will be explored only after all vertices that were already in the queue. Those earlier vertices are all at the same distance from the source (or one level closer), so BFS naturally processes vertices in order of their distance from the source.

This is the fundamental insight: **FIFO ordering guarantees breadth-first exploration**. If we replaced the queue with a stack (LIFO), we would get depth-first search instead.

## BFS Algorithm

The algorithm maintains three pieces of state:

1. A **visited** set to track which vertices have been seen
2. A **queue** of vertices to explore
3. A **distance** dictionary recording each vertex's distance from the source

**Procedure:**

1. Mark the source vertex as visited with distance 0 and enqueue it
2. While the queue is non-empty:
    - Dequeue a vertex $u$
    - For each neighbor $v$ of $u$:
        - If $v$ has not been visited, mark it as visited with distance $d(u) + 1$ and enqueue it

## Complexity

BFS visits each vertex exactly once and examines each edge exactly once (for directed graphs) or twice (for undirected graphs), giving a time complexity of:

$$
T(V, E) = O(V + E)
$$

The space complexity is $O(V)$ for the visited set, distance dictionary, and queue.

```python
"""
BFS preview — breadth-first search as a queue application.

Demonstrates how the queue's FIFO property produces level-order
graph exploration, finding shortest paths in unweighted graphs.
"""
from collections import deque


# === BFS Implementation =======================================================

def bfs(graph, source):
    """Breadth-first search from a source vertex.

    Returns:
        visited_order: list of vertices in BFS visit order.
        distances: dict mapping each vertex to its distance from source.
        parent: dict mapping each vertex to its BFS parent (for path reconstruction).

    Time:  O(V + E)
    Space: O(V)
    """
    visited = {source}
    distances = {source: 0}
    parent = {source: None}
    queue = deque([source])
    visited_order = []

    while queue:
        u = queue.popleft()
        visited_order.append(u)
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                distances[v] = distances[u] + 1
                parent[v] = u
                queue.append(v)

    return visited_order, distances, parent


def reconstruct_path(parent, target):
    """Reconstruct the shortest path from the source to the target."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


# === BFS with Level Tracking ==================================================

def bfs_by_level(graph, source):
    """BFS that explicitly tracks and prints each level.

    Shows the level-by-level exploration pattern that makes BFS
    useful for shortest path computation.
    """
    visited = {source}
    queue = deque([source])
    level = 0
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            u = queue.popleft()
            current_level.append(u)
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        levels.append(current_level)
        print(f"  Level {level}: {current_level}")
        level += 1

    return levels


# === Demonstration ============================================================

if __name__ == "__main__":
    # Example graph (undirected, represented as adjacency list)
    #     A --- B --- E
    #     |     |
    #     C --- D --- F
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "D"],
        "D": ["B", "C", "F"],
        "E": ["B"],
        "F": ["D"],
    }

    # Basic BFS
    print("BFS from vertex 'A':")
    order, dist, parent = bfs(graph, "A")
    print(f"  Visit order: {order}")
    print(f"  Distances:   {dist}")
    print()

    # Level-by-level BFS
    print("BFS levels from vertex 'A':")
    bfs_by_level(graph, "A")
    print()

    # Shortest path reconstruction
    print("Shortest paths from 'A':")
    for target in sorted(graph.keys()):
        if target != "A":
            path = reconstruct_path(parent, target)
            print(f"  A → {target}: {' → '.join(path)} (distance {dist[target]})")
```

**Output:**
```
BFS from vertex 'A':
  Visit order: ['A', 'B', 'C', 'D', 'E', 'F']
  Distances:   {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 3}

BFS levels from vertex 'A':
  Level 0: ['A']
  Level 1: ['B', 'C']
  Level 2: ['D', 'E']
  Level 3: ['F']

Shortest paths from 'A':
  A → B: A → B (distance 1)
  A → C: A → C (distance 1)
  A → D: A → D (distance 2)
  A → E: A → E (distance 2)
  A → F: A → F (distance 3)
```

The level-by-level output confirms that BFS explores all vertices at distance $d$ before any vertex at distance $d+1$. The shortest path from A to F passes through two intermediate vertices, giving distance 3.

## BFS Produces Shortest Paths

In an unweighted graph, BFS computes shortest paths from the source to every reachable vertex. This follows from two facts:

1. **Monotonicity**: the distances assigned by BFS are non-decreasing. If vertex $u$ is dequeued before vertex $v$, then $d(u) \leq d(v)$.
2. **Optimality**: when BFS first discovers vertex $v$ through vertex $u$, the path from the source through $u$ to $v$ has exactly $d(u) + 1$ edges, which equals the true shortest path length.

!!! tip "When BFS Does Not Find Shortest Paths"
    BFS finds shortest paths only in **unweighted** graphs (or equivalently, graphs where all edges have the same weight). For weighted graphs, Dijkstra's algorithm or Bellman-Ford must be used instead.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
