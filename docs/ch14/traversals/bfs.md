# Breadth-First Search (BFS)

Breadth-first search explores a graph level by level: it visits all neighbors of the starting node before moving to their neighbors, then their neighbors' neighbors, and so on. This systematic exploration guarantees that BFS finds the **shortest path** (in terms of number of edges) from the source to every reachable node, making it the foundation for shortest-path algorithms in unweighted graphs.

## Algorithm

BFS uses a **queue** (FIFO) to manage the frontier of nodes to visit:

1. Enqueue the source node and mark it as visited
2. While the queue is not empty:
    - Dequeue a node $v$
    - For each unvisited neighbor $u$ of $v$: mark $u$ as visited, record its distance as $d(u) = d(v) + 1$, and enqueue $u$

## Implementation

```python
"""Breadth-first search on an adjacency-list graph.

Computes shortest distances (in edge count) from a source node.
"""
from collections import deque


# === BFS ===
def bfs(graph, source):
    """Return a dict of shortest distances from source to all reachable nodes."""
    visited = {source}
    dist = {source: 0}
    queue = deque([source])
    while queue:
        v = queue.popleft()
        for u in graph.get(v, []):
            if u not in visited:
                visited.add(u)
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist


# === Main ===
if __name__ == "__main__":
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"],
    }
    distances = bfs(graph, "A")
    for node in sorted(distances):
        print(f"  A -> {node}: {distances[node]}")
```

**Output:**
```
  A -> A: 0
  A -> B: 1
  A -> C: 1
  A -> D: 2
  A -> E: 2
  A -> F: 2
```

## Complexity

| Metric | Complexity |
|:---|:---:|
| Time | $O(V + E)$ |
| Space | $O(V)$ |

BFS visits each vertex and each edge at most once, giving linear time in the size of the graph.

## BFS in the Search Landscape

BFS is one member of a family of graph search strategies that differ in how the frontier is managed:

| Strategy | Data Structure | Guarantee |
|:---|:---|:---|
| BFS | Queue (FIFO) | Shortest path (unweighted) |
| DFS | Stack (LIFO) | Explores deeply first |
| Best-first / A* | Priority queue | Optimal path (with admissible heuristic) |

## References

[Introduction to Algorithms (CLRS), Section 22.2](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
