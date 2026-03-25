# Bidirectional Search

Standard shortest-path algorithms search outward from the source, potentially exploring many vertices far from the target. **Bidirectional search** runs two simultaneous searches -- one forward from the source $s$ and one backward from the target $t$ -- and terminates when the two frontiers meet. In many practical scenarios this dramatically reduces the number of explored vertices compared to a unidirectional search.

## Intuition

Consider BFS on a graph where every vertex has $b$ neighbors and the shortest path has length $d$. Unidirectional BFS explores up to $O(b^d)$ vertices. Bidirectional BFS runs two searches, each reaching depth $d/2$, exploring roughly $O(b^{d/2})$ vertices in each direction. The total is

$$
O(2 \cdot b^{d/2}) = O(b^{d/2})
$$

which can be exponentially smaller than $O(b^d)$.

## Bidirectional BFS

For unweighted graphs, the algorithm runs two BFS instances simultaneously. On each step it expands the frontier with fewer vertices (to keep work balanced). The search terminates when a vertex appears in both visited sets.

```python
"""
Bidirectional BFS for shortest path in an unweighted graph.

Runs forward BFS from source and backward BFS from target,
terminating when the two frontiers meet.
"""

from collections import deque

# === Bidirectional BFS ========================================================

def bidirectional_bfs(graph, source, target):
    """Find shortest path using bidirectional BFS.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list for an undirected graph.
    source : int
        Starting vertex.
    target : int
        Goal vertex.

    Returns
    -------
    list[int] | None
        Shortest path from source to target, or None if unreachable.
    """
    if source == target:
        return [source]

    # Forward search state
    front_visited = {source: None}
    front_queue = deque([source])

    # Backward search state
    back_visited = {target: None}
    back_queue = deque([target])

    def build_path(meeting_point):
        """Reconstruct path from both predecessor maps."""
        # Forward part: source -> meeting point
        path = []
        node = meeting_point
        while node is not None:
            path.append(node)
            node = front_visited[node]
        path.reverse()

        # Backward part: meeting point -> target
        node = back_visited[meeting_point]
        while node is not None:
            path.append(node)
            node = back_visited[node]

        return path

    while front_queue and back_queue:
        # Expand the smaller frontier
        if len(front_queue) <= len(back_queue):
            node = front_queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in front_visited:
                    front_visited[neighbor] = node
                    front_queue.append(neighbor)
                    if neighbor in back_visited:
                        return build_path(neighbor)
        else:
            node = back_queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in back_visited:
                    back_visited[neighbor] = node
                    back_queue.append(neighbor)
                    if neighbor in front_visited:
                        return build_path(neighbor)

    return None


# === Main =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 4],
        3: [1, 5],
        4: [2, 5],
        5: [3, 4, 6],
        6: [5],
    }

    path = bidirectional_bfs(graph, 0, 6)
    print(f"Bidirectional BFS path: {path}")
    print(f"Path length: {len(path) - 1} edges")
```

**Output:**
```
Bidirectional BFS path: [0, 1, 3, 5, 6]
Path length: 4 edges
```

## Bidirectional Dijkstra

For weighted graphs, bidirectional Dijkstra runs two priority-queue-based searches. The termination condition is more subtle: the search stops when the sum of the minimum keys in the two priority queues exceeds the best path found so far through any meeting vertex. This ensures optimality because no unexplored path can be cheaper.

!!! warning "Termination requires care"
    Simply stopping when a vertex appears in both closed sets does not guarantee optimality in weighted graphs. The correct condition checks that the shortest candidate path is at most the sum of the two frontier minimums: $\mu \leq d_f^{\min} + d_b^{\min}$, where $\mu$ is the best meeting-point distance found so far.

## Complexity

| Variant | Time (worst case) | Space | Practical Speedup |
|---|---|---|---|
| Bidirectional BFS | $O(b^{d/2})$ | $O(b^{d/2})$ | Up to $\sqrt{b^d}$ over BFS |
| Bidirectional Dijkstra | $O((V + E) \log V)$ | $O(V)$ | ~2x in practice |

The worst-case complexity of bidirectional Dijkstra matches standard Dijkstra, but in practice the search explores roughly half the vertices, cutting wall-clock time significantly.

## When to Use Bidirectional Search

Bidirectional search is most effective when:

- The source and target are both known in advance.
- The graph is undirected (or has an easily reversed edge set).
- The branching factor is high and the solution depth is moderate.

For directed graphs, backward search requires the reverse graph (edges flipped), which must either be precomputed or the graph structure must allow efficient reverse adjacency lookups.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 24-25. MIT Press.
- Pohl, I. (1971). Bi-directional search. *Machine Intelligence*, 6, 127-140.
