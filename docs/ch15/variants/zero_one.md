# 0-1 BFS

Dijkstra's algorithm runs in $O((|V| + |E|) \log |V|)$ with a binary heap. When every edge weight is either 0 or 1, we can do much better. **0-1 BFS** exploits this restricted weight structure by using a **deque** (double-ended queue) instead of a priority queue, achieving $O(|V| + |E|)$ time -- the same as unweighted BFS. This technique appears frequently in grid-based problems where some transitions are free (weight 0) and others cost 1.

## Key Insight

In a graph with weights 0 and 1, the shortest-path distances form at most two "layers" at any time: the current distance $d$ and $d + 1$. A deque naturally maintains this two-level structure:

- When relaxing an edge with weight **0**, the neighbor gets the same distance as the current vertex. Push it to the **front** of the deque (it should be processed at the same priority level).
- When relaxing an edge with weight **1**, the neighbor gets distance $d + 1$. Push it to the **back** of the deque (it should be processed after all vertices at distance $d$).

This maintains the invariant that the deque is sorted by distance, with at most two distinct distance values at any time.

## Algorithm

1. Initialize $\text{dist}[s] = 0$ and $\text{dist}[v] = \infty$ for all $v \neq s$.
2. Push $s$ onto a deque.
3. While the deque is not empty:
    - Pop the front vertex $u$.
    - For each neighbor $v$ with edge weight $w \in \{0, 1\}$:
        - If $\text{dist}[u] + w < \text{dist}[v]$:
            - Set $\text{dist}[v] = \text{dist}[u] + w$.
            - If $w = 0$: push $v$ to the **front** of the deque.
            - If $w = 1$: push $v$ to the **back** of the deque.

## Correctness

The algorithm maintains the deque in non-decreasing order of distance. When a vertex is popped from the front, it has the smallest distance among all vertices in the deque, so its distance is final (same argument as Dijkstra). The weight-0 edges preserve this ordering because they do not increase the distance.

## Complexity

$$
\text{Time: } O(|V| + |E|), \qquad \text{Space: } O(|V|)
$$

Each vertex enters and leaves the deque at most twice (once at distance $d$, possibly once at distance $d$ via a weight-0 edge). Each edge is examined at most once.

## Implementation

```python
"""
0-1 BFS: shortest paths in graphs with edge weights 0 and 1.

Uses a deque instead of a priority queue to achieve O(V + E) time.
Weight-0 edges push to the front; weight-1 edges push to the back.
"""

from collections import deque


# === 0-1 BFS ===

def zero_one_bfs(graph: dict, source: int, n: int) -> list:
    """Compute shortest distances from source in a 0-1 weighted graph.

    Args:
        graph: Adjacency list {u: [(v, w), ...]} where w in {0, 1}.
        source: Starting vertex.
        n: Number of vertices (0-indexed).

    Returns:
        List of distances from source. dist[v] = float('inf') if
        v is unreachable.
    """
    dist = [float('inf')] * n
    dist[source] = 0
    dq = deque([source])

    while dq:
        u = dq.popleft()

        for v, w in graph.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                if w == 0:
                    dq.appendleft(v)  # front
                else:
                    dq.append(v)      # back

    return dist


# === Demonstration ===

if __name__ == "__main__":
    # Graph where some edges are free (0) and others cost 1
    #   0 --1-- 1 --0-- 2
    #   |               |
    #   0               1
    #   |               |
    #   3 --1-- 4 --0-- 5
    graph = {
        0: [(1, 1), (3, 0)],
        1: [(0, 1), (2, 0)],
        2: [(1, 0), (5, 1)],
        3: [(0, 0), (4, 1)],
        4: [(3, 1), (5, 0)],
        5: [(2, 1), (4, 0)]
    }

    print("0-1 BFS from vertex 0:")
    dist = zero_one_bfs(graph, 0, 6)
    for v in range(6):
        print(f"  dist[{v}] = {dist[v]}")

    print()

    # Grid example: reach goal with minimum wall-breaks
    # 0 = open (weight 0), 1 = wall (weight 1)
    grid = [
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    rows, cols = len(grid), len(grid[0])
    grid_graph = {}
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            grid_graph[node] = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb = nr * cols + nc
                    grid_graph[node].append((nb, grid[nr][nc]))

    dist = zero_one_bfs(grid_graph, 0, rows * cols)
    goal = rows * cols - 1
    print(f"Grid shortest path (0,0) to ({rows-1},{cols-1}):")
    print(f"  Minimum wall-breaks: {dist[goal]}")
```

**Output:**
```
0-1 BFS from vertex 0:
  dist[0] = 0
  dist[1] = 1
  dist[2] = 1
  dist[3] = 0
  dist[4] = 1
  dist[5] = 1

Grid shortest path (0,0) to (2,3):
  Minimum wall-breaks: 0
```

!!! tip "When to Use 0-1 BFS"
    Use 0-1 BFS whenever edge weights are restricted to $\{0, 1\}$. Common examples include: navigating grids where some cells are obstacles (breaking a wall costs 1, moving through open space costs 0), toggling switches, and problems where transformations are either free or have unit cost.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 24. MIT Press.
