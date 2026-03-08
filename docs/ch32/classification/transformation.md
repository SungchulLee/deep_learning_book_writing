# Problem Transformation

**Problem transformation** (or **reduction**) is the technique of converting an unfamiliar problem into a well-known one. Instead of inventing a new algorithm, you reformulate the problem so that an existing algorithm applies directly.

## Common Transformations

| Original Problem | Transform To | Algorithm |
|---|---|---|
| Longest increasing subsequence | Patience sorting / binary search | $O(n \log n)$ |
| Bipartite matching | Max flow | $O(V \cdot E)$ |
| 2-SAT | SCC on implication graph | $O(V + E)$ |
| Meeting rooms (min rooms) | Sweep line / sort events | $O(n \log n)$ |
| Median maintenance | Two heaps | $O(\log n)$ per insert |
| Matrix shortest path | Graph BFS/Dijkstra | Standard |
| Difference constraints | Shortest path (Bellman-Ford) | $O(VE)$ |

## Example: Grid as Graph

A common transformation converts a 2D grid into a graph. Each cell $(i, j)$ becomes a node, and adjacent cells become edges.

**Problem:** Find the shortest path in a grid from top-left to bottom-right, where each cell has a cost.

**Transformation:** Model as a weighted graph and apply Dijkstra.

```python
import heapq

def shortest_path_grid(grid):
    rows, cols = len(grid), len(grid[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[0][0] = grid[0][0]
    pq = [(grid[0][0], 0, 0)]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r][c]:
            continue
        if r == rows - 1 and c == cols - 1:
            return d
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nd = d + grid[nr][nc]
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
    return dist[rows-1][cols-1]

grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(shortest_path_grid(grid))  # Output: 7
```

## Example: Difference Constraints to Shortest Path

A system of difference constraints $x_j - x_i \le w_{ij}$ can be solved by constructing a graph with edge $(i, j)$ of weight $w_{ij}$ and running Bellman-Ford from a virtual source. If there is a negative cycle, the system has no solution.

## Why Transformation Matters

Problem transformation is arguably the most powerful meta-technique because it lets you leverage the entire body of known algorithms. The key insight is that you do not need a new algorithm -- you need to see the known algorithm hiding inside your problem.

# Reference

- Cormen, T. et al. *Introduction to Algorithms*, Chapter 24.4 (Difference Constraints), MIT Press, 2022.
- Skiena, S. *The Algorithm Design Manual*, Chapter 1 (Modeling), Springer, 2020.
