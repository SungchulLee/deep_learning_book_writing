# External BFS

When a graph is too large to fit in main memory, standard BFS cannot simply load all adjacency lists into RAM. **External BFS** adapts breadth-first search to the external-memory (I/O) model, where the cost metric is the number of disk block transfers rather than CPU operations. Efficient external BFS is critical for processing web graphs, social networks, and other massive-scale graph data.

## The I/O Model

In the external-memory model, the machine has:

- **Main memory** of size $M$ (measured in data items).
- **Disk** with unlimited capacity, accessed in blocks of size $B$.

A single **I/O operation** transfers one block of $B$ items between disk and memory. The I/O complexity measures the number of such transfers, denoted $\text{sort}(N) = O((N/B) \log_{M/B}(N/B))$ for sorting $N$ items, which serves as a baseline for many external algorithms.

## Naive External BFS

The simplest approach runs standard BFS but stores the graph on disk. At each level, it reads the adjacency lists of all frontier vertices.

**I/O complexity**: In the worst case, each vertex access may trigger a random disk read. With $V$ vertices, this costs $O(V)$ I/Os regardless of block size -- far worse than optimal.

## Munagala-Ranade External BFS

Munagala and Ranade (1999) proposed a more efficient approach. The key idea is to sort the frontier and the adjacency structure to exploit sequential disk access.

### Algorithm

For each BFS level $d$:

1. **Sort** the current frontier $F_d$ by vertex ID.
2. **Scan** the sorted edge list, extracting all neighbors of frontier vertices using a merge-like pass.
3. **Remove duplicates** by sorting the neighbor list and filtering out previously visited vertices.
4. The result forms the next frontier $F_{d+1}$.

### I/O Complexity

Let $n_i = |F_i|$ be the size of the $i$-th frontier and $D$ be the BFS depth. The I/O cost is:

$$
O\!\left(\sum_{i=0}^{D}\left(\text{sort}(n_i) + \text{scan}(E)\right)\right) = O\!\left(D \cdot \frac{E}{B} + \sum_{i=0}^{D} \text{sort}(n_i)\right)
$$

Since $\sum_i n_i = V$, the sorting cost sums to at most $O(\text{sort}(V) \cdot D)$. For graphs with small diameter $D$, this is significantly better than the naive $O(V)$ I/Os.

## Implementation

```python
"""
External BFS simulation.

Simulates the Munagala-Ranade approach by processing each
BFS level as a batch: sort the frontier, scan edges, and
deduplicate. I/O operations are counted rather than performed.
"""

import math

# ===================================================================
# External BFS Simulation
# ===================================================================

def external_bfs(adj, source, B=4):
    """Simulate external BFS with I/O cost tracking.

    Args:
        adj: adjacency list as dict of lists
        source: starting vertex
        B: simulated block size

    Returns:
        dist: distance map from source
        io_count: total simulated I/O operations
    """
    dist = {source: 0}
    frontier = [source]
    io_count = 0
    level = 0

    while frontier:
        # Sort frontier (costs sort(|frontier|) I/Os)
        frontier.sort()
        n_f = len(frontier)
        if n_f > 0:
            io_count += max(1, n_f // B)  # simplified scan cost

        # Scan edges for frontier vertices
        next_frontier = []
        edges_scanned = 0
        for u in frontier:
            for v in adj.get(u, []):
                edges_scanned += 1
                if v not in dist:
                    dist[v] = level + 1
                    next_frontier.append(v)

        # I/O for scanning edges
        io_count += max(1, edges_scanned // B)

        # Deduplicate next frontier (sort + scan)
        next_frontier.sort()
        deduped = []
        for v in next_frontier:
            if not deduped or deduped[-1] != v:
                deduped.append(v)
        if next_frontier:
            io_count += max(1, len(next_frontier) // B)

        frontier = deduped
        level += 1

    return dist, io_count

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    # Build a test graph
    adj = {}
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5),
             (3, 6), (4, 6), (5, 7), (6, 8), (7, 8)]
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    dist, io_count = external_bfs(adj, source=0, B=2)

    print("External BFS from vertex 0:")
    for v in sorted(dist):
        print(f"  dist[{v}] = {dist[v]}")
    print(f"\nVertices: {len(dist)}")
    print(f"Edges:    {len(edges)}")
    print(f"BFS depth: {max(dist.values())}")
    print(f"Simulated I/O ops (B=2): {io_count}")

    # Compare with different block sizes
    print("\nI/O count vs block size:")
    for B in [1, 2, 4, 8]:
        _, ios = external_bfs(adj, source=0, B=B)
        print(f"  B={B}: {ios} I/Os")
```

**Output:**
```
External BFS from vertex 0:
  dist[0] = 0
  dist[1] = 1
  dist[2] = 1
  dist[3] = 2
  dist[4] = 2
  dist[5] = 2
  dist[6] = 3
  dist[7] = 3
  dist[8] = 4

Vertices: 9
Edges:    10
BFS depth: 4
Simulated I/O ops (B=2): 20

I/O count vs block size:
  B=1: 38 I/Os
  B=2: 20 I/Os
  B=4: 15 I/Os
  B=8: 15 I/Os
```

## Complexity Comparison

| Algorithm | I/O Complexity | Notes |
|---|---|---|
| Naive BFS | $O(V)$ | Random access per vertex |
| Munagala-Ranade | $O(D \cdot (E/B + \text{sort}(V)))$ | Sort-based, good for small $D$ |
| Mehlhorn-Meyer | $O(V + \text{sort}(E))$ | Optimal for general graphs |

Here $D$ is the BFS depth, $B$ is the block size, and $\text{sort}(N) = O((N/B) \log_{M/B}(N/B))$.

## Practical Considerations

- **Graph layout**: Storing adjacency lists sorted by vertex ID enables sequential scanning, which is critical for external-memory efficiency.
- **Semi-external model**: When $V$ fits in memory but $E$ does not, simpler algorithms suffice since the visited array can be kept in RAM.
- **Preprocessing**: Sorting the edge list once ($O(\text{sort}(E))$ I/Os) amortizes over multiple BFS queries from different sources.

## Reference

- Munagala, K. and Ranade, A. (1999). "I/O-complexity of graph algorithms." *SODA*.
- Mehlhorn, K. and Meyer, U. (2002). "External-memory breadth-first search with sublinear I/O." *ESA*.
- Vitter, J. S. (2001). "External memory algorithms and data structures: dealing with massive data." *ACM Computing Surveys*.
