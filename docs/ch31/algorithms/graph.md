# External Graph Algorithms

Graphs arising in practice -- web link structures, social networks, road networks -- often contain billions of vertices and edges, far exceeding main memory capacity. **External graph algorithms** process such graphs efficiently by minimizing disk I/O operations rather than CPU time. This page introduces the fundamental techniques that enable graph computation on disk-resident data.

## The I/O Model for Graphs

Recall the external-memory model parameters:

- **$N$**: number of data items (vertices or edges).
- **$M$**: main memory capacity (in items).
- **$B$**: disk block size (items per block transfer).

The I/O cost of scanning $N$ items is $\text{scan}(N) = O(N/B)$, and sorting costs $\text{sort}(N) = O((N/B) \log_{M/B}(N/B))$.

For a graph $G = (V, E)$, we typically have $|V| = n$ and $|E| = m$. Graph algorithms face a fundamental challenge: graph traversals follow pointers (edges) that may point to arbitrary locations on disk, causing random I/O.

## Graph Representations on Disk

The choice of disk layout significantly affects I/O performance.

### Adjacency Array

Store vertices in sorted order, with each vertex pointing to a contiguous block of its neighbors:

$$
\text{I/O to scan all edges} = O\!\left(\frac{m}{B}\right)
$$

This representation supports efficient sequential scans and is the default for external graph algorithms.

### Edge List

Store all edges $(u, v)$ as a flat sorted list. Sorting by source vertex costs $O(\text{sort}(m))$ I/Os and enables efficient merge-based algorithms.

## Key Algorithmic Techniques

### Technique 1: Sort and Scan

Many graph problems reduce to sorting edges by some key and scanning the sorted list. For example, computing vertex degrees requires only sorting the edge list and scanning it, costing $O(\text{sort}(m))$ I/Os.

### Technique 2: Time-Forward Processing

When processing a DAG in topological order, each vertex may need data from its predecessors. **Time-forward processing** sends messages along edges to future vertices using a priority queue, avoiding random access to predecessor data.

- Vertices are processed in topological order.
- When vertex $u$ is processed, it sends results to each successor $v$ via the priority queue.
- When $v$ is processed, it retrieves all messages addressed to it.

The I/O cost is $O(\text{sort}(m))$ for the priority queue operations.

### Technique 3: Graph Contraction

Reduce the graph size by contracting vertices (e.g., removing degree-1 vertices, merging degree-2 paths). Process the contracted graph in memory, then expand the solution back.

## Example: External DFS

External DFS is significantly harder than external BFS because the DFS stack may require random access to the graph.

```python
"""
External graph algorithm fundamentals.

Demonstrates sorting-based edge processing and vertex degree
computation in the external-memory style (batch processing).
"""

# ===================================================================
# External-Memory Style Graph Processing
# ===================================================================

def compute_degrees_external(edges, n):
    """Compute vertex degrees using sort-and-scan approach.

    In an external-memory setting, this sorts the edge list
    and scans it to count degrees, avoiding random access.

    Args:
        edges: list of (u, v) tuples
        n: number of vertices

    Returns:
        List of degrees indexed by vertex
    """
    # Create directed edge entries (both directions)
    directed = []
    for u, v in edges:
        directed.append(u)
        directed.append(v)

    # Sort (external sort in real implementation)
    directed.sort()

    # Scan to count degrees
    degrees = [0] * n
    for vertex in directed:
        degrees[vertex] += 1

    return degrees


def connected_components_external(edges, n, B=4):
    """Find connected components using iterative label propagation.

    Simulates an external-memory approach where each round
    propagates the minimum label along edges.

    Args:
        edges: list of (u, v) tuples
        n: number of vertices
        B: simulated block size

    Returns:
        Component labels and I/O count
    """
    labels = list(range(n))
    io_count = 0
    changed = True

    while changed:
        changed = False
        # Sort edges by label (external sort)
        io_count += max(1, len(edges) // B)

        for u, v in edges:
            new_label = min(labels[u], labels[v])
            if labels[u] != new_label:
                labels[u] = new_label
                changed = True
            if labels[v] != new_label:
                labels[v] = new_label
                changed = True

        # Scan cost
        io_count += max(1, len(edges) // B)

    return labels, io_count

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (7, 8)]
    n = 9

    # Degree computation
    degrees = compute_degrees_external(edges, n)
    print("Degree computation (sort-and-scan):")
    for v in range(n):
        print(f"  deg({v}) = {degrees[v]}")

    # Connected components
    labels, ios = connected_components_external(edges, n)
    print(f"\nConnected components:")
    components = {}
    for v in range(n):
        root = labels[v]
        components.setdefault(root, []).append(v)
    for root, members in sorted(components.items()):
        print(f"  Component {root}: {members}")
    print(f"Simulated I/Os: {ios}")
```

**Output:**
```
Degree computation (sort-and-scan):
  deg(0) = 1
  deg(1) = 2
  deg(2) = 2
  deg(3) = 1
  deg(4) = 1
  deg(5) = 2
  deg(6) = 1
  deg(7) = 1
  deg(8) = 1

Connected components:
  Component 0: [0, 1, 2, 3]
  Component 4: [4, 5, 6]
  Component 7: [7, 8]
Simulated I/Os: 4
```

## I/O Complexity Summary

| Problem | I/O Complexity | Technique |
|---|---|---|
| Scanning all edges | $O(m/B)$ | Sequential scan |
| Sorting edges | $O(\text{sort}(m))$ | External merge sort |
| BFS | $O(n + m/B)$ to $O(D \cdot \text{sort}(m))$ | Level-synchronous |
| DFS | $O((n + m/B) \cdot n/M)$ | Complicated; open problem for optimal |
| Connected components | $O(\text{sort}(m))$ | Contraction + label propagation |
| MST | $O(\text{sort}(m))$ | See [External MST](mst.md) |

## Reference

- Vitter, J. S. (2001). "External memory algorithms and data structures: dealing with massive data." *ACM Computing Surveys*, 33(2), 209--271.
- Arge, L. (2003). "The buffer tree: a technique for designing batched external data structures." *Algorithmica*.
