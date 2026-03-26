# Union-Find Applications

Union-Find is one of those data structures that appears in unexpected places. Its ability to maintain dynamic connected components with near-constant-time operations makes it the backbone of algorithms in graph theory, image processing, network analysis, and many competitive programming problems. This page surveys the most important applications.

## Kruskal's MST Algorithm

The most classical application of Union-Find is in Kruskal's algorithm for minimum spanning trees. The algorithm processes edges in increasing weight order and adds an edge $(u, v)$ to the MST only if $u$ and $v$ are in different components. Union-Find makes this connectivity check efficient:

- `find(u) != find(v)` determines whether the edge creates a cycle.
- `union(u, v)` merges the two components when the edge is added.

With Union-Find, Kruskal's algorithm runs in $O(m \log m)$ time (dominated by sorting edges), with the Union-Find operations contributing only $O(m \cdot \alpha(n))$.

## Dynamic Connectivity

Given a graph where edges are added one at a time, Union-Find answers "Are $u$ and $v$ connected?" after each addition. This is the online connected components problem, and Union-Find solves it optimally.

## Cycle Detection in Undirected Graphs

When processing the edges of an undirected graph, an edge $(u, v)$ creates a cycle if and only if $u$ and $v$ are already in the same component. This gives an $O(m \cdot \alpha(n))$ cycle detection algorithm that is simpler than DFS-based approaches for some use cases.

## Image Segmentation (Connected Components Labeling)

In image processing, **connected component labeling** identifies connected regions of pixels sharing the same property (e.g., color or intensity). Union-Find processes pixels in raster order: when a pixel matches its neighbor, union them. After processing all pixels, each component has a unique label. This runs in near-linear time in the number of pixels.

## Network Percolation

In percolation theory, sites on a grid are randomly "opened" and we ask whether a path exists from top to bottom. Union-Find tracks connected clusters as sites are opened. By adding virtual top and bottom nodes, a single `connected(top, bottom)` query determines whether percolation has occurred.

## Implementation

```python
"""
Union-Find applications: Kruskal's MST and cycle detection.

Demonstrates two common uses of Union-Find in graph algorithms:
building minimum spanning trees and detecting cycles in
undirected graphs.
"""


# === Union-Find ===

class UnionFind:
    """Union-Find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """Union by rank. Returns True if a merge occurred."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """Check if a and b are in the same component."""
        return self.find(a) == self.find(b)


# === Kruskal's MST ===

def kruskal_mst(n: int, edges: list) -> list:
    """Find MST using Kruskal's algorithm with Union-Find.

    Args:
        n: Number of vertices.
        edges: List of (weight, u, v) tuples.

    Returns:
        List of (weight, u, v) edges in the MST.
    """
    edges_sorted = sorted(edges)
    uf = UnionFind(n)
    mst = []

    for w, u, v in edges_sorted:
        if uf.union(u, v):
            mst.append((w, u, v))
            if len(mst) == n - 1:
                break

    return mst


# === Cycle Detection ===

def has_cycle(n: int, edges: list) -> bool:
    """Detect if an undirected graph has a cycle using Union-Find.

    Args:
        n: Number of vertices.
        edges: List of (u, v) tuples.

    Returns:
        True if the graph contains a cycle.
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False


# === Demonstration ===

if __name__ == "__main__":
    # Kruskal's MST
    edges = [
        (4, 0, 1), (8, 0, 7), (11, 1, 7), (8, 1, 2),
        (7, 2, 3), (4, 2, 5), (2, 2, 8), (9, 3, 4),
        (14, 3, 5), (10, 4, 5), (2, 5, 6), (1, 6, 7),
        (6, 6, 8), (7, 7, 8)
    ]
    mst = kruskal_mst(9, edges)
    total = sum(w for w, u, v in mst)
    print("Kruskal's MST:")
    for w, u, v in mst:
        print(f"  ({u},{v}) weight={w}")
    print(f"Total MST weight: {total}")
    print()

    # Cycle detection
    print("Cycle detection:")
    edges_no_cycle = [(0, 1), (1, 2), (2, 3)]
    print(f"  Tree edges {edges_no_cycle}: "
          f"has_cycle={has_cycle(4, edges_no_cycle)}")

    edges_with_cycle = [(0, 1), (1, 2), (2, 3), (3, 0)]
    print(f"  Cycle edges {edges_with_cycle}: "
          f"has_cycle={has_cycle(4, edges_with_cycle)}")
```

**Output:**
```
Kruskal's MST:
  (6,7) weight=1
  (2,8) weight=2
  (5,6) weight=2
  (0,1) weight=4
  (2,5) weight=4
  (2,3) weight=7
  (0,7) weight=8
  (3,4) weight=9
Total MST weight: 37

Cycle detection:
  Tree edges [(0, 1), (1, 2), (2, 3)]: has_cycle=False
  Cycle edges [(0, 1), (1, 2), (2, 3), (3, 0)]: has_cycle=True
```

## Summary of Applications

| Application | Union-Find Role | Total Time |
|-------------|----------------|------------|
| Kruskal's MST | Cycle avoidance | $O(m \log m)$ |
| Dynamic connectivity | Online component tracking | $O(m \cdot \alpha(n))$ |
| Cycle detection | Same-component check | $O(m \cdot \alpha(n))$ |
| Image segmentation | Pixel region merging | $O(\text{pixels} \cdot \alpha(\text{pixels}))$ |
| Network percolation | Cluster tracking | $O(\text{sites} \cdot \alpha(\text{sites}))$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 19, 21. MIT Press.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.), Chapter 1.5. Addison-Wesley.
