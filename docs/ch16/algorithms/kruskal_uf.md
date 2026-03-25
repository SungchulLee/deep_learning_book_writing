# Kruskal with Union-Find

Kruskal's algorithm needs to answer one question repeatedly: "Are vertices $u$ and $v$ already in the same connected component?" A naive approach -- running BFS or DFS each time -- costs $O(V)$ per query, leading to $O(VE)$ total. The Union-Find (disjoint set) data structure reduces this to nearly constant time per query, making the sorting step the bottleneck rather than the connectivity checks.

## Why Union-Find Fits Kruskal's

Kruskal's algorithm processes edges in sorted order and performs two operations for each edge $(u, v)$:

1. **Query**: are $u$ and $v$ in the same component? (`FIND-SET`)
2. **Merge**: if not, combine their components (`UNION`)

These are exactly the operations that Union-Find provides. With union by rank and path compression, both operations run in amortized $O(\alpha(n))$ time, where $\alpha$ is the inverse Ackermann function -- effectively constant for all practical input sizes.

## Implementation

The implementation below uses two key optimizations:

- **Union by rank**: attach the shorter tree under the taller tree to keep heights small.
- **Path compression** (two-pass halving): during `find`, make each node point to its grandparent, flattening the tree incrementally.

```python
"""
Kruskal's MST algorithm with Union-Find.

Demonstrates the complete implementation using union by rank
and path compression for near-constant-time set operations.
"""


# === Union-Find data structure ===

class UnionFind:
    """Disjoint set forest with union by rank and path compression."""

    def __init__(self, n):
        """Initialize n singleton sets {0}, {1}, ..., {n-1}."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Return the root representative of x's set, with path halving."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        """Merge sets containing a and b. Return True if they were separate."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False  # already in the same set
        # union by rank: attach smaller tree under larger
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


# === Kruskal's algorithm ===

def kruskal(n, edges):
    """
    Compute the MST of a graph with n vertices.

    Parameters
    ----------
    n : int
        Number of vertices (labeled 0 to n-1).
    edges : list of (u, v, w)
        Edge list with integer endpoints and numeric weight w.

    Returns
    -------
    list of (u, v, w)
        Edges in the MST, in the order they were added.
    """
    edges.sort(key=lambda e: e[2])
    uf = UnionFind(n)
    mst = []
    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break  # MST complete
    return mst


# === Example ===

if __name__ == "__main__":
    #   0 ---4--- 1
    #   |  \      |
    #   1    3    2
    #   |      \  |
    #   2 ---5--- 3
    edges = [
        (0, 1, 4),
        (0, 2, 1),
        (1, 2, 3),
        (1, 3, 2),
        (2, 3, 5),
    ]
    result = kruskal(4, edges)
    total = sum(w for _, _, w in result)
    print(f"MST edges: {result}")
    print(f"Total weight: {total}")
```

**Output:**
```
MST edges: [(0, 2, 1), (1, 3, 2), (1, 2, 3)]
Total weight: 6
```

## Execution Trace

The table below shows the Union-Find state as Kruskal's algorithm processes each edge:

| Step | Edge | Weight | FIND(u) | FIND(v) | Action | Components |
|------|------|--------|---------|---------|--------|------------|
| 1 | (0, 2) | 1 | 0 | 2 | Union | {0, 2}, {1}, {3} |
| 2 | (1, 3) | 2 | 1 | 3 | Union | {0, 2}, {1, 3} |
| 3 | (1, 2) | 3 | 1 | 0 | Union | {0, 1, 2, 3} |
| 4 | (0, 1) | 4 | 0 | 0 | Skip | same component |
| 5 | (2, 3) | 5 | 0 | 0 | Skip | same component |

At step 3, we have $n - 1 = 3$ MST edges, so the algorithm terminates.

## Complexity Analysis

The total cost breaks down as follows:

**Sorting**: $O(E \log E)$. Since $E \le V^2$, this is also $O(E \log V)$.

**Union-Find operations**: the algorithm performs at most $2E$ `FIND` operations (two per edge) and at most $V - 1$ `UNION` operations. With union by rank and path compression, a sequence of $m$ operations on $n$ elements takes $O(m \cdot \alpha(n))$. Here $m = O(E)$ and $n = V$, giving $O(E \cdot \alpha(V))$.

**Total time**:

$$
T(V, E) = O(E \log E + E \cdot \alpha(V)) = O(E \log E)
$$

The sorting step dominates because $\alpha(V) \le 4$ for all practical values of $V$ (up to $2^{65536}$).

**Space**: $O(V + E)$ for the Union-Find arrays and the edge list.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *JACM*, 22(2), 215--225.
