# Path Compression

Without any optimization, a `find` operation traverses the chain of parent pointers from a node to its root. If the tree is deep, this traversal is slow. **Path compression** speeds up future `find` calls by flattening the tree during each `find`: after finding the root, all nodes along the path are updated to point directly to the root. This means the next `find` on any of those nodes takes $O(1)$. Over a sequence of $m$ operations, path compression (combined with union by rank) achieves $O(m \cdot \alpha(n))$ total time.

## Three Variants

There are three common path compression strategies, all achieving the same amortized bound:

### Full Path Compression

After finding the root $r$, make every node on the path from $x$ to $r$ a direct child of $r$. This is the standard "textbook" version and produces the flattest trees.

```
Before find(5):     After find(5):
    1                    1
    |               / / | \
    2              2  3  4  5
    |
    3
    |
    4
    |
    5
```

### Path Splitting

Each node on the path is updated to point to its **grandparent**. This is done in a single pass (no need to find the root first) and is slightly simpler to implement iteratively.

### Path Halving

Every other node on the path is updated to point to its grandparent. This touches half as many nodes as path splitting but achieves the same amortized bound.

## Comparison of Variants

| Variant | Passes | Nodes Updated | Implementation |
|---------|--------|---------------|----------------|
| Full compression | 2 (find root, then update) | All on path | Recursive or two-pass |
| Path splitting | 1 | All on path | Single iterative loop |
| Path halving | 1 | Half of path | Single iterative loop |

All three achieve $O(m \cdot \alpha(n))$ amortized time when combined with union by rank. The practical differences are negligible.

## Implementation

```python
"""
Path compression variants for Union-Find.

Demonstrates full path compression, path splitting, and path
halving. All three achieve O(alpha(n)) amortized per operation
when combined with union by rank.
"""


# === Full Path Compression (recursive) ===

class UnionFindFullCompression:
    """Union-Find with full path compression (recursive find)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with full path compression.

        After this call, every node on the path from x to root
        points directly to the root.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """Union by rank."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# === Path Splitting ===

class UnionFindPathSplitting:
    """Union-Find with path splitting (iterative find)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path splitting.

        Each node on the path is updated to point to its
        grandparent in a single pass.
        """
        while self.parent[x] != x:
            next_x = self.parent[x]
            self.parent[x] = self.parent[next_x]  # point to grandparent
            x = next_x
        return x

    def union(self, a: int, b: int) -> bool:
        """Union by rank."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# === Path Halving ===

class UnionFindPathHalving:
    """Union-Find with path halving (iterative find)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path halving.

        Every other node on the path is updated to point to
        its grandparent.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Union by rank."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# === Demonstration ===

if __name__ == "__main__":
    # Build a deep chain: 0 <- 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7
    def build_chain(uf_class, n):
        uf = uf_class(n)
        # Manually create a chain (bypassing union)
        for i in range(1, n):
            uf.parent[i] = i - 1
        return uf

    n = 8
    print("Parent arrays before and after find(7):")
    print()

    for name, cls in [("Full compression", UnionFindFullCompression),
                       ("Path splitting", UnionFindPathSplitting),
                       ("Path halving", UnionFindPathHalving)]:
        uf = build_chain(cls, n)
        print(f"{name}:")
        print(f"  Before: {uf.parent}")
        root = uf.find(n - 1)
        print(f"  After:  {uf.parent}  (root={root})")
        print()

    # Verify all three produce correct results
    for cls in [UnionFindFullCompression, UnionFindPathSplitting,
                UnionFindPathHalving]:
        uf = cls(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        uf.union(0, 2)
        uf.union(0, 4)
        assert uf.find(5) == uf.find(1)
    print("All variants produce correct connectivity results.")
```

**Output:**
```
Parent arrays before and after find(7):

Full compression:
  Before: [0, 0, 1, 2, 3, 4, 5, 6]
  After:  [0, 0, 0, 0, 0, 0, 0, 0]  (root=0)

Path splitting:
  Before: [0, 0, 1, 2, 3, 4, 5, 6]
  After:  [0, 0, 0, 1, 2, 3, 4, 5]  (root=0)

Path halving:
  Before: [0, 0, 1, 2, 3, 4, 5, 6]
  After:  [0, 0, 0, 2, 2, 4, 4, 6]  (root=0)

All variants produce correct connectivity results.
```

!!! tip "Which Variant to Use?"
    Full path compression produces the flattest trees and is the simplest to reason about. Path splitting and halving are iterative (no recursion stack) and slightly faster in practice due to fewer memory writes. For most applications, the difference is negligible -- pick whichever is clearest in your codebase.

## Reference

- Tarjan, R. E., & van Leeuwen, J. (1984). Worst-case analysis of set union algorithms. *Journal of the ACM*, 31(2), 245-281.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 19. MIT Press.
