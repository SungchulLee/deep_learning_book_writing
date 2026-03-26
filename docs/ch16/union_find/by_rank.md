# Union by Rank

In a naive Union-Find implementation, the `union` operation attaches one tree's root as a child of the other without regard to tree structure. This can produce a long chain (a degenerate tree of height $n - 1$), making `find` operations take $O(n)$ time. **Union by rank** prevents this by always attaching the shorter tree under the root of the taller tree, keeping the tree height bounded by $O(\log n)$. Combined with path compression, this yields the near-optimal $O(\alpha(n))$ amortized bound.

## What Is Rank?

The **rank** of a node is an upper bound on the height of its subtree. Initially, every node is its own root with rank 0 (a single-node tree has height 0). When two trees are merged:

- If they have **different ranks**, the root with the smaller rank becomes a child of the root with the larger rank. The larger root's rank stays the same, since the height of the merged tree has not increased.
- If they have **equal ranks**, one root becomes a child of the other, and the surviving root's rank increases by 1. Attaching a tree of height $r$ under another tree of height $r$ creates a tree of height $r + 1$.

## Rank Properties

Union by rank maintains several important invariants:

1. A node of rank $r$ has at least $2^r$ descendants. This follows by induction: a rank increase only happens when two rank-$r$ subtrees merge, doubling the minimum size.
2. The maximum rank is at most $\lfloor \log_2 n \rfloor$, since $2^r \leq n$ implies $r \leq \log_2 n$.
3. At most $n / 2^r$ nodes have rank $\geq r$, bounding how many nodes share each rank level.

These properties guarantee that `find` operations traverse at most $O(\log n)$ parent pointers, even without path compression.

## Union by Rank vs Union by Size

An alternative is **union by size**, which attaches the smaller tree (by node count) under the root of the larger tree. Both strategies achieve $O(\log n)$ tree height. The differences are subtle:

| Property | Union by Rank | Union by Size |
|----------|--------------|---------------|
| Stored value | Upper bound on height | Exact subtree count |
| Update rule | Increment on equal-rank merge | Sum sizes |
| With path compression | Rank may overestimate height | Size remains exact |
| Amortized bound | $O(\alpha(n))$ | $O(\alpha(n))$ |

Union by rank is preferred in theoretical analysis because rank properties are simpler to prove. Union by size is sometimes preferred in practice because the size count has secondary uses (e.g., reporting component sizes).

## Implementation

```python
"""
Union by rank optimization for Union-Find.

Keeps trees balanced by attaching the shorter tree under the
taller tree's root. Guarantees O(log n) find without path
compression, and O(alpha(n)) amortized with path compression.
"""


# === Union-Find with Union by Rank ===

class UnionFindByRank:
    """Union-Find with union by rank and full path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """Union by rank. Returns True if a merge occurred."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Attach smaller-rank tree under larger-rank root
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# === Union-Find with Union by Size (comparison) ===

class UnionFindBySize:
    """Union-Find with union by size and full path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """Union by size. Returns True if a merge occurred."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def component_size(self, x: int) -> int:
        """Return the size of the component containing x."""
        return self.size[self.find(x)]


# === Demonstration ===

if __name__ == "__main__":
    print("=== Union by Rank ===")
    uf_rank = UnionFindByRank(8)
    merges = [(0, 1), (2, 3), (4, 5), (6, 7),
              (0, 2), (4, 6), (0, 4)]
    for a, b in merges:
        uf_rank.union(a, b)
        print(f"union({a},{b}): ranks={uf_rank.rank}")

    print()
    print("=== Union by Size ===")
    uf_size = UnionFindBySize(8)
    for a, b in merges:
        uf_size.union(a, b)
        print(f"union({a},{b}): sizes={uf_size.size}")

    print()
    print(f"Component size of 5: {uf_size.component_size(5)}")

    # Show that both produce the same connectivity
    for i in range(8):
        for j in range(i + 1, 8):
            rank_conn = uf_rank.find(i) == uf_rank.find(j)
            size_conn = uf_size.find(i) == uf_size.find(j)
            assert rank_conn == size_conn
    print("Both methods produce identical connectivity.")
```

**Output:**
```
=== Union by Rank ===
union(0,1): ranks=[1, 0, 0, 0, 0, 0, 0, 0]
union(2,3): ranks=[1, 0, 1, 0, 0, 0, 0, 0]
union(4,5): ranks=[1, 0, 1, 0, 1, 0, 0, 0]
union(6,7): ranks=[1, 0, 1, 0, 1, 0, 1, 0]
union(0,2): ranks=[2, 0, 1, 0, 1, 0, 1, 0]
union(4,6): ranks=[2, 0, 1, 0, 2, 0, 1, 0]
union(0,4): ranks=[3, 0, 1, 0, 2, 0, 1, 0]

=== Union by Size ===
union(0,1): sizes=[2, 1, 1, 1, 1, 1, 1, 1]
union(2,3): sizes=[2, 1, 2, 1, 1, 1, 1, 1]
union(4,5): sizes=[2, 1, 2, 1, 2, 1, 1, 1]
union(6,7): sizes=[2, 1, 2, 1, 2, 1, 2, 1]
union(0,2): sizes=[4, 1, 2, 1, 2, 1, 2, 1]
union(4,6): sizes=[4, 1, 2, 1, 4, 1, 2, 1]
union(0,4): sizes=[8, 1, 2, 1, 4, 1, 2, 1]

Component size of 5: 8
Both methods produce identical connectivity.
```

## Reference

- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215-225.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 19. MIT Press.
