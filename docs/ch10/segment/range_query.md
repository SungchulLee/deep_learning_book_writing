# Range Queries

The primary purpose of a segment tree is answering **range queries** — computing an aggregate (sum, minimum, maximum, GCD, etc.) over any contiguous subarray $a[l..r]$ in $O(\log n)$ time. This page explains how the recursive query algorithm works by decomposing $[l, r]$ into a minimal set of pre-computed node ranges, and proves the $O(\log n)$ bound.

## The Three Cases

When querying a node that covers range $[lo, hi]$ for the query range $[l, r]$, exactly one of three situations arises:

1. **No overlap** ($r < lo$ or $hi < l$). The query range does not intersect this node's range. Return the identity element (e.g., 0 for sum, $+\infty$ for min).

2. **Full containment** ($l \leq lo$ and $hi \leq r$). The node's range is entirely inside the query range. Return the node's pre-computed value directly — no recursion needed.

3. **Partial overlap.** The query range partially overlaps the node's range. Recurse on both children, then combine their results.

!!! note "Why Three Cases Suffice"
    Every node in the tree falls into exactly one of these three categories. Cases 1 and 2 terminate immediately, while case 3 decomposes the problem into two subproblems that get closer to the leaves. This guarantees termination and correctness.

## Step-by-Step Trace

Consider the array $a = [1, 3, 5, 7, 9]$ stored in a segment tree.

**Query: sum over [1, 3]**

Starting from the root (node 1, range $[0, 4]$):

| Node | Range | Case | Action |
|:----:|:-----:|:----:|--------|
| 1 | $[0,4]$ | Partial | Split: query children |
| 2 | $[0,2]$ | Partial | Split: query children |
| 3 | $[3,4]$ | Partial | Split: query children |
| 4 | $[0,1]$ | Partial | Split: query children |
| 5 | $[2,2]$ | Full | Return 5 |
| 6 | $[3,3]$ | Full | Return 7 |
| 7 | $[4,4]$ | No overlap | Return 0 |
| 8 | $[0,0]$ | No overlap | Return 0 |
| 9 | $[1,1]$ | Full | Return 3 |

Result: $3 + 5 + 7 = 15$.

The query visits 9 nodes, but returns immediately from 5 of them (cases 1 and 2). At most $O(\log n)$ nodes contribute non-trivially.

## Why the Query is O(log n)

At each level of the tree, the query visits at most 4 nodes, but at most 2 of them result in further recursion (case 3). The argument is:

- At the top level, there is 1 active node.
- At each level, partial-overlap nodes produce at most 2 children that continue, while fully contained or non-overlapping children stop.
- Since the tree has $O(\log n)$ levels, the total work is $O(\log n)$.

More precisely, at any level at most 2 nodes can have partial overlap with $[l, r]$: one at the left boundary and one at the right boundary. All nodes between them are fully contained.

## Implementation

```python
"""
Segment tree range queries.

Demonstrates the O(log n) range query algorithm using the
three-case decomposition: no overlap, full containment,
and partial overlap.
"""


# === Segment Tree with Range Queries ===

class SegmentTree:
    """Segment tree for range sum queries."""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(data, 1, 0, self.n - 1)

    def _build(self, data: list, node: int, lo: int, hi: int) -> None:
        if lo == hi:
            self.tree[node] = data[lo]
            return
        mid = (lo + hi) // 2
        self._build(data, 2 * node, lo, mid)
        self._build(data, 2 * node + 1, mid + 1, hi)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node: int, lo: int, hi: int,
              l: int, r: int) -> int:
        """Return sum of a[l..r] using the three-case decomposition.

        Args:
            node: Current position in the tree array.
            lo, hi: Range this node covers.
            l, r: Query range.
        """
        # Case 1: No overlap
        if r < lo or hi < l:
            return 0

        # Case 2: Full containment
        if l <= lo and hi <= r:
            return self.tree[node]

        # Case 3: Partial overlap — recurse on both children
        mid = (lo + hi) // 2
        left_sum = self.query(2 * node, lo, mid, l, r)
        right_sum = self.query(2 * node + 1, mid + 1, hi, l, r)
        return left_sum + right_sum

    def update(self, node: int, lo: int, hi: int,
               idx: int, val: int) -> None:
        """Point update: set a[idx] = val."""
        if lo == hi:
            self.tree[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            self.update(2 * node, lo, mid, idx, val)
        else:
            self.update(2 * node + 1, mid + 1, hi, idx, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    n = len(data)
    st = SegmentTree(data)

    print(f"Array: {data}")
    print()

    # Various range queries
    queries = [(0, 4), (1, 3), (0, 0), (2, 4), (3, 3)]
    for l, r in queries:
        result = st.query(1, 0, n - 1, l, r)
        expected = sum(data[l:r + 1])
        print(f"  Sum [{l},{r}] = {result}  (expected {expected})")

    # After point update
    print()
    st.update(1, 0, n - 1, 2, 50)
    data[2] = 50
    print("After setting a[2] = 50:")
    for l, r in queries:
        result = st.query(1, 0, n - 1, l, r)
        expected = sum(data[l:r + 1])
        print(f"  Sum [{l},{r}] = {result}  (expected {expected})")
```

**Output:**
```
Array: [1, 3, 5, 7, 9]

  Sum [0,4] = 25  (expected 25)
  Sum [1,3] = 15  (expected 15)
  Sum [0,0] = 1  (expected 1)
  Sum [2,4] = 21  (expected 21)
  Sum [3,3] = 7  (expected 7)

After setting a[2] = 50:
  Sum [0,4] = 70  (expected 70)
  Sum [1,3] = 60  (expected 60)
  Sum [0,0] = 1  (expected 1)
  Sum [2,4] = 66  (expected 66)
  Sum [3,3] = 7  (expected 7)
```

## Complexity

| Aspect | Bound |
|--------|-------|
| Time per query | $O(\log n)$ |
| Space per query | $O(\log n)$ stack frames (recursion depth) |

The $O(\log n)$ bound follows because at most 2 nodes at each of the $O(\log n)$ levels have partial overlap. All other nodes are resolved in $O(1)$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
