# Range Updates

Many real-world problems require modifying contiguous blocks of data in bulk. Consider adjusting exam scores for an entire class, or applying a brightness offset to a row of pixels. A point-update segment tree handles each element individually in $O(\log n)$, so updating $k$ elements costs $O(k \log n)$. Range updates backed by **lazy propagation** reduce the cost of updating an arbitrary interval $[l, r]$ to a single $O(\log n)$ operation, regardless of the interval's length.

## Range Addition Update

The most common range update adds a value $\delta$ to every element in $[l, r]$:

$$
a[i] \leftarrow a[i] + \delta \quad \text{for all } l \leq i \leq r
$$

In the segment tree, every node whose range is fully contained in $[l, r]$ receives the update immediately. Nodes that partially overlap push the update to their children via lazy propagation. This deferred propagation is what keeps the per-operation cost logarithmic.

## Algorithm

The range update procedure visits a node covering $[lo, hi]$ and branches into one of three cases:

1. **No overlap** ($r < lo$ or $hi < l$): return immediately.
2. **Full containment** ($l \leq lo$ and $hi \leq r$): add $\delta \cdot (hi - lo + 1)$ to the node's stored sum and record $\delta$ in its lazy tag. Return without recursing.
3. **Partial overlap**: push down any existing lazy tag to the two children, recurse on both children, then recompute the node's value from its updated children.

The lazy tag at a node represents a deferred per-element addition that has not yet been propagated to that node's children. Each push-down transfers the tag one level deeper, clearing it at the current node.

## Simpler Alternatives

When only range updates and **point queries** are needed (no range queries), a **difference array** backed by a Fenwick tree (BIT) solves the problem with the same $O(\log n)$ bounds and simpler code. However, when both range updates and range queries are required, the lazy segment tree is the standard approach because a difference array alone cannot answer range-sum queries efficiently.

## Implementation

```python
"""
Segment tree with lazy propagation for range-add updates.

Supports two operations in O(log n) each:
  - range_update(l, r, delta): add delta to every element in [l, r]
  - range_query(l, r): return the sum of elements in [l, r]

The lazy tag at each node stores the pending per-element addition
that has not yet been pushed to its children.
"""


# === Segment Tree with Range Updates ===

class RangeUpdateSegTree:
    """Segment tree with lazy propagation for range add + range sum."""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
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

    def _push_down(self, node: int, lo: int, hi: int) -> None:
        """Propagate the lazy tag to children."""
        if self.lazy[node] != 0 and lo != hi:
            mid = (lo + hi) // 2
            left, right = 2 * node, 2 * node + 1

            self.tree[left] += self.lazy[node] * (mid - lo + 1)
            self.lazy[left] += self.lazy[node]

            self.tree[right] += self.lazy[node] * (hi - mid)
            self.lazy[right] += self.lazy[node]

            self.lazy[node] = 0

    def range_update(self, node: int, lo: int, hi: int,
                     l: int, r: int, delta: int) -> None:
        """Add delta to every element in [l, r].

        Args:
            node: Current position in the tree array.
            lo, hi: Range this node covers.
            l, r: Update range.
            delta: Value to add to each element.
        """
        if r < lo or hi < l:
            return
        if l <= lo and hi <= r:
            self.tree[node] += delta * (hi - lo + 1)
            self.lazy[node] += delta
            return
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        self.range_update(2 * node, lo, mid, l, r, delta)
        self.range_update(2 * node + 1, mid + 1, hi, l, r, delta)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def range_query(self, node: int, lo: int, hi: int,
                    l: int, r: int) -> int:
        """Return sum of elements in [l, r]."""
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return self.tree[node]
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        return (self.range_query(2 * node, lo, mid, l, r)
                + self.range_query(2 * node + 1, mid + 1, hi, l, r))

    def point_query(self, idx: int) -> int:
        """Return the current value of a[idx]."""
        return self.range_query(1, 0, self.n - 1, idx, idx)


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    n = len(data)
    st = RangeUpdateSegTree(data)

    print(f"Original: {data}")
    print(f"Sum [0,5] = {st.range_query(1, 0, n-1, 0, 5)}")
    print()

    # Range update: add 10 to [1, 4]
    print("Range update: add 10 to [1, 4]")
    st.range_update(1, 0, n - 1, 1, 4, 10)
    print(f"Sum [0,5] = {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [1,4] = {st.range_query(1, 0, n-1, 1, 4)}")
    print(f"a[0] = {st.point_query(0)}")
    print(f"a[2] = {st.point_query(2)}")
    print(f"a[5] = {st.point_query(5)}")
    print()

    # Another range update: add 5 to [0, 2]
    print("Range update: add 5 to [0, 2]")
    st.range_update(1, 0, n - 1, 0, 2, 5)
    print(f"Sum [0,5] = {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"a[0] = {st.point_query(0)}")
    print(f"a[1] = {st.point_query(1)}")
    print(f"a[2] = {st.point_query(2)}")
```

**Output:**
```
Original: [1, 3, 5, 7, 9, 11]
Sum [0,5] = 36

Range update: add 10 to [1, 4]
Sum [0,5] = 76
Sum [1,4] = 64
a[0] = 1
a[2] = 15
a[5] = 11

Range update: add 5 to [0, 2]
Sum [0,5] = 91
a[0] = 6
a[1] = 18
a[2] = 20
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Range update (add $\delta$ to $[l, r]$) | $O(\log n)$ | $O(\log n)$ stack |
| Range query after updates | $O(\log n)$ | $O(\log n)$ stack |
| Total space | — | $O(n)$ (tree + lazy arrays) |

Each operation visits at most $O(\log n)$ nodes. The recursion depth is bounded by the tree height, which is $\lceil \log_2 n \rceil$.

## Range Assignment

A useful variant **sets** all elements in $[l, r]$ to a value $v$ rather than adding $\delta$. The key difference lies in the lazy tag semantics: the tag now stores the assigned value, and a sentinel (typically `None` in Python, or $-1$ when values are non-negative) distinguishes "no pending assignment" from "assign zero." During push-down, children's values are **replaced** rather than incremented.

!!! warning "Composing Different Operations"
    When both range-add and range-assign operations coexist on the same tree, the lazy tag must store both components. During push-down, the assignment is applied first (it overwrites), then the addition is applied on top. Reversing this order produces incorrect results.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
