# Point Updates

After building a segment tree, the underlying array may change. A **point update** modifies a single element $a[\text{idx}]$ and then recomputes every ancestor node on the path from the updated leaf back to the root. Because the tree has height $O(\log n)$, this path has at most $\lfloor \log_2 n \rfloor + 1$ nodes, yielding an $O(\log n)$ update.

## Algorithm

To set $a[\text{idx}]$ to a new value $v$:

1. **Navigate to the leaf.** Starting from the root (range $[0, n-1]$), compute $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$. If $\text{idx} \leq \text{mid}$, recurse on the left child; otherwise recurse on the right child.
2. **Update the leaf.** When $lo = hi = \text{idx}$, set `tree[node] = v`.
3. **Recompute ancestors.** On the way back up the recursion, recompute each internal node as the aggregate of its two children: `tree[node] = tree[2*node] + tree[2*node+1]`.

The recursion visits exactly one child at each level, so the total number of nodes visited is $h + 1$ where $h$ is the height of the tree.

## Step-by-Step Trace

Consider the array $a = [1, 3, 5, 7, 9]$ stored in a segment tree. We update position 2 from 5 to 50.

**Before update:**

| Node | Range | Value |
|:----:|:-----:|:-----:|
| 1 | $[0,4]$ | 25 |
| 2 | $[0,2]$ | 9 |
| 3 | $[3,4]$ | 16 |
| 4 | $[0,1]$ | 4 |
| 5 | $[2,2]$ | 5 |

**Update(idx=2, val=50):**

1. Node 1 ($[0,4]$): $\text{mid}=2$, $\text{idx}=2 \leq 2$ → go left to node 2.
2. Node 2 ($[0,2]$): $\text{mid}=1$, $\text{idx}=2 > 1$ → go right to node 5.
3. Node 5 ($[2,2]$): leaf, set value to 50.
4. Return to node 2: recompute $4 + 50 = 54$.
5. Return to node 1: recompute $54 + 16 = 70$.

**After update:**

| Node | Range | Value |
|:----:|:-----:|:-----:|
| 1 | $[0,4]$ | 70 |
| 2 | $[0,2]$ | 54 |
| 5 | $[2,2]$ | 50 |

Only 3 nodes changed — all on the root-to-leaf path for position 2.

## Implementation

```python
"""
Segment tree point updates.

Demonstrates the O(log n) point update operation that modifies
a leaf and recomputes all ancestors on the root-to-leaf path.
"""


# === Segment Tree with Point Updates ===

class SegmentTree:
    """Segment tree for sum queries with point updates."""

    def __init__(self, data: list):
        """Build the segment tree from input data."""
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

    def update(self, node: int, lo: int, hi: int,
               idx: int, val: int) -> None:
        """Set a[idx] = val and recompute all ancestors.

        Args:
            node: Current node index in the tree array.
            lo, hi: Range this node covers.
            idx: Position in the original array to update.
            val: New value for a[idx].
        """
        if lo == hi:
            self.tree[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            self.update(2 * node, lo, mid, idx, val)
        else:
            self.update(2 * node + 1, mid + 1, hi, idx, val)
        # Recompute this node from its children
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node: int, lo: int, hi: int,
              l: int, r: int) -> int:
        """Range sum query over [l, r]."""
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return self.tree[node]
        mid = (lo + hi) // 2
        return (self.query(2 * node, lo, mid, l, r)
                + self.query(2 * node + 1, mid + 1, hi, l, r))


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    n = len(data)
    st = SegmentTree(data)

    print(f"Original array: {data}")
    print(f"Sum [0,4] = {st.query(1, 0, n-1, 0, 4)}")
    print(f"Sum [1,3] = {st.query(1, 0, n-1, 1, 3)}")
    print()

    # Point update: set a[2] = 50
    print("Update: a[2] = 50")
    st.update(1, 0, n - 1, 2, 50)
    print(f"Sum [0,4] = {st.query(1, 0, n-1, 0, 4)}")
    print(f"Sum [1,3] = {st.query(1, 0, n-1, 1, 3)}")
    print(f"Sum [2,2] = {st.query(1, 0, n-1, 2, 2)}")
    print()

    # Another update: set a[0] = 100
    print("Update: a[0] = 100")
    st.update(1, 0, n - 1, 0, 100)
    print(f"Sum [0,4] = {st.query(1, 0, n-1, 0, 4)}")
    print(f"Sum [0,0] = {st.query(1, 0, n-1, 0, 0)}")
```

**Output:**
```
Original array: [1, 3, 5, 7, 9]
Sum [0,4] = 25
Sum [1,3] = 15

Update: a[2] = 50
Sum [0,4] = 70
Sum [1,3] = 60
Sum [2,2] = 50

Update: a[0] = 100
Sum [0,4] = 169
Sum [0,0] = 100
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Point update | $O(\log n)$ | $O(1)$ auxiliary |

The update touches exactly $h + 1$ nodes where $h = \lfloor \log_2 n \rfloor$ is the tree height. Each node requires $O(1)$ work (one comparison and one addition).

!!! tip "Update vs Replace"
    The implementation above **sets** the value at a position. To **add** a delta instead, change the leaf update from `tree[node] = val` to `tree[node] += val`. The ancestor recomputation remains the same.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
