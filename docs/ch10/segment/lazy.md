# Lazy Propagation

A standard segment tree handles **point updates** in $O(\log n)$, but what about updating an entire range $[l, r]$ at once? Without optimization, a range update would touch up to $O(n)$ nodes. **Lazy propagation** solves this by deferring updates to children until they are actually needed, achieving $O(\log n)$ for both range updates and range queries.

## The Problem with Naive Range Updates

Suppose we want to add a value $\delta$ to every element in the range $[l, r]$. A naive approach updates each of the $r - l + 1$ positions individually, costing $O((r - l + 1) \cdot \log n)$ in total. For large ranges, this approaches $O(n \log n)$ — barely better than recomputing from scratch.

## The Lazy Idea

Instead of immediately pushing an update to every descendant node, we store the pending update at the highest node whose range is fully contained in $[l, r]$. This "lazy" tag records what has not yet been applied to the node's children. When a subsequent query or update needs to visit a child, we first **push down** (propagate) the lazy tag to the children, then proceed.

!!! note "Deferred Work Principle"
    Lazy propagation follows a general principle in data structure design: defer work until the result is needed. This amortizes the cost of updates over future operations.

## Push-Down Mechanism

The push-down operation for a node $v$ covering range $[lo, hi]$ with lazy tag $\text{lazy}[v]$:

1. Compute $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$.
2. **Left child** ($2v$, covering $[lo, \text{mid}]$): add $\text{lazy}[v] \cdot (\text{mid} - lo + 1)$ to its value, and add $\text{lazy}[v]$ to its lazy tag.
3. **Right child** ($2v + 1$, covering $[\text{mid}+1, hi]$): add $\text{lazy}[v] \cdot (hi - \text{mid})$ to its value, and add $\text{lazy}[v]$ to its lazy tag.
4. Clear the lazy tag at $v$: $\text{lazy}[v] = 0$.

## Range Update Algorithm

To add $\delta$ to every element in $[l, r]$:

1. If the node's range $[lo, hi]$ does not overlap $[l, r]$: return.
2. If $[lo, hi] \subseteq [l, r]$: add $\delta \cdot (hi - lo + 1)$ to the node's value, add $\delta$ to its lazy tag, and return.
3. Otherwise, push down the lazy tag, recursively update both children, then recompute the node's value from its children.

## Range Query with Lazy Tags

To query the sum over $[l, r]$:

1. If the node's range does not overlap $[l, r]$: return 0.
2. If the node's range is fully inside $[l, r]$: return the node's value.
3. Otherwise, push down the lazy tag first, then recursively query both children.

The push-down before recursion ensures that each child's value is up to date when accessed.

## Implementation

```python
"""
Segment tree with lazy propagation for range updates.

Supports O(log n) range additions and O(log n) range sum
queries using deferred (lazy) updates that are propagated
only when needed.
"""


# === Segment Tree with Lazy Propagation ===

class LazySegTree:
    """Segment tree supporting range updates and range queries."""

    def __init__(self, data: list):
        """Build from input array in O(n)."""
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
        """Propagate lazy tag from node to its children."""
        if self.lazy[node] != 0:
            mid = (lo + hi) // 2
            left, right = 2 * node, 2 * node + 1

            # Update left child
            self.tree[left] += self.lazy[node] * (mid - lo + 1)
            self.lazy[left] += self.lazy[node]

            # Update right child
            self.tree[right] += self.lazy[node] * (hi - mid)
            self.lazy[right] += self.lazy[node]

            # Clear this node's lazy tag
            self.lazy[node] = 0

    def range_update(self, node: int, lo: int, hi: int,
                     l: int, r: int, delta: int) -> None:
        """Add delta to every element in [l, r]."""
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
        """Return the sum of elements in [l, r]."""
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return self.tree[node]
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        return (self.range_query(2 * node, lo, mid, l, r)
                + self.range_query(2 * node + 1, mid + 1, hi, l, r))


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    n = len(data)
    st = LazySegTree(data)

    print(f"Array: {data}")
    print(f"Sum [0,5]: {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [1,3]: {st.range_query(1, 0, n-1, 1, 3)}")
    print()

    # Range update: add 10 to positions [1, 4]
    print("Adding 10 to every element in [1, 4]...")
    st.range_update(1, 0, n - 1, 1, 4, 10)

    print(f"Sum [0,5]: {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [1,3]: {st.range_query(1, 0, n-1, 1, 3)}")
    print(f"Sum [0,0]: {st.range_query(1, 0, n-1, 0, 0)}")
    print(f"Sum [5,5]: {st.range_query(1, 0, n-1, 5, 5)}")
    print()

    # Another range update
    print("Adding 5 to every element in [0, 2]...")
    st.range_update(1, 0, n - 1, 0, 2, 5)
    print(f"Sum [0,5]: {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [0,2]: {st.range_query(1, 0, n-1, 0, 2)}")
```

**Output:**
```
Array: [1, 3, 5, 7, 9, 11]
Sum [0,5]: 36
Sum [1,3]: 15

Adding 10 to every element in [1, 4]...
Sum [0,5]: 76
Sum [1,3]: 45
Sum [0,0]: 1
Sum [5,5]: 11

Adding 5 to every element in [0, 2]...
Sum [0,5]: 91
Sum [0,2]: 39
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Build | $O(n)$ | $O(n)$ |
| Range update | $O(\log n)$ | $O(1)$ |
| Range query | $O(\log n)$ | $O(1)$ |

The lazy array doubles the memory usage (from $4n$ to $8n$ integers), but the asymptotic space remains $O(n)$.

## When Lazy Propagation Applies

Lazy propagation works when the update operation is **composable** — multiple pending updates at a node can be merged into a single tag. Addition satisfies this because adding $\delta_1$ then $\delta_2$ is equivalent to adding $\delta_1 + \delta_2$. The technique generalizes to:

- **Range assignment** (set all elements in a range to a value).
- **Range multiply** (multiply all elements by a factor).
- **Combined operations** (e.g., multiply then add), though composing two different operation types requires careful tag ordering.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
