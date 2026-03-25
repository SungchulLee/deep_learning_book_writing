# Construction

A **segment tree** is a binary tree that stores aggregate information about contiguous sub-ranges of an array. Before answering any queries or processing updates, the tree must be built from the input array. This page explains the segment tree's structure, derives the array size required to store it, and presents the $O(n)$ recursive build algorithm.

## Structure of a Segment Tree

Given an array $a[0..n-1]$, the segment tree is a complete binary tree where:

- Each **leaf** corresponds to a single element $a[i]$.
- Each **internal node** stores the aggregate (e.g., sum, minimum, maximum) of the range covered by its two children.
- The **root** stores the aggregate of the entire array $a[0..n-1]$.

A node responsible for the range $[lo, hi]$ has:

- Left child covering $[lo, \text{mid}]$ where $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$.
- Right child covering $[\text{mid}+1, hi]$.

## Array Representation

Segment trees are stored in a flat array using 1-based indexing (similar to binary heaps):

- Root is at index 1.
- Left child of node $k$ is at $2k$.
- Right child of node $k$ is at $2k + 1$.

!!! note "Why Allocate 4n Nodes"
    A segment tree for an array of size $n$ has at most $2n - 1$ nodes when $n$ is a power of 2. For arbitrary $n$, the tree must accommodate the next power of 2, requiring up to $4n$ entries in the worst case. Allocating `4 * n` is a safe upper bound that avoids index-out-of-bounds errors.

## Recursive Build Algorithm

The build procedure constructs the tree bottom-up in a single pass:

1. **Base case.** If $lo = hi$, the node is a leaf: store $a[lo]$.
2. **Recursive case.** Compute $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$. Recursively build the left child (range $[lo, \text{mid}]$) and right child (range $[\text{mid}+1, hi]$). Set the node's value to the aggregate of its children.

Each of the $O(n)$ nodes is visited exactly once, so the build runs in $O(n)$ time.

## Implementation

```python
"""
Segment tree construction.

Demonstrates the O(n) recursive build algorithm that transforms
an input array into a segment tree stored in a flat array.
Both a sum-based and a min-based segment tree are shown.
"""


# === Segment Tree (Sum) ===

class SegmentTree:
    """Segment tree for range sum queries, built in O(n)."""

    def __init__(self, data: list):
        """Build a segment tree from the input array."""
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(data, 1, 0, self.n - 1)

    def _build(self, data: list, node: int, lo: int, hi: int) -> None:
        """Recursively build the tree.

        Args:
            data: The original array.
            node: Current node index in the tree array.
            lo, hi: Range of the original array this node covers.
        """
        if lo == hi:
            # Leaf node — store the single element
            self.tree[node] = data[lo]
            return

        mid = (lo + hi) // 2
        self._build(data, 2 * node, lo, mid)         # build left child
        self._build(data, 2 * node + 1, mid + 1, hi)  # build right child
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node: int, lo: int, hi: int, l: int, r: int) -> int:
        """Return the sum of elements in [l, r]."""
        if r < lo or hi < l:
            return 0  # identity for sum
        if l <= lo and hi <= r:
            return self.tree[node]
        mid = (lo + hi) // 2
        return (self.query(2 * node, lo, mid, l, r)
                + self.query(2 * node + 1, mid + 1, hi, l, r))

    def print_tree(self, node: int, lo: int, hi: int, depth: int = 0) -> None:
        """Print the tree structure for visualization."""
        indent = "  " * depth
        print(f"{indent}Node {node}: [{lo},{hi}] = {self.tree[node]}")
        if lo < hi:
            mid = (lo + hi) // 2
            self.print_tree(2 * node, lo, mid, depth + 1)
            self.print_tree(2 * node + 1, mid + 1, hi, depth + 1)


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    print(f"Input array: {data}")
    print(f"Array size n = {len(data)}")
    print(f"Tree array size = {4 * len(data)}")
    print()

    st = SegmentTree(data)

    print("Segment tree structure:")
    st.print_tree(1, 0, len(data) - 1)
    print()

    # Verify range sums
    queries = [(0, 2), (1, 4), (0, 5), (3, 5), (2, 2)]
    for l, r in queries:
        result = st.query(1, 0, len(data) - 1, l, r)
        expected = sum(data[l:r + 1])
        print(f"Sum [{l},{r}] = {result}  (expected {expected})")
```

**Output:**
```
Input array: [1, 3, 5, 7, 9, 11]
Array size n = 6
Tree array size = 24

Segment tree structure:
Node 1: [0,5] = 36
  Node 2: [0,2] = 9
    Node 4: [0,1] = 4
      Node 8: [0,0] = 1
      Node 9: [1,1] = 3
    Node 5: [2,2] = 5
  Node 3: [3,5] = 27
    Node 6: [3,4] = 16
      Node 12: [3,3] = 7
      Node 13: [4,4] = 9
    Node 7: [5,5] = 11

Sum [0,2] = 9  (expected 9)
Sum [1,4] = 24  (expected 24)
Sum [0,5] = 36  (expected 36)
Sum [3,5] = 27  (expected 27)
Sum [2,2] = 5  (expected 5)
```

## Build Complexity

| Aspect | Complexity |
|--------|-----------|
| Time | $O(n)$ |
| Space | $O(n)$ (the tree array of size $4n$) |

The $O(n)$ time follows from the fact that the tree has at most $4n$ nodes and each is initialized exactly once. This is faster than building by $n$ individual point updates, which would take $O(n \log n)$.

## Generalization to Other Aggregates

The construction works for any associative binary operation. Common choices include:

| Aggregate | Identity element | Merge operation |
|-----------|-----------------|-----------------|
| Sum | 0 | $a + b$ |
| Minimum | $+\infty$ | $\min(a, b)$ |
| Maximum | $-\infty$ | $\max(a, b)$ |
| GCD | 0 | $\gcd(a, b)$ |
| XOR | 0 | $a \oplus b$ |

The only change required is the merge function and the identity element used when a query falls outside a node's range.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
