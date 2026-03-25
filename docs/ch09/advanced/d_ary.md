# d-ary Heaps

A binary heap gives every node exactly two children. A natural question arises: what happens if we allow each node to have $d$ children instead? A **$d$-ary heap** generalizes the binary heap by letting each internal node have up to $d$ children, where $d \ge 2$. The binary heap is the special case $d = 2$.

The motivation for this generalization is a performance tradeoff. Increasing $d$ reduces the tree height from $\log_2 n$ to $\log_d n$, which speeds up operations that traverse root-to-leaf paths (like decrease-key). However, finding the minimum child among $d$ children now takes $O(d)$ instead of $O(1)$. Choosing the right $d$ can optimize performance for specific workloads -- for instance, Dijkstra's algorithm benefits from $d$-ary heaps when the graph is dense.

## Structure and Index Formulas

Like a binary heap, a $d$-ary heap is stored as an array using level-order indexing (0-based). For a node at index $i$:

**Children** of node $i$: indices $di + 1, \; di + 2, \; \ldots, \; di + d$ (those that are $\le n-1$).

**Parent** of node $i$ (for $i > 0$):

$$
\text{parent}(i) = \left\lfloor \frac{i - 1}{d} \right\rfloor
$$

The tree has height:

$$
h = \lfloor \log_d n \rfloor = \left\lfloor \frac{\ln n}{\ln d} \right\rfloor
$$

!!! example "A 3-ary Min-Heap"
    ```
    Array: [2, 5, 7, 3, 8, 9, 10, 6, 4, 11]

    Tree (d=3):
                    2
                /   |   \
              5     7     3
            / | \   |   / | \
           8  9 10  6  4  11
    ```
    Each node has at most 3 children. The last level may be partially filled.

## Operations

### Sift-Up (for Insert and Decrease-Key)

Sift-up compares a node with its single parent and swaps if the heap property is violated. Since the tree height is $\lfloor \log_d n \rfloor$, sift-up traverses at most this many levels with one comparison per level:

$$
T_{\text{sift-up}} = O(\log_d n)
$$

### Sift-Down (for Extract-Min and Build-Heap)

Sift-down must find the minimum among a node's $d$ children before swapping. Each level requires $d - 1$ comparisons to find the minimum child, and sift-down traverses at most $\lfloor \log_d n \rfloor$ levels:

$$
T_{\text{sift-down}} = O(d \log_d n)
$$

### Operation Complexities

| Operation | Complexity |
|-----------|:----------:|
| Insert | $O(\log_d n)$ |
| Find-min | $O(1)$ |
| Extract-min | $O(d \log_d n)$ |
| Decrease-key | $O(\log_d n)$ |
| Build-heap | $O(n)$ |

The build-heap complexity remains $O(n)$ by the same bottom-up argument as for binary heaps. The sum telescopes regardless of $d$.

## The Tradeoff

The key insight is that increasing $d$ creates opposing effects:

- **Decrease-key becomes faster**: $O(\log_d n) = O(\log n / \log d)$, which decreases as $d$ grows.
- **Extract-min becomes slower**: $O(d \log_d n) = O(d \log n / \log d)$, which increases with $d$ when $d$ grows beyond $\log n$.

For algorithms like Dijkstra's shortest paths, which perform $|V|$ extract-min and $|E|$ decrease-key operations, the total heap cost is:

$$
T = O\left(|V| \cdot d \cdot \frac{\log |V|}{\log d} + |E| \cdot \frac{\log |V|}{\log d}\right)
$$

Setting $d = \max(2, \lceil |E|/|V| \rceil)$ balances the two terms. For dense graphs where $|E| = \Theta(|V|^2)$, choosing $d = |V|$ gives:

$$
T = O(|V|^2)
$$

which matches the performance of an unordered-array priority queue and is optimal for dense Dijkstra.

## Implementation

```python
"""
d-ary heap implementation.

A d-ary heap generalizes the binary heap by giving each node
up to d children. This trades off extract-min cost against
decrease-key cost, controlled by the branching factor d.
"""


# === d-ary Heap ===

class DAryHeap:
    """A min-d-ary-heap stored as an array."""

    def __init__(self, d=2):
        """Initialize with branching factor d >= 2."""
        if d < 2:
            raise ValueError("Branching factor d must be >= 2")
        self.d = d
        self.heap = []

    def _parent(self, i):
        """Return the parent index of node i."""
        return (i - 1) // self.d

    def _children(self, i):
        """Return the range of children indices for node i."""
        start = self.d * i + 1
        end = min(start + self.d, len(self.heap))
        return range(start, end)

    def _sift_up(self, i):
        """Move node i up until the heap property is restored."""
        while i > 0:
            parent = self._parent(i)
            if self.heap[i] < self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        """Move node i down until the heap property is restored."""
        n = len(self.heap)
        while True:
            min_idx = i
            for c in self._children(i):
                if c < n and self.heap[c] < self.heap[min_idx]:
                    min_idx = c
            if min_idx == i:
                break
            self.heap[i], self.heap[min_idx] = self.heap[min_idx], self.heap[i]
            i = min_idx

    def insert(self, key):
        """Insert a key into the heap. O(log_d n)."""
        self.heap.append(key)
        self._sift_up(len(self.heap) - 1)

    def find_min(self):
        """Return the minimum key. O(1)."""
        if not self.heap:
            raise IndexError("find_min from empty heap")
        return self.heap[0]

    def extract_min(self):
        """Remove and return the minimum key. O(d * log_d n)."""
        if not self.heap:
            raise IndexError("extract_min from empty heap")
        min_val = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        return min_val

    def decrease_key(self, i, new_key):
        """Decrease the key at index i to new_key. O(log_d n)."""
        if new_key > self.heap[i]:
            raise ValueError("New key is greater than current key")
        self.heap[i] = new_key
        self._sift_up(i)

    @classmethod
    def build_heap(cls, data, d=2):
        """Build a d-ary heap from a list in O(n) time."""
        h = cls(d=d)
        h.heap = list(data)
        # Sift down from the last parent to the root
        n = len(h.heap)
        for i in range((n - 2) // d, -1, -1):
            h._sift_down(i)
        return h

    def is_empty(self):
        """Check if the heap is empty."""
        return len(self.heap) == 0


# === Demonstration ===

if __name__ == "__main__":
    import math

    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]

    for d in [2, 3, 4]:
        h = DAryHeap(d=d)
        for v in values:
            h.insert(v)

        extracted = []
        while not h.is_empty():
            extracted.append(h.extract_min())

        height = math.floor(math.log(len(values)) / math.log(d))
        print(f"d={d}: height={height}, extracted={extracted}")

    # Build-heap demonstration
    print("\nBuild-heap (d=3):")
    h = DAryHeap.build_heap([7, 3, 8, 1, 5, 2, 9, 4, 6], d=3)
    extracted = []
    while not h.is_empty():
        extracted.append(h.extract_min())
    print(f"  Extracted: {extracted}")
```

**Output:**
```
d=2: height=3, extracted=[1, 2, 3, 4, 5, 6, 7, 8, 9]
d=3: height=1, extracted=[1, 2, 3, 4, 5, 6, 7, 8, 9]
d=4: height=1, extracted=[1, 2, 3, 4, 5, 6, 7, 8, 9]

Build-heap (d=3):
  Extracted: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Comparison with Binary Heap

| Aspect | Binary Heap ($d=2$) | $d$-ary Heap |
|--------|:-------------------:|:------------:|
| Height | $\lfloor \log_2 n \rfloor$ | $\lfloor \log_d n \rfloor$ |
| Insert | $O(\log_2 n)$ | $O(\log_d n)$ |
| Extract-min | $O(\log_2 n)$ | $O(d \log_d n)$ |
| Decrease-key | $O(\log_2 n)$ | $O(\log_d n)$ |
| Cache behavior | Good | Better for large $d$ (wider, shallower) |

!!! tip "Practical Guidance"
    In practice, $d = 4$ often outperforms $d = 2$ due to better cache utilization -- the shallower tree means fewer cache misses during sift-down. For Dijkstra on dense graphs, $d = |E|/|V|$ is theoretically optimal.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Problem 6-2: d-ary Heaps. MIT Press.
- Johnson, D. B. "Efficient algorithms for shortest paths in sparse networks." *Journal of the ACM*, 24(1):1--13, 1977.
