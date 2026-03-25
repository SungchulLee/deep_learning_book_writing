# Leftist Heaps

Binary heaps are stored in arrays and support efficient insert and extract-min, but merging two binary heaps requires $O(n)$ time. **Leftist heaps** are pointer-based heap-ordered binary trees that support merge in $O(\log n)$ time. The key idea is to maintain a structural bias: the right spine of the tree is always short. Since merge operations follow the right spine, this guarantees logarithmic merge cost. All other operations (insert, extract-min, delete) reduce to merge, making leftist heaps one of the simplest mergeable priority queue implementations.

## The s-value (Null Path Length)

The leftist property is defined in terms of the **s-value** (also called the **null path length** or **rank**) of each node.

**Definition**: The s-value of a node $x$, denoted $s(x)$, is the length of the shortest path from $x$ to a null (missing) descendant:

$$
s(x) = \begin{cases} 0 & \text{if } x = \text{null} \\ 1 + \min(s(\text{left}(x)),\; s(\text{right}(x))) & \text{otherwise} \end{cases}
$$

!!! example "s-values in a Tree"
    ```
           4 (s=2)
          / \
      8 (s=1) 6 (s=1)
       /     / \
    10(s=1) 7(s=1) 9(s=1)
    ```
    Node 4 has s-value 2 because the shortest path to a null descendant goes right-right (two edges).

## The Leftist Property

A heap-ordered binary tree is **leftist** if for every internal node $x$:

$$
s(\text{left}(x)) \ge s(\text{right}(x))
$$

This property biases the tree so that the **right spine** (the path from the root following only right children) is always the shortest path from root to null. Consequently, the right spine has length at most $\lfloor \log_2(n+1) \rfloor$.

!!! tip "Right Spine Length Bound"
    If the right spine has length $r$, the subtree rooted at its top contains at least $2^r - 1$ nodes (since s-values increase by at least 1 at each level). Therefore $n \ge 2^r - 1$, which gives $r \le \lfloor \log_2(n+1) \rfloor$. This bound is what makes merge efficient.

## Merge Operation

Merge is the fundamental operation. Given two leftist heaps $H_1$ and $H_2$:

1. Compare the roots. The smaller root becomes the root of the merged heap.
2. Recursively merge the larger root's heap with the winner's **right** subtree.
3. After the recursive call, if the leftist property is violated (right child has larger s-value than left child), swap the left and right children.
4. Update the s-value: $s(\text{root}) = s(\text{right}) + 1$.

Since each recursive call descends along the right spine of one of the two heaps, the total number of recursive calls is at most the sum of the two right spine lengths, giving:

$$
T_{\text{merge}} = O(\log n_1 + \log n_2) = O(\log n)
$$

where $n = n_1 + n_2$.

## Other Operations

All operations reduce to merge:

| Operation | Reduction | Time |
|-----------|----------|:----:|
| Insert | Create a single-node heap, merge | $O(\log n)$ |
| Find-min | Return root key | $O(1)$ |
| Extract-min | Merge root's left and right children | $O(\log n)$ |
| Delete-min | Same as extract-min | $O(\log n)$ |

## Implementation

```python
"""
Leftist heap implementation.

A leftist heap is a heap-ordered binary tree where the s-value
(null path length) of the left child is always >= the right child.
All operations reduce to merge, which runs in O(log n).
"""


# === Leftist Heap Node ===

class LeftistNode:
    """A node in a leftist heap.

    Attributes:
        key: the priority value
        s: the s-value (null path length)
        left: left child
        right: right child
    """

    def __init__(self, key):
        self.key = key
        self.s = 1  # a single node has s-value 1
        self.left = None
        self.right = None

    def __repr__(self):
        return f"LeftistNode(key={self.key}, s={self.s})"


# === Leftist Heap ===

class LeftistHeap:
    """A min-leftist-heap where all operations reduce to merge."""

    def __init__(self):
        self.root = None
        self.size = 0

    @staticmethod
    def _s_value(node):
        """Return the s-value of a node (0 for null)."""
        return 0 if node is None else node.s

    @staticmethod
    def _merge_nodes(h1, h2):
        """Merge two leftist heap subtrees. Returns the new root."""
        if h1 is None:
            return h2
        if h2 is None:
            return h1

        # Ensure h1 has the smaller root
        if h1.key > h2.key:
            h1, h2 = h2, h1

        # Recursively merge h2 with h1's right subtree
        h1.right = LeftistHeap._merge_nodes(h1.right, h2)

        # Restore leftist property: left s-value >= right s-value
        if LeftistHeap._s_value(h1.left) < LeftistHeap._s_value(h1.right):
            h1.left, h1.right = h1.right, h1.left

        # Update s-value
        h1.s = LeftistHeap._s_value(h1.right) + 1
        return h1

    def merge(self, other):
        """Merge another leftist heap into this one. O(log n)."""
        self.root = self._merge_nodes(self.root, other.root)
        self.size += other.size

    def insert(self, key):
        """Insert a key by creating a single-node heap and merging. O(log n)."""
        new_node = LeftistNode(key)
        self.root = self._merge_nodes(self.root, new_node)
        self.size += 1

    def find_min(self):
        """Return the minimum key. O(1)."""
        if self.root is None:
            raise IndexError("find_min from empty heap")
        return self.root.key

    def extract_min(self):
        """Remove and return the minimum key. O(log n)."""
        if self.root is None:
            raise IndexError("extract_min from empty heap")
        min_key = self.root.key
        self.root = self._merge_nodes(self.root.left, self.root.right)
        self.size -= 1
        return min_key

    def is_empty(self):
        """Check if the heap is empty."""
        return self.root is None

    def _verify_leftist(self, node=None, check_root=True):
        """Verify the leftist property holds for all nodes."""
        if check_root:
            node = self.root
        if node is None:
            return True
        left_s = self._s_value(node.left)
        right_s = self._s_value(node.right)
        assert left_s >= right_s, \
            f"Leftist violated at key={node.key}: left_s={left_s}, right_s={right_s}"
        assert node.s == right_s + 1, \
            f"s-value wrong at key={node.key}"
        return (self._verify_leftist(node.left, False) and
                self._verify_leftist(node.right, False))


# === Demonstration ===

if __name__ == "__main__":
    h = LeftistHeap()
    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]

    print("Inserting values:")
    for v in values:
        h.insert(v)
        print(f"  Inserted {v}, min = {h.find_min()}, size = {h.size}")

    # Verify leftist property
    h._verify_leftist()
    print("Leftist property verified.")

    print(f"\nExtracting in order:")
    extracted = []
    while not h.is_empty():
        val = h.extract_min()
        extracted.append(val)
        print(f"  Extracted {val}")

    print(f"\nExtracted sequence: {extracted}")
    assert extracted == sorted(values), "Extraction order incorrect!"
    print("Correctness verified.")

    # Demonstrate merge
    print("\n--- Merge Demo ---")
    h1 = LeftistHeap()
    for v in [5, 3, 7]:
        h1.insert(v)

    h2 = LeftistHeap()
    for v in [2, 8, 1]:
        h2.insert(v)

    print(f"H1 min: {h1.find_min()}, H2 min: {h2.find_min()}")
    h1.merge(h2)
    print(f"After merge, min: {h1.find_min()}, size: {h1.size}")

    h1._verify_leftist()
    print("Leftist property verified after merge.")

    merged = []
    while not h1.is_empty():
        merged.append(h1.extract_min())
    print(f"Merged extraction: {merged}")
```

**Output:**
```
Inserting values:
  Inserted 7, min = 7, size = 1
  Inserted 3, min = 3, size = 2
  Inserted 8, min = 3, size = 3
  Inserted 1, min = 1, size = 4
  Inserted 5, min = 1, size = 5
  Inserted 2, min = 1, size = 6
  Inserted 9, min = 1, size = 7
  Inserted 4, min = 1, size = 8
  Inserted 6, min = 1, size = 9
Leftist property verified.

Extracting in order:
  Extracted 1
  Extracted 2
  Extracted 3
  Extracted 4
  Extracted 5
  Extracted 6
  Extracted 7
  Extracted 8
  Extracted 9

Extracted sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9]
Correctness verified.

--- Merge Demo ---
H1 min: 3, H2 min: 1
After merge, min: 1, size: 6
Leftist property verified after merge.
Merged extraction: [1, 2, 3, 5, 7, 8]
```

## Complexity Summary

| Operation | Leftist Heap | Binary Heap |
|-----------|:------------:|:-----------:|
| Insert | $O(\log n)$ | $O(\log n)$ |
| Find-min | $O(1)$ | $O(1)$ |
| Extract-min | $O(\log n)$ | $O(\log n)$ |
| Merge | $O(\log n)$ | $O(n)$ |

Leftist heaps match binary heaps on all standard operations and add $O(\log n)$ merge. The tradeoff is pointer overhead (each node stores two child pointers and an s-value) versus the array-based simplicity and cache-friendliness of binary heaps.

## Reference

- Crane, C. A. "Linear lists and priority queues as balanced binary trees." Ph.D. thesis, Stanford University, 1972.
- Tarjan, R. E. *Data Structures and Network Algorithms*. SIAM, 1983.
