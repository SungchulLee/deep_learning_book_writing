# Array Representation

A binary heap gains much of its practical efficiency from a remarkably simple insight: a complete binary tree can be stored in a flat array with no pointers at all. Because every level is fully filled except possibly the last (which is filled left to right), the position of each node's parent and children can be computed with elementary arithmetic. This pointer-free layout gives excellent cache performance and zero memory overhead for child/parent links.

## Level-Order Storage

A complete binary tree is stored in an array by placing nodes in **level order** -- the root first, then all nodes at depth 1 from left to right, then depth 2, and so on. The result is a compact array with no gaps.

!!! example "Tree-to-Array Mapping"
    Consider a max-heap with 10 elements:

    ```
    Tree view:                    Array view (0-indexed):

              16                  Index: 0  1  2  3  4  5  6  7  8  9
            /    \                Value: 16 14 10  8  7  9  3  2  4  1
          14      10
         /  \    /  \
        8    7  9    3
       / \  /
      2  4 1
    ```

    Node 16 is at index 0, its children 14 and 10 are at indices 1 and 2, and so on.

## Index Formulas

The parent-child relationships in the array follow directly from the level-order layout. Two conventions are common.

### 0-Indexed (used by Python heapq)

For a node at index $i$:

$$
\text{parent}(i) = \left\lfloor \frac{i - 1}{2} \right\rfloor
$$

$$
\text{left}(i) = 2i + 1
$$

$$
\text{right}(i) = 2i + 2
$$

The root is at index 0. A node at index $i$ is a leaf if $2i + 1 \ge n$, where $n$ is the total number of elements.

### 1-Indexed (used in CLRS)

For a node at index $i$:

$$
\text{parent}(i) = \left\lfloor \frac{i}{2} \right\rfloor
$$

$$
\text{left}(i) = 2i
$$

$$
\text{right}(i) = 2i + 1
$$

The root is at index 1, and index 0 is unused. The 1-indexed formulas are slightly simpler because multiplication and division by 2 correspond to left and right bit shifts.

??? tip "Bit-Shift Optimization"
    In the 1-indexed scheme, the parent, left-child, and right-child operations reduce to single bitwise instructions:

    - `parent(i) = i >> 1`
    - `left(i) = i << 1`
    - `right(i) = (i << 1) | 1`

    These are constant-time operations that modern CPUs execute in a single cycle.

## Why Arrays Work for Complete Binary Trees

The array representation works only because a complete binary tree has no "holes" in its level-order traversal. An arbitrary binary tree might waste vast amounts of space: a skewed tree with $n$ nodes at depth $n-1$ would require an array of size $2^n - 1$, with most entries empty.

For a complete binary tree, the array is always compact:

| Property | Value |
|----------|-------|
| Nodes on level $k$ | $2^k$ (except possibly the last level) |
| Total nodes in a complete tree of height $h$ | Between $2^h$ and $2^{h+1} - 1$ |
| Array size needed | Exactly $n$ (no wasted slots) |
| Memory overhead for pointers | Zero |

## Navigating the Heap Array

The following implementation demonstrates parent-child navigation using the 0-indexed convention.

```python
"""
Array representation of a binary heap.

Demonstrates how parent-child relationships in a complete binary
tree map to simple index arithmetic in a flat array.
"""


# === Index Navigation (0-indexed) ===

def parent(i):
    """Return the index of the parent of node i."""
    return (i - 1) // 2


def left_child(i):
    """Return the index of the left child of node i."""
    return 2 * i + 1


def right_child(i):
    """Return the index of the right child of node i."""
    return 2 * i + 2


def is_leaf(i, n):
    """Check if node i is a leaf in a heap of size n."""
    return left_child(i) >= n


# === Tree Visualization ===

def print_heap_tree(arr):
    """Print the array as a tree structure showing parent-child relationships."""
    n = len(arr)
    if n == 0:
        print("Empty heap")
        return

    print(f"Array: {arr}")
    print(f"Size:  {n}\n")

    for i in range(n):
        l = left_child(i)
        r = right_child(i)
        children = []
        if l < n:
            children.append(f"left={arr[l]} (idx {l})")
        if r < n:
            children.append(f"right={arr[r]} (idx {r})")

        parent_info = ""
        if i > 0:
            p = parent(i)
            parent_info = f"  parent={arr[p]} (idx {p})"

        child_info = ", ".join(children) if children else "leaf"
        print(f"  idx {i}: value={arr[i]}{parent_info}  -> {child_info}")


# === Demonstration ===

if __name__ == "__main__":
    heap = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
    print_heap_tree(heap)

    print("\n--- Index formula verification ---")
    for i in range(len(heap)):
        if i > 0:
            p = parent(i)
            assert heap[p] >= heap[i], f"Max-heap violated at index {i}"
    print("All parent-child relationships satisfy the max-heap property.")
```

**Output:**
```
Array: [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Size:  10

  idx 0: value=16  -> left=14 (idx 1), right=10 (idx 2)
  idx 1: value=14  parent=16 (idx 0)  -> left=8 (idx 3), right=7 (idx 4)
  idx 2: value=10  parent=16 (idx 0)  -> left=9 (idx 5), right=3 (idx 6)
  idx 3: value=8  parent=14 (idx 1)  -> left=2 (idx 7), right=4 (idx 8)
  idx 4: value=7  parent=14 (idx 1)  -> left=1 (idx 9)
  idx 5: value=9  parent=10 (idx 2)  -> leaf
  idx 6: value=3  parent=10 (idx 2)  -> leaf
  idx 7: value=2  parent=8 (idx 3)  -> leaf
  idx 8: value=4  parent=8 (idx 3)  -> leaf
  idx 9: value=1  parent=7 (idx 4)  -> leaf

--- Index formula verification ---
All parent-child relationships satisfy the max-heap property.
```

## Cache Efficiency

Storing a tree as a contiguous array provides significant performance benefits on modern hardware. When the processor accesses an element at index $i$, the cache line typically loads neighboring elements as well. Since a node's children at indices $2i+1$ and $2i+2$ are nearby in memory, traversing from parent to child frequently hits the cache. By contrast, a pointer-based tree scatters nodes throughout memory, causing frequent cache misses during traversal.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6: Heapsort. MIT Press.
