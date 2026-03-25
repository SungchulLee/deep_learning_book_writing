# Complexity

Every core BST operation -- [search](search.md), [insertion](insertion.md), [deletion](deletion.md), [min/max](min_max.md), and [successor/predecessor](successor.md) -- follows a root-to-leaf path or a root-to-node path. This means the running time of each operation is bounded by the height $h$ of the tree. The critical question is: how does $h$ relate to the number of nodes $n$?

## Operations are O(h)

Each BST operation performs a constant amount of work at each level it visits, descending one level per iteration or recursive call. Since the longest possible path has $h$ edges, every operation runs in $O(h)$ time.

| Operation | Time complexity |
|---|---|
| Search | $O(h)$ |
| Insertion | $O(h)$ |
| Deletion | $O(h)$ |
| Minimum / Maximum | $O(h)$ |
| Successor / Predecessor | $O(h)$ |
| Inorder traversal | $\Theta(n)$ |

The traversal is the exception: it visits every node exactly once, so it takes $\Theta(n)$ regardless of the tree shape.

## Best Case: Balanced Tree

When the tree is balanced (every node's left and right subtrees differ in size by at most a constant factor), the height is:

$$
h = \Theta(\log n)
$$

In a perfectly balanced BST, each comparison eliminates roughly half the remaining nodes, just like binary search on a sorted array. All operations run in $O(\log n)$ time.

**Example**: A balanced BST with $n = 15$ nodes has height $h = 3$, so search visits at most 4 nodes.

## Worst Case: Degenerate Tree

If keys are inserted in sorted (or reverse-sorted) order, each new node becomes a right (or left) child of the previous node, producing a **degenerate** tree that looks like a linked list:

```
Insert 1, 2, 3, 4, 5:

1
 \
  2
   \
    3
     \
      4
       \
        5
```

Here $h = n - 1$, and all operations degrade to $O(n)$.

$$
h = n - 1 \quad \Rightarrow \quad \text{all operations are } O(n)
$$

## Average Case: Random Insertions

When $n$ distinct keys are inserted in random order (every permutation equally likely), the expected height of the resulting BST is:

$$
E[h] = O(\log n)
$$

More precisely, the expected height of a randomly built BST on $n$ keys is at most $3 \ln n \approx 4.33 \log_2 n$. This result, proven in CLRS using the analysis of [randomly built BSTs](random.md), shows that the degenerate worst case is unlikely under random input.

!!! note "Expected vs Guaranteed"
    The $O(\log n)$ expected height of a randomly built BST assumes that the insertion order is a uniformly random permutation. Real-world data is often not random -- sorted input, nearly-sorted input, and adversarial input can all produce degenerate trees. Balanced BST variants (AVL trees, red-black trees, B-trees) provide $O(\log n)$ **worst-case** height regardless of insertion order.

## Summary Table

| Tree shape | Height $h$ | Operation time | When it occurs |
|---|---|---|---|
| Perfectly balanced | $\lfloor \log_2 n \rfloor$ | $O(\log n)$ | Careful construction |
| Random | $O(\log n)$ expected | $O(\log n)$ expected | Random insertion order |
| Degenerate (linear) | $n - 1$ | $O(n)$ | Sorted insertion order |

## Space Complexity

A BST with $n$ nodes uses $\Theta(n)$ space regardless of the tree shape. The shape affects time complexity but not space.

Recursive operations use $O(h)$ stack space. For a balanced tree this is $O(\log n)$; for a degenerate tree this is $O(n)$. Iterative implementations of search and insertion use $O(1)$ auxiliary space.

```python
"""
BST complexity demonstration.

Builds BSTs from sorted and random insertion orders to illustrate
how tree shape affects height and operation time.
"""

import random
import time


# === Node definition ===

class Node:
    """A node in a binary search tree."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


# === BST operations ===

def insert(root, key):
    """Insert a key into the BST (iterative to avoid stack overflow)."""
    new_node = Node(key)
    if root is None:
        return new_node
    current = root
    while True:
        if key <= current.key:
            if current.left is None:
                current.left = new_node
                return root
            current = current.left
        else:
            if current.right is None:
                current.right = new_node
                return root
            current = current.right


def height(node):
    """Return the height of the tree (iterative BFS-based)."""
    if node is None:
        return -1
    from collections import deque
    queue = deque([(node, 0)])
    max_depth = 0
    while queue:
        current, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        if current.left:
            queue.append((current.left, depth + 1))
        if current.right:
            queue.append((current.right, depth + 1))
    return max_depth


def search(root, key):
    """Search for a key, returning the number of comparisons."""
    comparisons = 0
    current = root
    while current is not None:
        comparisons += 1
        if key == current.key:
            return comparisons
        elif key < current.key:
            current = current.left
        else:
            current = current.right
    return comparisons


# === Main ===

if __name__ == "__main__":
    n = 1000

    # Sorted insertion -> degenerate tree
    sorted_root = None
    for k in range(1, n + 1):
        sorted_root = insert(sorted_root, k)
    sorted_height = height(sorted_root)

    # Random insertion -> balanced tree (expected)
    keys = list(range(1, n + 1))
    random.seed(42)
    random.shuffle(keys)
    random_root = None
    for k in keys:
        random_root = insert(random_root, k)
    random_height = height(random_root)

    print(f"n = {n}")
    print(f"  Sorted insertion height:  {sorted_height} (worst case: {n - 1})")
    print(f"  Random insertion height:  {random_height} (ideal: {n.bit_length() - 1})")
    print()

    # Compare search times
    target = n // 2
    sorted_comps = search(sorted_root, target)
    random_comps = search(random_root, target)
    print(f"  Search for {target}:")
    print(f"    Sorted tree comparisons: {sorted_comps}")
    print(f"    Random tree comparisons: {random_comps}")
```

**Output:**
```
n = 1000
  Sorted insertion height:  999 (worst case: 999)
  Random insertion height:  22 (ideal: 9)

  Search for 500:
    Sorted tree comparisons: 500
    Random tree comparisons: 12
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12.4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
