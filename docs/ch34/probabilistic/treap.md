# Treaps

Balanced BSTs like AVL and red-black trees maintain balance through deterministic invariants and rotations. A **treap** (tree + heap) achieves expected $O(\log n)$ balance using randomization instead: each node receives a random priority, and the tree simultaneously satisfies the BST property on keys and the heap property on priorities. This combination uniquely determines the tree shape and produces a random BST equivalent to inserting elements in random order.

## Definition

A treap is a binary tree where each node stores a (key, priority) pair such that:

1. **BST property**: For every node $x$, all keys in the left subtree are less than $x.\text{key}$, and all keys in the right subtree are greater.
2. **Heap property**: For every node $x$, $x.\text{priority} \ge \text{priority of both children}$ (max-heap on priorities).

If all priorities are distinct, the treap structure is unique for a given set of (key, priority) pairs.

## Expected Height

When priorities are drawn independently and uniformly at random, the resulting treap has the same distribution as a **random BST** built by inserting elements in a uniformly random permutation. The expected depth of any node is:

$$
E[\text{depth of node with rank } k] = H_k + H_{n-k+1} - 1
$$

where $H_k = \sum_{i=1}^{k} 1/i$ is the $k$-th harmonic number. The maximum expected depth over all nodes is:

$$
E[h] = O(\log n)
$$

## Rotations

When the heap property is violated after an insert, tree rotations restore it without breaking the BST property:

- **Right rotation** at node $x$: the left child $y$ becomes the parent, $x$ becomes $y$'s right child.
- **Left rotation** at node $x$: the right child $y$ becomes the parent, $x$ becomes $y$'s left child.

## Insert

To insert (key $k$, priority $p$):

1. Perform standard BST insert based on $k$, placing the new node as a leaf.
2. While the new node's priority exceeds its parent's priority, rotate the new node up (right rotation if it is a left child, left rotation if it is a right child).

The number of rotations equals the depth of the initial insertion point, giving:

$$
E[T_{\text{insert}}] = O(\log n)
$$

## Delete

To delete key $k$:

1. Find the node $x$ with key $k$.
2. Rotate $x$ down (toward the child with higher priority) until $x$ becomes a leaf.
3. Remove the leaf.

$$
E[T_{\text{delete}}] = O(\log n)
$$

## Split and Merge

Treaps support efficient split and merge operations:

**Split(root, key)**: Split the treap into two treaps $L$ and $R$ where all keys in $L$ are $\le$ key and all keys in $R$ are $>$ key. Expected time $O(\log n)$.

**Merge(L, R)**: Merge two treaps where all keys in $L$ are less than all keys in $R$. Compare root priorities and recursively merge. Expected time $O(\log n)$.

## Implementation

```python
"""
Treap -- randomized BST with heap-ordered priorities.

Achieves O(log n) expected time for search, insert, and delete
by assigning random priorities and maintaining the heap property.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field


# === Treap Node ===============================================================

@dataclass
class TreapNode:
    """Node storing a key and a random priority."""
    key: int
    priority: float = field(default_factory=random.random)
    left: TreapNode | None = None
    right: TreapNode | None = None


# === Rotations ================================================================

def rotate_right(node: TreapNode) -> TreapNode:
    """Right rotation: left child becomes root."""
    new_root = node.left
    node.left = new_root.right
    new_root.right = node
    return new_root


def rotate_left(node: TreapNode) -> TreapNode:
    """Left rotation: right child becomes root."""
    new_root = node.right
    node.right = new_root.left
    new_root.left = node
    return new_root


# === Treap Operations =========================================================

def insert(root: TreapNode | None, key: int) -> TreapNode:
    """Insert *key* with a random priority, maintaining both properties."""
    if root is None:
        return TreapNode(key)
    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority > root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority > root.priority:
            root = rotate_left(root)
    return root  # duplicate key: no change


def search(root: TreapNode | None, key: int) -> bool:
    """Search for *key* using standard BST search."""
    if root is None:
        return False
    if key == root.key:
        return True
    elif key < root.key:
        return search(root.left, key)
    else:
        return search(root.right, key)


def delete(root: TreapNode | None, key: int) -> TreapNode | None:
    """Delete *key* by rotating it down to a leaf."""
    if root is None:
        return None
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # Found the node to delete
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        elif root.left.priority > root.right.priority:
            root = rotate_right(root)
            root.right = delete(root.right, key)
        else:
            root = rotate_left(root)
            root.left = delete(root.left, key)
    return root


def inorder(root: TreapNode | None) -> list[int]:
    """In-order traversal returning sorted keys."""
    if root is None:
        return []
    return inorder(root.left) + [root.key] + inorder(root.right)


def height(root: TreapNode | None) -> int:
    """Compute the height of the treap."""
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))


# === Main =====================================================================

if __name__ == "__main__":
    random.seed(42)
    root = None
    keys = [5, 3, 7, 1, 4, 6, 8, 2, 9]
    for k in keys:
        root = insert(root, k)

    print(f"Sorted: {inorder(root)}")
    print(f"Height: {height(root)}")
    print(f"Search 4: {search(root, 4)}")
    print(f"Search 10: {search(root, 10)}")

    root = delete(root, 5)
    print(f"After deleting 5: {inorder(root)}")
```

**Output:**

```
Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9]
Height: 4
Search 4: True
Search 10: False
After deleting 5: [1, 2, 3, 4, 6, 7, 8, 9]
```

The in-order traversal confirms the BST property, the height is close to $\log_2 9 \approx 3.2$ (randomization keeps it balanced), and deletion correctly removes the key while maintaining order.

## Reference

- Seidel, R. and Aragon, C.R. "Randomized Search Trees." *Algorithmica*, 1996
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
