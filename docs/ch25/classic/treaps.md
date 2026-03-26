# Randomized Treaps

A treap (tree + heap) is a randomized binary search tree that maintains
BST order on keys and heap order on randomly assigned priorities.
By choosing priorities uniformly at random, the treap's structure is
equivalent to a random BST — giving $O(\log n)$ expected height and
$O(\log n)$ expected time for search, insertion, and deletion, without
the complex rebalancing logic of AVL or red-black trees.

## Definition

A **treap** is a binary tree where each node stores a key-priority pair
$(k, p)$ satisfying two properties simultaneously:

1. **BST property on keys:** For every node, all keys in the left subtree
   are smaller, and all keys in the right subtree are larger.
2. **Min-heap property on priorities:** Each node's priority is less than
   or equal to the priorities of its children.

!!! note "Unique Structure"
    Given $n$ distinct keys and $n$ distinct priorities, there is exactly
    one treap satisfying both properties. The structure is uniquely
    determined by the priority ordering.

## Why Random Priorities Work

Assigning each key a uniformly random priority produces a treap whose
structure is identical (in distribution) to a **random BST** — a BST
built by inserting the keys in random order. A random BST has expected
depth $O(\log n)$ and expected height $\Theta(\log n)$.

**Key insight:** The node with the smallest priority becomes the root.
Its key splits the remaining nodes into left and right subtrees, and
the process recurses. This is exactly like choosing a random insertion
order.

## Expected Height

**Theorem.** The expected depth of any node in a treap with $n$ elements
is $O(\log n)$.

For a node with rank $i$ (the $i$-th smallest key), the expected depth is:

$$
E[\text{depth}(i)] = H_i + H_{n - i + 1} - 1
$$

where $H_k = \sum_{j=1}^{k} 1/j$ is the $k$-th harmonic number.
Since $H_k = O(\log k)$, the expected depth is $O(\log n)$.

## Rotations

Treap operations use **rotations** to restore the heap property after
BST insertions and deletions:

- **Right rotation** at node $x$: lifts $x$'s left child to $x$'s position.
- **Left rotation** at node $x$: lifts $x$'s right child to $x$'s position.

Rotations preserve the BST property while allowing us to move a
higher-priority node upward.

## Operations

### Search

Search works exactly as in a standard BST — ignore priorities and
follow the key ordering. Expected time: $O(\log n)$.

### Insertion

1. Insert the new node as a leaf using standard BST insertion.
2. Assign a random priority.
3. While the node's priority is smaller than its parent's, rotate it
   upward (right rotation if it's a left child, left rotation if right).

### Deletion

1. Find the node to delete.
2. Rotate it downward (toward the child with smaller priority) until
   it becomes a leaf.
3. Remove the leaf.

Alternatively, set the node's priority to $\infty$ and let it sink to a
leaf position through rotations.

## Implementation

```python
"""
Randomized treap: BST + heap with random priorities.

Supports search, insert, and delete in O(log n) expected time
using rotations to maintain the heap property on priorities.
"""

import random


# === Treap Node ===

class TreapNode:
    """A node in the treap."""

    def __init__(self, key, priority=None):
        self.key = key
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None


# === Rotations ===

def rotate_right(node):
    """Right rotation: lift node's left child."""
    new_root = node.left
    node.left = new_root.right
    new_root.right = node
    return new_root


def rotate_left(node):
    """Left rotation: lift node's right child."""
    new_root = node.right
    node.right = new_root.left
    new_root.left = node
    return new_root


# === Insert ===

def insert(root, key):
    """Insert a key into the treap.

    Returns the new root of the subtree.
    """
    if root is None:
        return TreapNode(key)

    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority < root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority < root.priority:
            root = rotate_left(root)

    return root


# === Delete ===

def delete(root, key):
    """Delete a key from the treap.

    Returns the new root of the subtree.
    """
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
        else:
            # Rotate toward the child with smaller priority
            if root.left.priority < root.right.priority:
                root = rotate_right(root)
                root.right = delete(root.right, key)
            else:
                root = rotate_left(root)
                root.left = delete(root.left, key)

    return root


# === Search ===

def search(root, key):
    """Search for a key in the treap."""
    if root is None:
        return False
    if key == root.key:
        return True
    elif key < root.key:
        return search(root.left, key)
    else:
        return search(root.right, key)


# === Inorder Traversal ===

def inorder(root):
    """Return keys in sorted order."""
    if root is None:
        return []
    return inorder(root.left) + [root.key] + inorder(root.right)


# === Tree Height ===

def height(root):
    """Compute the height of the treap."""
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))


# === Main ===

if __name__ == "__main__":
    random.seed(42)
    root = None

    keys = [5, 2, 8, 1, 4, 7, 9, 3, 6]
    for k in keys:
        root = insert(root, k)

    print(f"Inserted: {keys}")
    print(f"Inorder:  {inorder(root)}")
    print(f"Height:   {height(root)}")
    print(f"Root:     key={root.key}, priority={root.priority:.4f}")

    # Search
    for k in [4, 10]:
        print(f"Search {k}: {search(root, k)}")

    # Delete
    root = delete(root, 5)
    print(f"\nAfter deleting 5:")
    print(f"Inorder:  {inorder(root)}")
    print(f"Height:   {height(root)}")

    # Average height over many treaps
    total_height = 0
    n = 1000
    trials = 100
    for _ in range(trials):
        r = None
        for k in range(n):
            r = insert(r, k)
        total_height += height(r)
    avg_h = total_height / trials
    print(f"\nAverage height of treap with {n} keys: {avg_h:.1f}")
    print(f"Expected O(log n) = {2 * 13.8:.1f} (2 * ln 1000)")
```

## Split and Merge

Treaps support efficient **split** and **merge** operations, making them
useful for implementing sequences and interval operations.

**Split(T, k):** Split treap $T$ into $T_1$ (keys $\le k$) and $T_2$
(keys $> k$) in $O(\log n)$ expected time.

**Merge(T_1, T_2):** Merge two treaps where all keys in $T_1$ are
smaller than all keys in $T_2$, in $O(\log n)$ expected time.

## Complexity Summary

| Operation | Expected | Worst Case |
|---|---|---|
| Search | $O(\log n)$ | $O(n)$ |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |
| Split | $O(\log n)$ | $O(n)$ |
| Merge | $O(\log n)$ | $O(n)$ |
| Space | $O(n)$ | $O(n)$ |

## Reference

- Aragon, C. R. & Seidel, R. "Randomized Search Trees." *Algorithmica*, 1996.
- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press.
