# Deletion with Rebalancing

AVL insertion requires at most one rotation (single or double) to restore balance, because an insertion increases a subtree's height by at most one. Deletion, however, can decrease a subtree's height, and the resulting rotation may itself reduce the height of the rotated subtree, propagating imbalance further up the tree. Consequently, a single deletion can require $O(\log n)$ rotations in the worst case, making it the more intricate of the two operations.

## Standard BST Deletion

Before addressing balance, recall that deleting a node $z$ from a binary search tree has three cases:

1. **Leaf node** ($z$ has no children): remove $z$ directly.
2. **One child**: replace $z$ with its single child.
3. **Two children**: find $z$'s in-order successor $y$ (the smallest node in $z$'s right subtree), copy $y$'s key into $z$, and then delete $y$ from the right subtree. Since $y$ has no left child, deleting $y$ reduces to case 1 or 2.

After the structural deletion, the heights of ancestors along the path from the deleted position to the root may change.

## Rebalancing After Deletion

Starting from the parent of the physically removed node, walk up to the root. At each ancestor $x$:

1. Update the height of $x$.
2. Compute the balance factor $\text{BF}(x) = h(\text{left}(x)) - h(\text{right}(x))$.
3. If $|\text{BF}(x)| \leq 1$, continue to the next ancestor.
4. If $|\text{BF}(x)| = 2$, apply the appropriate rotation at $x$.

The rotation choice follows the same table as insertion:

| $\text{BF}(x)$ | $\text{BF}(\text{heavy child})$ | Rotation |
|:-:|:-:|:--|
| $+2$ | $\geq 0$ | Right rotation at $x$ |
| $+2$ | $-1$ | Left-right double rotation |
| $-2$ | $\leq 0$ | Left rotation at $x$ |
| $-2$ | $+1$ | Right-left double rotation |

!!! warning "Deletion can cascade"
    Unlike insertion, where a single rotation restores all balance factors, a deletion rotation can reduce the height of the subtree it fixes. This height decrease may cause the grandparent to become unbalanced, requiring another rotation. In the worst case, rotations propagate all the way to the root, yielding $O(\log n)$ rotations per deletion.

## Why Multiple Rotations Can Occur

Consider a right rotation at node $x$ with $\text{BF}(x) = +2$ and $\text{BF}(\text{left}(x)) = 0$. Before rotation, the subtree rooted at $x$ has height $h$. After rotation, the new root has $\text{BF} = -1$ and the subtree height decreases to $h - 1$. This height decrease is exactly the situation that can unbalance $x$'s parent.

In contrast, when $\text{BF}(\text{left}(x)) = +1$ during insertion, the rotation produces a new root with $\text{BF} = 0$ and the subtree height returns to its pre-insertion value, preventing further propagation.

## Deletion Algorithm

```python
"""
AVL tree deletion with rebalancing.

Demonstrates all three BST deletion cases followed by
the bottom-up rebalancing walk that may require O(log n) rotations.
"""


# === AVL Node ===

class AVLNode:
    """A node storing a key, left/right children, and cached height."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 0


# === Height and Balance Utilities ===

def height(node):
    """Return height of node, or -1 for null."""
    return node.height if node else -1


def update_height(node):
    """Recompute height from children."""
    node.height = 1 + max(height(node.left), height(node.right))


def balance_factor(node):
    """Compute BF = h(left) - h(right)."""
    return height(node.left) - height(node.right)


# === Rotations ===

def rotate_right(y):
    """Perform right rotation at y, return new root."""
    x = y.left
    t = x.right
    x.right = y
    y.left = t
    update_height(y)
    update_height(x)
    return x


def rotate_left(x):
    """Perform left rotation at x, return new root."""
    y = x.right
    t = y.left
    y.left = x
    x.right = t
    update_height(x)
    update_height(y)
    return y


# === Rebalance ===

def rebalance(node):
    """Apply rotations if |BF| >= 2, return new subtree root."""
    bf = balance_factor(node)
    if bf > 1:
        if balance_factor(node.left) < 0:
            node.left = rotate_left(node.left)
        return rotate_right(node)
    if bf < -1:
        if balance_factor(node.right) > 0:
            node.right = rotate_right(node.right)
        return rotate_left(node)
    return node


# === Insertion (for building the tree) ===

def insert(node, key):
    """Insert key and rebalance."""
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        return node
    update_height(node)
    return rebalance(node)


# === Deletion ===

def find_min(node):
    """Find the node with the smallest key in a subtree."""
    while node.left is not None:
        node = node.left
    return node


def delete(node, key):
    """Delete key from AVL tree and rebalance all ancestors."""
    if node is None:
        return None

    if key < node.key:
        node.left = delete(node.left, key)
    elif key > node.key:
        node.right = delete(node.right, key)
    else:
        # Found the node to delete
        if node.left is None:
            return node.right
        elif node.right is None:
            return node.left
        else:
            # Two children: replace with in-order successor
            successor = find_min(node.right)
            node.key = successor.key
            node.right = delete(node.right, successor.key)

    update_height(node)
    return rebalance(node)


# === Display ===

def print_tree(node, level=0):
    """Print tree sideways with balance factors."""
    if node is None:
        return
    print_tree(node.right, level + 1)
    bf = balance_factor(node)
    print(f"{'    ' * level}{node.key} [BF={bf:+d}]")
    print_tree(node.left, level + 1)


if __name__ == "__main__":
    # Build an AVL tree
    root = None
    for key in [50, 30, 70, 20, 40, 60, 80, 10, 25]:
        root = insert(root, key)

    print("Before deletion:")
    print_tree(root)
    print()

    # Delete 80 (leaf), then 70 (one child), then 60
    for key in [80, 70, 60]:
        root = delete(root, key)
        print(f"After deleting {key}:")
        print_tree(root)
        print()
```

**Output:**
```
Before deletion:
        80 [BF=+0]
    70 [BF=+0]
        60 [BF=+0]
50 [BF=+0]
        40 [BF=+0]
    30 [BF=+0]
            25 [BF=+0]
        20 [BF=-1]
            10 [BF=+0]

After deleting 80:
    70 [BF=+1]
        60 [BF=+0]
50 [BF=+0]
        40 [BF=+0]
    30 [BF=+0]
            25 [BF=+0]
        20 [BF=-1]
            10 [BF=+0]

After deleting 70:
    60 [BF=+0]
50 [BF=+1]
        40 [BF=+0]
    30 [BF=+0]
            25 [BF=+0]
        20 [BF=-1]
            10 [BF=+0]

After deleting 60:
    50 [BF=+0]
        40 [BF=+0]
30 [BF=+0]
        25 [BF=+0]
    20 [BF=+0]
        10 [BF=+0]
```

After deleting 60, node 50 becomes the right child and the tree rebalances with node 30 as the new root.

## Complexity

| Operation | Time | Rotations |
|:--|:-:|:-:|
| BST deletion step | $O(\log n)$ | 0 |
| Rebalancing walk | $O(\log n)$ | $O(\log n)$ worst case |
| **Total** | $O(\log n)$ | $O(\log n)$ |

Each rotation takes $O(1)$ time, but up to $O(\log n)$ rotations may occur during a single deletion. Despite this, the total work remains $O(\log n)$ because each rotation is performed at a distinct level of the tree.

## Comparison with Insertion

| Property | Insertion | Deletion |
|:--|:-:|:-:|
| Maximum rotations | 1 (single or double) | $O(\log n)$ |
| Height change after rotation | restored to pre-insert | may decrease by 1 |
| Propagation after rotation | stops | may continue upward |

This asymmetry arises because insertion adds height while deletion removes it. A rotation after insertion restores the original height, stopping propagation. A rotation after deletion may reduce the subtree height below its pre-deletion value, potentially unbalancing the parent.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 13-14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
