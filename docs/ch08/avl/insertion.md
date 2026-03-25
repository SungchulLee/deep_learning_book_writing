# Insertion with Rebalancing

Inserting a new key into an ordinary binary search tree is straightforward: walk down from the root, go left if the key is smaller and right if larger, and attach the new node as a leaf. The challenge is that this may increase the height of a subtree, violating the AVL balance condition. AVL insertion solves this by walking back up the insertion path, checking balance factors, and applying at most **one rotation** (single or double) to restore the invariant.

## Insertion Algorithm

AVL insertion proceeds in two phases:

1. **BST insertion**: insert the key as a leaf, exactly as in a standard binary search tree.
2. **Fix-up walk**: retrace the path from the new leaf to the root, updating heights and rebalancing at the first node whose balance factor reaches $\pm 2$.

The fix-up walk is the heart of the algorithm. At each node $x$ on the path:

1. Update $h(x) = 1 + \max(h(\text{left}(x)),\, h(\text{right}(x)))$.
2. Compute $\text{BF}(x) = h(\text{left}(x)) - h(\text{right}(x))$.
3. If $|\text{BF}(x)| \leq 1$, move to the parent.
4. If $|\text{BF}(x)| = 2$, determine the rotation case and apply it.

## Rotation Cases

The rotation depends on the balance factor of $x$ and its heavy child:

| $\text{BF}(x)$ | Heavy child direction | $\text{BF}(\text{child})$ | Case | Fix |
|:-:|:-:|:-:|:-:|:--|
| $+2$ | Left child $y$ | $+1$ | Left-Left | Right rotation at $x$ |
| $+2$ | Left child $y$ | $-1$ | Left-Right | Left rotation at $y$, then right at $x$ |
| $-2$ | Right child $y$ | $-1$ | Right-Right | Left rotation at $x$ |
| $-2$ | Right child $y$ | $+1$ | Right-Left | Right rotation at $y$, then left at $x$ |

!!! info "At most one rotation per insertion"
    After performing the rotation at the lowest unbalanced ancestor, the subtree's height returns to its **pre-insertion** value. This means no ancestor above the rotation point can become unbalanced, so the fix-up terminates immediately. This is a fundamental difference from deletion, which may require $O(\log n)$ rotations.

## Why One Rotation Suffices

Before the insertion, let the subtree rooted at $x$ have height $h$. The insertion increases the height of one of $x$'s subtrees from $h-1$ to $h$, making $\text{BF}(x) = +2$ (or $-2$). After the rotation, the new root of this subtree has height $h$ --- the same as before the insertion. Since the height at $x$'s position did not change from the perspective of $x$'s parent, no further rebalancing is needed.

## Step-by-Step Example

Insert the keys 10, 20, 30 into an initially empty AVL tree.

**Insert 10**: The tree has a single node with $\text{BF} = 0$.

```
10 [BF=0]
```

**Insert 20**: Node 20 goes to the right of 10. Node 10 now has $\text{BF} = -1$.

```
10 [BF=-1]
  \
   20 [BF=0]
```

**Insert 30**: Node 30 goes to the right of 20. Now $\text{BF}(20) = -1$ and $\text{BF}(10) = -2$. This is a Right-Right case, so we perform a left rotation at node 10:

```
Before rotation:        After rotation:
10 [BF=-2]                 20 [BF=0]
  \                       /  \
   20 [BF=-1]           10    30
     \                [BF=0] [BF=0]
      30 [BF=0]
```

The tree is now balanced. The height at the position of the old root (10) went from 2 (before rotation) back to 1 (after rotation), matching the pre-insertion height.

## Implementation

```python
"""
AVL tree insertion with rebalancing.

Demonstrates the two-phase approach: standard BST insertion
followed by a bottom-up fix-up walk with at most one rotation.
"""


# === AVL Node ===

class AVLNode:
    """AVL tree node with key, children, and cached height."""

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
    """Right rotation at y."""
    x = y.left
    y.left = x.right
    x.right = y
    update_height(y)
    update_height(x)
    return x


def rotate_left(x):
    """Left rotation at x."""
    y = x.right
    x.right = y.left
    y.left = x
    update_height(x)
    update_height(y)
    return y


# === Insert with Rebalancing ===

def insert(node, key):
    """Insert key into AVL tree, rebalancing as needed.

    Returns the new root of the subtree.
    At most one rotation (single or double) is performed.
    """
    # Phase 1: BST insertion
    if node is None:
        return AVLNode(key)

    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        return node  # duplicate key, no insertion

    # Phase 2: fix-up
    update_height(node)
    bf = balance_factor(node)

    # Left-Left case
    if bf > 1 and balance_factor(node.left) >= 0:
        return rotate_right(node)

    # Left-Right case
    if bf > 1 and balance_factor(node.left) < 0:
        node.left = rotate_left(node.left)
        return rotate_right(node)

    # Right-Right case
    if bf < -1 and balance_factor(node.right) <= 0:
        return rotate_left(node)

    # Right-Left case
    if bf < -1 and balance_factor(node.right) > 0:
        node.right = rotate_right(node.right)
        return rotate_left(node)

    return node


# === Display ===

def print_tree(node, level=0):
    """Print tree sideways with balance factors."""
    if node is None:
        return
    print_tree(node.right, level + 1)
    bf = balance_factor(node)
    print(f"{'    ' * level}{node.key} [BF={bf:+d}]")
    print_tree(node.left, level + 1)


# === Inorder Traversal ===

def inorder(node):
    """Return sorted list of keys."""
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


if __name__ == "__main__":
    # Demonstrate all four rotation cases
    print("=== Right-Right case: insert 10, 20, 30 ===")
    root = None
    for key in [10, 20, 30]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    print("=== Left-Left case: insert 30, 20, 10 ===")
    root = None
    for key in [30, 20, 10]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    print("=== Left-Right case: insert 30, 10, 20 ===")
    root = None
    for key in [30, 10, 20]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    print("=== Right-Left case: insert 10, 30, 20 ===")
    root = None
    for key in [10, 30, 20]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    # Larger example
    print("=== Larger example: insert 50,30,70,20,40,60,80,10,25,35 ===")
    root = None
    for key in [50, 30, 70, 20, 40, 60, 80, 10, 25, 35]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
```

**Output:**
```
=== Right-Right case: insert 10, 20, 30 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Left-Left case: insert 30, 20, 10 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Left-Right case: insert 30, 10, 20 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Right-Left case: insert 10, 30, 20 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Larger example: insert 50,30,70,20,40,60,80,10,25,35 ===
        80 [BF=+0]
    70 [BF=+1]
        60 [BF=+0]
50 [BF=+0]
            40 [BF=+1]
                35 [BF=+0]
        30 [BF=+0]
            25 [BF=+0]
    20 [BF=-1]
        10 [BF=+0]
Inorder: [10, 20, 25, 30, 35, 40, 50, 60, 70, 80]
```

## Complexity

| Aspect | Cost |
|:--|:-:|
| BST walk to leaf | $O(\log n)$ |
| Fix-up walk (height updates) | $O(\log n)$ |
| Rotations | $O(1)$ (at most one single or double rotation) |
| **Total insertion time** | $O(\log n)$ |

The fix-up walk touches at most $O(\log n)$ ancestors, but only one of them requires a rotation. After that rotation, the subtree height returns to its pre-insertion value, so no further rebalancing is needed.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 13-14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
