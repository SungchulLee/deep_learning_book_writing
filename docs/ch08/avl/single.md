# Single Rotation

When a node in an AVL tree becomes unbalanced and the heavy subtree leans in the **same direction** as the imbalance --- left child of the left subtree (Left-Left) or right child of the right subtree (Right-Right) --- a single rotation restores balance. Single rotations are the simplest rebalancing tool in AVL trees: one pointer reassignment raises the heavy child to take the place of its unbalanced parent, reducing the subtree height by exactly one.

## Right Rotation (for Left-Left imbalance)

When node $z$ has $\text{BF}(z) = +2$ and its left child $y$ has $\text{BF}(y) \geq 0$, the heavy path runs straight down the left side. A **right rotation** at $z$ makes $y$ the new root:

```
Before:           After:
    z                y
   / \              / \
  y   C    →       A   z
 / \                  / \
A   B                B   C
```

The operation reassigns three pointers:

1. $z.\text{left} \leftarrow y.\text{right}$ (subtree $B$ moves to become $z$'s left child)
2. $y.\text{right} \leftarrow z$ (old root $z$ becomes $y$'s right child)
3. Update the parent of $z$ to point to $y$ instead

### Correctness

The BST property is preserved because:

- All keys in $A$ are less than $y.\text{key}$ (unchanged).
- $y.\text{key} < z.\text{key}$ (BST property of the original tree).
- All keys in $B$ satisfy $y.\text{key} < B.\text{key} < z.\text{key}$ (BST property). Moving $B$ from $y$'s right to $z$'s left preserves this.
- All keys in $C$ are greater than $z.\text{key}$ (unchanged).

### Height Analysis

Let $h(A) = a$, $h(B) = b$, $h(C) = c$. Before the rotation:

- $h(y) = 1 + \max(a, b)$
- $h(z) = 1 + \max(h(y), c) = 1 + h(y)$ since $\text{BF}(z) = +2$

After the rotation:

$$
h(z_{\text{new}}) = 1 + \max(b, c)
$$

$$
h(y_{\text{new}}) = 1 + \max(a, h(z_{\text{new}})) = 1 + \max(a, 1 + \max(b, c))
$$

When $\text{BF}(y) = +1$ (the insertion case), we have $a = b + 1$ and $c = a - 1 = b$. Then $h(z_{\text{new}}) = 1 + b$ and $h(y_{\text{new}}) = 1 + a = 2 + b$, giving $\text{BF}(y_{\text{new}}) = 0$.

## Left Rotation (for Right-Right imbalance)

When node $z$ has $\text{BF}(z) = -2$ and its right child $y$ has $\text{BF}(y) \leq 0$, a **left rotation** at $z$ is the mirror:

```
Before:           After:
  z                  y
 / \                / \
A   y      →       z   C
   / \            / \
  B   C          A   B
```

The pointer reassignments mirror the right rotation:

1. $z.\text{right} \leftarrow y.\text{left}$ (subtree $B$ moves to $z$'s right child)
2. $y.\text{left} \leftarrow z$ ($z$ becomes $y$'s left child)
3. Update $z$'s parent to point to $y$

## Implementation

```python
"""
AVL single rotations: left and right.

Demonstrates the fundamental pointer operations that fix
Left-Left and Right-Right imbalances in O(1) time.
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


# === Single Rotations ===

def rotate_right(z):
    """Right rotation at z (fixes Left-Left imbalance).

    Before:     z          After:     y
               / \\                  / \\
              y   C                A   z
             / \\                    / \\
            A   B                  B   C
    """
    y = z.left
    b = y.right

    # Perform rotation
    y.right = z
    z.left = b

    # Update heights (z first, since y depends on z)
    update_height(z)
    update_height(y)

    return y  # new root


def rotate_left(z):
    """Left rotation at z (fixes Right-Right imbalance).

    Before:   z            After:     y
             / \\                    / \\
            A   y                  z   C
               / \\              / \\
              B   C            A   B
    """
    y = z.right
    b = y.left

    # Perform rotation
    y.left = z
    z.right = b

    # Update heights (z first, since y depends on z)
    update_height(z)
    update_height(y)

    return y  # new root


# === Insert with single-rotation rebalancing ===

def insert(node, key):
    """Insert key, applying single rotations for LL/RR cases."""
    if node is None:
        return AVLNode(key)

    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        return node

    update_height(node)
    bf = balance_factor(node)

    # Left-Left case: single right rotation
    if bf > 1 and balance_factor(node.left) >= 0:
        return rotate_right(node)

    # Right-Right case: single left rotation
    if bf < -1 and balance_factor(node.right) <= 0:
        return rotate_left(node)

    # Left-Right and Right-Left cases handled by double rotation
    # (covered in the Double Rotation page)
    if bf > 1 and balance_factor(node.left) < 0:
        node.left = rotate_left(node.left)
        return rotate_right(node)
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


if __name__ == "__main__":
    # Left-Left case: inserting 30, 20, 10
    print("=== Left-Left Case (Right Rotation) ===")
    print("Inserting 30, 20, 10:")
    root = None
    for key in [30, 20, 10]:
        root = insert(root, key)
    print_tree(root)
    print()

    # Right-Right case: inserting 10, 20, 30
    print("=== Right-Right Case (Left Rotation) ===")
    print("Inserting 10, 20, 30:")
    root = None
    for key in [10, 20, 30]:
        root = insert(root, key)
    print_tree(root)
    print()

    # A longer sequence showing multiple single rotations
    print("=== Sorted Insertion (multiple rotations) ===")
    print("Inserting 1, 2, 3, 4, 5, 6, 7:")
    root = None
    for key in range(1, 8):
        root = insert(root, key)
    print_tree(root)
```

**Output:**
```
=== Left-Left Case (Right Rotation) ===
Inserting 30, 20, 10:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]

=== Right-Right Case (Left Rotation) ===
Inserting 10, 20, 30:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]

=== Sorted Insertion (multiple rotations) ===
Inserting 1, 2, 3, 4, 5, 6, 7:
        7 [BF=+0]
    6 [BF=+0]
        5 [BF=+0]
4 [BF=+0]
        3 [BF=+0]
    2 [BF=+0]
        1 [BF=+0]
```

Inserting keys in sorted order would create a degenerate chain in a plain BST. The AVL tree applies left rotations at each step, producing a perfectly balanced tree of height 2.

## Complexity

Each single rotation performs a constant number of pointer reassignments and two height updates:

| Operation | Cost |
|:--|:-:|
| Pointer reassignments | $O(1)$ |
| Height updates | $O(1)$ |
| **Total per rotation** | $O(1)$ |

The rotation itself is $O(1)$. The overall insertion cost is $O(\log n)$ due to the BST walk and the fix-up path, not the rotation.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 13-14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
