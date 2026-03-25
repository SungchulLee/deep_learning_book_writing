# Double Rotation

A single rotation corrects an imbalance when the heavy subtree leans in the same direction as its parent --- both left-left or both right-right. However, when the imbalance follows a **zig-zag** pattern (left-right or right-left), a single rotation fails to restore balance. In these cases, AVL trees apply a **double rotation**: two single rotations composed into one operation that straightens the zig-zag into a line before fixing the height.

## When Single Rotation Fails

Consider node $z$ with $\text{BF}(z) = +2$, meaning its left subtree is too tall. If the heavy path goes left then **right** (through $z$'s left child $y$ and then $y$'s right child $x$), a single right rotation at $z$ does not fix the problem. It merely moves the tall right subtree of $y$ to the other side without reducing the overall height.

The key insight is that the node $x$ causing the imbalance is "between" $y$ and $z$ in value, so it must become the new root of this subtree. A double rotation achieves this by first rotating $x$ up to $y$'s position, then rotating $x$ up to $z$'s position.

## Left-Right Double Rotation

This handles the case where $\text{BF}(z) = +2$ and $\text{BF}(y) = -1$, where $y$ is the left child of $z$.

**Step 1: Left rotation at $y$** (the left child of $z$)

```
      z                z
     / \              / \
    y   D    →       x   D
   / \              / \
  A   x            y   C
     / \          / \
    B   C        A   B
```

**Step 2: Right rotation at $z$**

```
      z                x
     / \              / \
    x   D    →       y   z
   / \              / \ / \
  y   C            A  B C  D
 / \
A   B
```

After the double rotation, $x$ sits at the root with $y$ as its left child and $z$ as its right child. The four subtrees $A, B, C, D$ are distributed so that BST ordering is preserved and all balance factors return to $\{-1, 0, +1\}$.

### Formal Height Analysis

Let the subtrees $A, B, C, D$ have heights $a, b, c, d$ respectively. Before the double rotation:

- $h(y) = 1 + \max(a, 1 + \max(b, c)) = 1 + (1 + \max(b, c))$ since $\text{BF}(y) = -1$
- $h(z) = 1 + \max(h(y), d)$ with $\text{BF}(z) = +2$

After the double rotation, node $x$ has:

$$
h(\text{left of } x) = 1 + \max(a, b)
$$

$$
h(\text{right of } x) = 1 + \max(c, d)
$$

Since $\max(b, c) = a = d$ (from the balance conditions), both children of $x$ have the same height, giving $\text{BF}(x) = 0$ or $|\text{BF}(x)| \leq 1$.

## Right-Left Double Rotation

This is the symmetric case where $\text{BF}(z) = -2$ and $\text{BF}(y) = +1$, with $y$ the right child of $z$.

**Step 1: Right rotation at $y$** (the right child of $z$)

```
    z                z
   / \              / \
  A   y    →       A   x
     / \              / \
    x   D            B   y
   / \                  / \
  B   C                C   D
```

**Step 2: Left rotation at $z$**

```
    z                  x
   / \                / \
  A   x      →      z   y
     / \            / \ / \
    B   y          A  B C  D
       / \
      C   D
```

The mechanics are a perfect mirror of the left-right case, with left and right swapped.

## Implementation

```python
"""
AVL double rotations: left-right and right-left.

Shows how two single rotations compose to fix zig-zag imbalances
that a single rotation cannot resolve.
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

def rotate_left(x):
    """Left rotation at x."""
    y = x.right
    x.right = y.left
    y.left = x
    update_height(x)
    update_height(y)
    return y


def rotate_right(y):
    """Right rotation at y."""
    x = y.left
    y.left = x.right
    x.right = y
    update_height(y)
    update_height(x)
    return x


# === Double Rotations ===

def left_right_rotate(z):
    """Left-right double rotation at z.

    First left-rotate z's left child, then right-rotate z.
    Used when BF(z) = +2 and BF(z.left) = -1.
    """
    z.left = rotate_left(z.left)
    return rotate_right(z)


def right_left_rotate(z):
    """Right-left double rotation at z.

    First right-rotate z's right child, then left-rotate z.
    Used when BF(z) = -2 and BF(z.right) = +1.
    """
    z.right = rotate_right(z.right)
    return rotate_left(z)


# === Rebalance (unified) ===

def rebalance(node):
    """Apply single or double rotation as needed."""
    bf = balance_factor(node)
    if bf > 1:
        if balance_factor(node.left) < 0:
            return left_right_rotate(node)  # double
        return rotate_right(node)           # single
    if bf < -1:
        if balance_factor(node.right) > 0:
            return right_left_rotate(node)  # double
        return rotate_left(node)            # single
    return node


# === Insert ===

def insert(node, key):
    """Insert key into AVL tree."""
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
    # Demonstrate left-right double rotation
    # Insert 30, 10, 20 -> triggers LR rotation at 30
    print("=== Left-Right Double Rotation ===")
    root = None
    for key in [30, 10, 20]:
        root = insert(root, key)
        print(f"After inserting {key}:")
        print_tree(root)
        print()

    # Demonstrate right-left double rotation
    # Insert 10, 30, 20 -> triggers RL rotation at 10
    print("=== Right-Left Double Rotation ===")
    root = None
    for key in [10, 30, 20]:
        root = insert(root, key)
        print(f"After inserting {key}:")
        print_tree(root)
        print()
```

**Output:**
```
=== Left-Right Double Rotation ===
After inserting 30:
30 [BF=+0]

After inserting 10:
30 [BF=+1]
    10 [BF=+0]

After inserting 20:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]

=== Right-Left Double Rotation ===
After inserting 10:
10 [BF=+0]

After inserting 30:
    30 [BF=+0]
10 [BF=-1]

After inserting 20:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
```

In both cases, the zig-zag pattern (30-10-20 or 10-30-20) produces node 20 as the balanced root, which no single rotation could achieve.

## Summary of Rotation Cases

| Imbalance Pattern | $\text{BF}(z)$ | $\text{BF}(\text{child})$ | Fix |
|:--|:-:|:-:|:--|
| Left-Left (straight) | $+2$ | $+1$ or $0$ | Single right rotation |
| Left-Right (zig-zag) | $+2$ | $-1$ | **LR double rotation** |
| Right-Right (straight) | $-2$ | $-1$ or $0$ | Single left rotation |
| Right-Left (zig-zag) | $-2$ | $+1$ | **RL double rotation** |

Double rotations cost two pointer updates more than single rotations but still run in $O(1)$ time, preserving the $O(\log n)$ complexity of AVL insertion and deletion.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 13-14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
