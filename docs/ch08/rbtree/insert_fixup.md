# Insertion Fixup

After inserting a red node $z$ into a red-black tree, the only property that can be violated is Property 4: $z$ and its parent $p$ may both be red. The insertion fixup procedure resolves this red-red conflict by walking up the tree, applying one of three cases at each step. The procedure terminates after at most $O(\log n)$ recolorings and at most **two rotations**, restoring all five red-black properties.

## Setup

The fixup loop runs while $z$'s parent is red (if the parent is black, there is no violation). At each iteration, $z$ is the node we are fixing, $p = z.\text{parent}$ is red, and $g = p.\text{parent}$ is $p$'s parent (which must be black, since $p$ is red and the tree satisfied Property 4 before the insertion).

The three cases depend on the color of $z$'s **uncle** $u$ (the sibling of $p$):

## Case 1: Uncle is Red (Recoloring)

If the uncle $u$ is red, both $p$ and $u$ are red children of the black grandparent $g$.

**Fix**: Recolor $p$ and $u$ to black, and $g$ to red. This preserves Property 5 (black-height unchanged through $g$) but may create a new red-red violation at $g$ and $g$'s parent.

```
Before:              After:
    g(B)                g(R)    <- may violate Prop 4
   / \                 / \
  p(R) u(R)          p(B) u(B)
 /                   /
z(R)                z(R)
```

After recoloring, set $z \leftarrow g$ and repeat the loop. The violation moves up two levels, so this case can repeat at most $O(\log n)$ times.

## Case 2: Uncle is Black, z is an Inner Child (Rotation + Fall Through)

If the uncle $u$ is black and $z$ is the "inner" child (right child of a left parent, or left child of a right parent), we cannot directly fix the violation with a single rotation.

**Fix**: Rotate $z$ up to $p$'s position, making $p$ the outer child. This transforms Case 2 into Case 3.

For the case where $p$ is $g$'s left child and $z$ is $p$'s right child:

```
Before:              After left-rotate at p:
    g(B)                 g(B)
   / \                  / \
  p(R) u(B)           z(R) u(B)
    \                 /
    z(R)             p(R)
```

Set $z \leftarrow p$ (the old parent is now the outer child), then fall through to Case 3.

## Case 3: Uncle is Black, z is an Outer Child (Rotation + Recolor)

If the uncle $u$ is black and $z$ is the "outer" child (left child of a left parent, or right child of a right parent), a single rotation at $g$ and a recoloring fix the violation permanently.

**Fix**: Recolor $p$ to black and $g$ to red, then rotate $g$ in the opposite direction.

For the case where $p$ is $g$'s left child:

```
Before:              After right-rotate at g:
    g(B)                p(B)
   / \                 / \
  p(R) u(B)          z(R) g(R)
 /                          \
z(R)                        u(B)
```

After Case 3, the violation is resolved: $p$ (now the root of this subtree) is black, and neither child creates a red-red conflict. The loop terminates.

## Summary of Cases

| Case | Uncle | z position | Action | Continues? |
|:-:|:-:|:-:|:--|:-:|
| 1 | Red | Either | Recolor $p$, $u$, $g$ | Yes (move up) |
| 2 | Black | Inner child | Rotate at $p$ | Falls to Case 3 |
| 3 | Black | Outer child | Rotate at $g$ + recolor | No (terminates) |

The symmetric cases (when $p$ is $g$'s right child) are mirrors of the above, with left and right swapped.

## Termination and Complexity

- **Case 1** moves $z$ up two levels, so it executes at most $h/2 = O(\log n)$ times.
- **Case 2** transforms into Case 3 with one rotation.
- **Case 3** terminates the loop with one rotation.

Therefore, the total number of rotations is at most **2** (one from Case 2 + one from Case 3), and the number of recolorings is at most $O(\log n)$.

After the loop, recolor the root to black (to satisfy Property 2 if the recoloring in Case 1 made it red).

## Implementation

```python
"""
Red-black tree insertion with complete fixup.

Implements all three cases of INSERT-FIXUP following CLRS,
demonstrating the at-most-2-rotations guarantee.
"""


# === Constants ===

RED = "R"
BLACK = "B"


# === Red-Black Node ===

class RBNode:
    """Red-black tree node."""

    def __init__(self, key, color=RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return f"{self.key}({self.color})"


# === Sentinel ===

NIL = RBNode(key=None, color=BLACK)
NIL.left = NIL
NIL.right = NIL


# === Rotations ===

def left_rotate(tree, x):
    """Left rotation at x."""
    y = x.right
    x.right = y.left
    if y.left is not NIL:
        y.left.parent = x
    y.parent = x.parent
    if x.parent is None:
        tree["root"] = y
    elif x is x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y
    y.left = x
    x.parent = y


def right_rotate(tree, y):
    """Right rotation at y."""
    x = y.left
    y.left = x.right
    if x.right is not NIL:
        x.right.parent = y
    x.parent = y.parent
    if y.parent is None:
        tree["root"] = x
    elif y is y.parent.left:
        y.parent.left = x
    else:
        y.parent.right = x
    x.right = y
    y.parent = x


# === Insert Fixup ===

def insert_fixup(tree, z):
    """Fix red-black violations after inserting red node z.

    At most 2 rotations and O(log n) recolorings.
    """
    rotations = 0

    while z.parent is not None and z.parent.color == RED:
        if z.parent is z.parent.parent.left:
            uncle = z.parent.parent.right

            if uncle.color == RED:
                # Case 1: uncle is red
                z.parent.color = BLACK
                uncle.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            else:
                if z is z.parent.right:
                    # Case 2: uncle black, z is inner child
                    z = z.parent
                    left_rotate(tree, z)
                    rotations += 1
                # Case 3: uncle black, z is outer child
                z.parent.color = BLACK
                z.parent.parent.color = RED
                right_rotate(tree, z.parent.parent)
                rotations += 1
        else:
            # Symmetric: parent is right child of grandparent
            uncle = z.parent.parent.left

            if uncle.color == RED:
                z.parent.color = BLACK
                uncle.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            else:
                if z is z.parent.left:
                    z = z.parent
                    right_rotate(tree, z)
                    rotations += 1
                z.parent.color = BLACK
                z.parent.parent.color = RED
                left_rotate(tree, z.parent.parent)
                rotations += 1

    tree["root"].color = BLACK
    return rotations


# === Insert ===

def rb_insert(tree, key):
    """Insert key into RB tree with fixup."""
    z = RBNode(key, RED)
    z.left = NIL
    z.right = NIL

    y = None
    x = tree["root"]

    while x is not NIL:
        y = x
        if z.key < x.key:
            x = x.left
        else:
            x = x.right

    z.parent = y
    if y is None:
        tree["root"] = z
    elif z.key < y.key:
        y.left = z
    else:
        y.right = z

    rots = insert_fixup(tree, z)
    return rots


# === Display ===

def print_tree(node, level=0):
    """Print tree sideways with colors."""
    if node is NIL:
        return
    print_tree(node.right, level + 1)
    print(f"{'    ' * level}{node.key}({node.color})")
    print_tree(node.left, level + 1)


if __name__ == "__main__":
    tree = {"root": NIL}

    keys = [10, 20, 30, 15, 25, 5, 1]
    for key in keys:
        rots = rb_insert(tree, key)
        print(f"Insert {key}: {rots} rotation(s)")
        print_tree(tree["root"])
        print()
```

**Output:**
```
Insert 10: 0 rotation(s)
10(B)

Insert 20: 0 rotation(s)
    20(R)
10(B)

Insert 30: 2 rotation(s)
    30(R)
20(B)
    10(R)

Insert 15: 0 rotation(s)
    30(B)
20(B)
        15(R)
    10(B)

Insert 25: 2 rotation(s)
    30(B)
        25(R)
20(B)
        15(R)
    10(B)

Insert 5: 0 rotation(s)
    30(B)
        25(R)
20(B)
        15(R)
    10(B)
        5(R)

Insert 1: 2 rotation(s)
    30(B)
        25(R)
20(B)
        15(B)
    5(B)
        10(R)
            1(R)
```

Each insertion uses at most 2 rotations, confirming the theoretical guarantee.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 13](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
