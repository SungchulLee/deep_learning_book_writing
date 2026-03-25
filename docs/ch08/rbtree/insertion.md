# Insertion

Inserting into a red-black tree begins exactly like a standard BST insertion: walk down from the root and attach the new node as a leaf. The new node is always colored **red**, because adding a red node does not change any path's black-height (Property 5 is preserved). However, if the new node's parent is also red, Property 4 (no two consecutive red nodes) is violated. The **insertion fixup** procedure, covered in the next section, restores all properties through recoloring and at most two rotations.

## Why Color the New Node Red

Coloring the new node black would increase the black-height of every path passing through it by 1, violating Property 5 for all ancestors. Repairing Property 5 is expensive because it affects global path counts.

Coloring the new node red preserves Property 5 but may violate Property 4 (if the parent is red). Repairing Property 4 is local: it only requires fixing a single red-red conflict, which can be resolved by recoloring and/or rotations that stay on the insertion path.

## Insertion Procedure

The insertion procedure follows the CLRS formulation:

**Step 1.** Perform standard BST insertion. Walk from the root, comparing the new key with each node. When a NIL sentinel is reached, replace it with the new node.

**Step 2.** Color the new node red.

**Step 3.** Set the new node's children to NIL sentinels.

**Step 4.** Call `INSERT-FIXUP` to restore the red-black properties.

### Pseudocode

```
RB-INSERT(T, z):
    y = T.nil
    x = T.root
    while x != T.nil:
        y = x
        if z.key < x.key:
            x = x.left
        else:
            x = x.right
    z.parent = y
    if y == T.nil:
        T.root = z
    elif z.key < y.key:
        y.left = z
    else:
        y.right = z
    z.left = T.nil
    z.right = T.nil
    z.color = RED
    RB-INSERT-FIXUP(T, z)
```

## What Can Go Wrong

After inserting a red node $z$:

- **Property 1** (every node is red or black): satisfied, $z$ is red.
- **Property 2** (root is black): violated only if $z$ is the root (the tree was empty). Fix: recolor the root to black.
- **Property 3** (leaves are black): satisfied, $z$'s children are NIL sentinels (black).
- **Property 4** (no red-red): violated if $z$'s parent is red.
- **Property 5** (uniform black-height): satisfied, because $z$ is red and replaces a black NIL with a red node having two black NILs.

So the only possible violations are Property 2 (trivial fix) and Property 4 (handled by fixup).

## Implementation

```python
"""
Red-black tree insertion (without fixup, which is in the next section).

Demonstrates the BST insertion phase and the initial red coloring
that sets up the need for fixup.
"""


# === Constants ===

RED = "R"
BLACK = "B"


# === Red-Black Node ===

class RBNode:
    """Red-black tree node with key, color, children, and parent."""

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
NIL.parent = NIL


# === BST Insertion Phase ===

def bst_insert(tree, z):
    """Insert node z into tree using standard BST insertion.

    Colors z red and sets children to NIL.
    Does NOT fix up red-black violations.
    """
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

    z.left = NIL
    z.right = NIL
    z.color = RED


# === Display ===

def print_tree(node, level=0):
    """Print tree sideways with colors."""
    if node is NIL:
        return
    print_tree(node.right, level + 1)
    indent = "    " * level
    print(f"{indent}{node.key}({node.color})")
    print_tree(node.left, level + 1)


def check_property_4(node):
    """Check if Property 4 is violated anywhere."""
    if node is NIL:
        return True
    if node.color == RED:
        if node.left.color == RED or node.right.color == RED:
            print(f"  Property 4 VIOLATED: {node} has red child")
            return False
    return check_property_4(node.left) and check_property_4(node.right)


if __name__ == "__main__":
    # Insert nodes and show the red-red violation before fixup
    tree = {"root": NIL}

    # Insert root (will be red, needs recolor to black)
    node10 = RBNode(10)
    bst_insert(tree, node10)
    tree["root"].color = BLACK  # Fix Property 2
    print("After inserting 10 (root, colored black):")
    print_tree(tree["root"])
    print()

    # Insert 5 (red, parent is black -> no violation)
    node5 = RBNode(5)
    bst_insert(tree, node5)
    print("After inserting 5 (red, parent black -> OK):")
    print_tree(tree["root"])
    check_property_4(tree["root"])
    print()

    # Insert 3 (red, parent 5 is red -> VIOLATION)
    node3 = RBNode(3)
    bst_insert(tree, node3)
    print("After inserting 3 (red, parent red -> VIOLATION):")
    print_tree(tree["root"])
    check_property_4(tree["root"])
    print("  -> INSERT-FIXUP needed to resolve this violation")
```

**Output:**
```
After inserting 10 (root, colored black):
10(B)

After inserting 5 (red, parent black -> OK):
10(B)
    5(R)

After inserting 3 (red, parent red -> VIOLATION):
10(B)
    5(R)
        3(R)
  Property 4 VIOLATED: 5(R) has red child
  -> INSERT-FIXUP needed to resolve this violation
```

The violation at node 5 (a red node with a red child) is exactly the situation that `INSERT-FIXUP` resolves, as detailed in the next section.

## Complexity

The BST insertion phase takes $O(\log n)$ time (walking down the tree). The fixup (next section) also takes $O(\log n)$ time with at most 2 rotations. Therefore, the total insertion time is $O(\log n)$.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 13](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
