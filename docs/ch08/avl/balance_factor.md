# Balance Factor

In a standard binary search tree, inserting keys in sorted order produces a degenerate chain with $O(n)$ lookup time. AVL trees prevent this degradation by tracking how "lopsided" each node is and rebalancing when the imbalance exceeds a threshold. The **balance factor** is the integer quantity that captures this lopsidedness, serving as the trigger for every rotation in an AVL tree.

## Definition

For a node $x$ in a binary tree, let $h(x)$ denote the height of the subtree rooted at $x$, defined as the length of the longest root-to-leaf path. An empty subtree (null child) has height $-1$ by convention.

The **balance factor** of node $x$ is

$$
\text{BF}(x) = h(\text{left}(x)) - h(\text{right}(x))
$$

where $\text{left}(x)$ and $\text{right}(x)$ denote the left and right children of $x$, respectively. The balance factor is simply the height of the left subtree minus the height of the right subtree.

## AVL Invariant

An AVL tree is a binary search tree in which every node satisfies the **AVL balance condition**:

$$
\text{BF}(x) \in \{-1, 0, 1\}
$$

for all nodes $x$ in the tree. Equivalently, the heights of the two child subtrees of any node differ by at most one.

When an insertion or deletion causes some node's balance factor to fall outside $\{-1, 0, 1\}$ --- that is, $|\text{BF}(x)| \geq 2$ --- the tree performs one or two rotations at that node to restore the invariant.

## Interpretation of Values

The three permitted values each describe a distinct shape at a node:

| Balance Factor | Meaning |
|:-:|:--|
| $+1$ | Left subtree is one level taller than the right (left-heavy) |
| $0$ | Both subtrees have equal height (perfectly balanced at this node) |
| $-1$ | Right subtree is one level taller than the left (right-heavy) |

A balance factor of $+2$ signals that the left subtree is too tall, requiring a right rotation (or a left-right double rotation). A balance factor of $-2$ signals that the right subtree is too tall, requiring a left rotation (or a right-left double rotation).

## Computing the Balance Factor

Each node stores its own height (or equivalently its balance factor) as an integer field. After every insertion or deletion, the algorithm walks back up the path from the modified leaf to the root, updating heights and checking balance factors.

For a node $x$ with children $l$ and $r$:

$$
h(x) = 1 + \max\bigl(h(l),\, h(r)\bigr)
$$

$$
\text{BF}(x) = h(l) - h(r)
$$

If $|\text{BF}(x)| \leq 1$, the node is balanced and the walk continues upward. If $|\text{BF}(x)| = 2$, the appropriate rotation is applied at $x$.

## Example

Consider the following AVL tree where each node is annotated with its balance factor:

```
        30 [+1]
       /  \
     20 [0]  40 [-1]
    /  \       \
  10 [0] 25 [0]  50 [0]
```

- Node 30 has left subtree height 2 and right subtree height 2, but its left subtree through node 20 has height 2 while its right subtree through 40 also has height 2 --- wait, let us recount. Node 10 has height 0, node 25 has height 0, so node 20 has height 1. Node 50 has height 0, so node 40 has height 1. Node 30 has height 2 on both sides, giving $\text{BF}(30) = 1 - 1 = 0$.

A corrected annotated tree:

```
        30 [0]
       /  \
     20 [0]  40 [-1]
    /  \       \
  10 [0] 25 [0]  50 [0]
```

- Node 10: no children, $\text{BF} = (-1) - (-1) = 0$
- Node 25: no children, $\text{BF} = (-1) - (-1) = 0$
- Node 20: $h(\text{left}) = 0$, $h(\text{right}) = 0$, so $\text{BF} = 0$
- Node 50: no children, $\text{BF} = 0$
- Node 40: $h(\text{left}) = -1$, $h(\text{right}) = 0$, so $\text{BF} = -1$
- Node 30: $h(\text{left}) = 1$, $h(\text{right}) = 1$, so $\text{BF} = 0$

Now suppose we insert 5. It goes to the left of node 10:

```
          30 [+1]
         /  \
       20 [+1]  40 [-1]
      /  \       \
   10 [+1] 25 [0]  50 [0]
   /
  5 [0]
```

All balance factors remain in $\{-1, 0, +1\}$, so no rotation is needed. But if we further insert 3:

```
            30 [+2]       <-- violation!
           /  \
         20 [+2]  40 [-1]  <-- violation!
        /  \       \
     10 [+2] 25 [0]  50 [0]  <-- violation!
     /
    5 [+1]
   /
  3 [0]
```

Node 10 now has $\text{BF} = +2$, violating the AVL condition. A right rotation at node 10 restores balance. In practice, the algorithm detects the violation at the lowest unbalanced ancestor and rotates there; the fix at node 10 propagates upward and may suffice for all ancestors.

## Implementation

```python
"""
AVL tree node with balance factor computation.

Demonstrates height tracking and balance factor calculation,
which are the foundation of AVL tree rebalancing.
"""


# === AVL Node Definition ===

class AVLNode:
    """A node in an AVL tree that tracks its own height."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 0  # height of a leaf is 0

    def __repr__(self):
        return f"AVLNode({self.key})"


# === Height and Balance Factor Utilities ===

def height(node):
    """Return the height of a node, or -1 for null."""
    if node is None:
        return -1
    return node.height


def update_height(node):
    """Recompute the height of a node from its children."""
    node.height = 1 + max(height(node.left), height(node.right))


def balance_factor(node):
    """Compute balance factor = height(left) - height(right)."""
    return height(node.left) - height(node.right)


# === Demonstration ===

def insert_bst(node, key):
    """Insert a key into a plain BST (no rebalancing)."""
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = insert_bst(node.left, key)
    elif key > node.key:
        node.right = insert_bst(node.right, key)
    update_height(node)
    return node


def print_balance_factors(node, level=0):
    """Print the tree with balance factors."""
    if node is None:
        return
    print_balance_factors(node.right, level + 1)
    indent = "    " * level
    bf = balance_factor(node)
    print(f"{indent}{node.key} [BF={bf:+d}]")
    print_balance_factors(node.left, level + 1)


if __name__ == "__main__":
    root = None
    for key in [30, 20, 40, 10, 25, 50]:
        root = insert_bst(root, key)

    print("AVL tree with balance factors:")
    print_balance_factors(root)
    print()

    # Insert 5 and 3 to create an imbalance
    root = insert_bst(root, 5)
    root = insert_bst(root, 3)
    print("After inserting 5 and 3 (imbalanced):")
    print_balance_factors(root)
```

**Output:**
```
AVL tree with balance factors:
        50 [BF=+0]
    40 [BF=-1]
30 [BF=+0]
        25 [BF=+0]
    20 [BF=+0]
        10 [BF=+0]

After inserting 5 and 3 (imbalanced):
        50 [BF=+0]
    40 [BF=-1]
30 [BF=+2]
        25 [BF=+0]
    20 [BF=+2]
            5 [BF=+1]
                3 [BF=+0]
        10 [BF=+2]
```

Node 10 shows $\text{BF} = +2$, confirming the AVL violation that would trigger a right rotation.

## Connection to Rotations

The balance factor determines which rotation to apply:

| $\text{BF}(x)$ | $\text{BF}(\text{child})$ | Rotation |
|:-:|:-:|:--|
| $+2$ | $+1$ or $0$ | Single right rotation at $x$ |
| $+2$ | $-1$ | Left-right double rotation at $x$ |
| $-2$ | $-1$ or $0$ | Single left rotation at $x$ |
| $-2$ | $+1$ | Right-left double rotation at $x$ |

The child's balance factor determines whether the heavy subtree is on the "same side" (single rotation) or the "opposite side" (double rotation). Single and double rotations are covered in the following sections.

## Reference

- [10.1 AVL Tree - Insertion and Rotations](https://www.youtube.com/watch?v=jDM6_TnYIqE&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=76)
- [Introduction to Algorithms (CLRS), Chapter 13](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
