# Height Bound

The practical value of a red-black tree hinges on the guarantee that its height is $O(\log n)$. The five red-black properties do not mention height explicitly, yet they constrain the tree shape tightly enough to ensure logarithmic height. This section proves the bound $h \leq 2\log_2(n+1)$ using the black-height lemma developed in the previous section.

## Statement

!!! info "Theorem: Red-Black Tree Height Bound"
    A red-black tree with $n$ internal nodes has height at most $2\log_2(n+1)$.

## Proof

The proof combines two facts:

**Fact 1.** The black-height of the root satisfies $\text{bh}(\text{root}) \geq h/2$.

By Property 4 (no two consecutive red nodes), at most half the nodes on any root-to-leaf path can be red. Since the path has $h$ edges (equivalently $h$ nodes below the root), at least $h/2$ of those nodes are black. Therefore $\text{bh}(\text{root}) \geq h/2$.

**Fact 2.** The subtree rooted at any node $x$ contains at least $2^{\text{bh}(x)} - 1$ internal nodes (the black-height lemma from the previous section).

Applying Fact 2 to the root:

$$
n \geq 2^{\text{bh}(\text{root})} - 1 \geq 2^{h/2} - 1
$$

Solving for $h$:

$$
n + 1 \geq 2^{h/2}
$$

$$
\log_2(n + 1) \geq h/2
$$

$$
h \leq 2\log_2(n + 1)
$$

$\square$

## Interpretation

The bound $h \leq 2\log_2(n+1)$ means:

- A red-black tree with $n = 10^6$ nodes has height at most $2 \times 20 = 40$.
- A perfectly balanced binary tree would have height $\approx 20$.
- An AVL tree would have height at most $\approx 29$.

Red-black trees are roughly twice as tall as perfect trees in the worst case, but this factor of 2 is a small constant that does not affect asymptotic complexity.

## Tightness of the Bound

The bound is tight up to lower-order terms. Consider a tree where every root-to-leaf path alternates between red and black nodes (starting with black at the root). Such a path of length $h = 2b$ has $b$ black nodes, and the tree achieves $h = 2 \cdot \text{bh}(\text{root})$.

However, for large $n$, the actual height of a red-black tree created by random insertions tends to be close to $\log_2 n$, well below the $2\log_2 n$ worst case.

## Comparison with AVL Trees

| Tree Type | Height Bound | Ratio to Perfect |
|:--|:--|:-:|
| Perfect binary tree | $\lfloor \log_2 n \rfloor$ | 1.0 |
| AVL tree | $1.44 \log_2 n$ | 1.44 |
| Red-black tree | $2 \log_2(n+1)$ | 2.0 |

The AVL tree provides a tighter height bound, which means faster lookups. The red-black tree compensates with simpler rebalancing (fewer rotations per modification). For applications dominated by lookups, AVL trees may be preferable. For applications with frequent insertions and deletions, red-black trees typically perform better in practice.

## Verification

```python
"""
Verify the red-black tree height bound h <= 2 * log2(n + 1).

Builds red-black trees of various sizes and checks that
the actual height never exceeds the theoretical bound.
"""

import math
import random


# === Constants ===

RED = "R"
BLACK = "B"


# === Red-Black Tree Implementation ===

class RBNode:
    """Red-black tree node."""

    def __init__(self, key, color=RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None


# Sentinel
NIL = RBNode(key=None, color=BLACK)


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


def rb_insert(tree, key):
    """Insert key into RB tree and fix up."""
    z = RBNode(key, RED)
    z.left = NIL
    z.right = NIL

    y_node = None
    x_node = tree["root"]

    while x_node is not NIL:
        y_node = x_node
        if z.key < x_node.key:
            x_node = x_node.left
        else:
            x_node = x_node.right

    z.parent = y_node
    if y_node is None:
        tree["root"] = z
    elif z.key < y_node.key:
        y_node.left = z
    else:
        y_node.right = z

    rb_insert_fixup(tree, z)


def rb_insert_fixup(tree, z):
    """Fix red-black violations after insertion."""
    while z.parent is not None and z.parent.color == RED:
        if z.parent is z.parent.parent.left:
            uncle = z.parent.parent.right
            if uncle.color == RED:
                z.parent.color = BLACK
                uncle.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            else:
                if z is z.parent.right:
                    z = z.parent
                    left_rotate(tree, z)
                z.parent.color = BLACK
                z.parent.parent.color = RED
                right_rotate(tree, z.parent.parent)
        else:
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
                z.parent.color = BLACK
                z.parent.parent.color = RED
                left_rotate(tree, z.parent.parent)
    tree["root"].color = BLACK


# === Measurement ===

def tree_height(node):
    """Compute height of tree."""
    if node is NIL:
        return -1
    return 1 + max(tree_height(node.left), tree_height(node.right))


def count_nodes(node):
    """Count internal nodes."""
    if node is NIL:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)


if __name__ == "__main__":
    print(f"{'n':>8} | {'h':>4} | {'bound':>8} | {'ok':>4}")
    print("-" * 32)

    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        tree = {"root": NIL}
        keys = list(range(n))
        random.seed(42)
        random.shuffle(keys)
        for k in keys:
            rb_insert(tree, k)

        h = tree_height(tree["root"])
        bound = 2 * math.log2(n + 1)
        ok = h <= bound
        print(f"{n:8d} | {h:4d} | {bound:8.2f} | {'yes' if ok else 'NO':>4}")
```

**Output:**
```
       n |    h |    bound |   ok
--------------------------------
      10 |    4 |     6.91 |  yes
      50 |    8 |    11.33 |  yes
     100 |   10 |    13.32 |  yes
     500 |   14 |    17.93 |  yes
    1000 |   16 |    19.93 |  yes
    5000 |   20 |    24.58 |  yes
   10000 |   22 |    26.58 |  yes
```

The actual heights are well below the theoretical bound in all cases.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 13](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
