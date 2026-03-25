# Black-Height

The red-black properties constrain how colors are distributed along root-to-leaf paths. Property 5 demands that every path from a node to a descendant leaf contains the same number of black nodes. This count --- the **black-height** --- is the central quantity that connects coloring rules to the logarithmic height guarantee. Understanding black-height is essential before proving the height bound of red-black trees.

## Definition

The **black-height** of a node $x$, denoted $\text{bh}(x)$, is the number of black nodes on any simple path from $x$ down to a leaf (NIL sentinel), **not counting $x$ itself**.

By Property 5, this count is the same regardless of which descendant leaf path is chosen, so $\text{bh}(x)$ is well-defined.

For the NIL sentinel nodes (leaves), the black-height is 0:

$$
\text{bh}(\text{NIL}) = 0
$$

## Computing Black-Height

For an internal node $x$ with children $l$ and $r$:

$$
\text{bh}(x) = \begin{cases} \text{bh}(l) & \text{if } l \text{ is black, then } \text{bh}(l) = \text{bh}(r) \\ \text{bh}(l) + 1 & \text{wait — let us be more careful} \end{cases}
$$

Actually, the relationship depends on the children's colors:

- If child $c$ is **black**, then $\text{bh}(x) = \text{bh}(c) + 1$ (the path from $x$ through $c$ picks up one more black node at $c$).
- If child $c$ is **red**, then $\text{bh}(x) = \text{bh}(c)$ (red nodes do not count).

Since Property 5 guarantees both children yield the same $\text{bh}(x)$, we can use either child:

$$
\text{bh}(x) = \begin{cases} \text{bh}(\text{child}) + 1 & \text{if child is black} \\ \text{bh}(\text{child}) & \text{if child is red} \end{cases}
$$

## Key Lemma

!!! info "Lemma: Minimum subtree size"
    A subtree rooted at node $x$ contains at least $2^{\text{bh}(x)} - 1$ internal nodes.

**Proof by induction on the height of $x$.**

*Base case*: If $x$ is a leaf (NIL), then $\text{bh}(x) = 0$ and the subtree has $0 = 2^0 - 1$ internal nodes.

*Inductive step*: Let $x$ be an internal node with children $l$ and $r$. Each child has black-height at least $\text{bh}(x) - 1$ (exactly $\text{bh}(x) - 1$ if the child is black, or $\text{bh}(x)$ if the child is red). By the inductive hypothesis, each child's subtree has at least $2^{\text{bh}(x)-1} - 1$ internal nodes. Therefore:

$$
n(x) \geq 1 + 2\bigl(2^{\text{bh}(x)-1} - 1\bigr) = 2^{\text{bh}(x)} - 1
$$

$\square$

## Example

Consider a red-black tree (B = black, R = red):

```
          10(B)          bh = 2
         /     \
       5(R)    15(B)     bh(5)=2, bh(15)=1
      /   \    /   \
    3(B) 7(B) 13(R) 20(B)   bh(3)=1, bh(7)=1, bh(13)=1, bh(20)=0
   / \  / \   / \   / \
  N  N N  N 11(B) N  N  N
             / \
            N   N
```

- NIL nodes: $\text{bh} = 0$
- Node 20 (black, leaf-like): $\text{bh}(20) = 0$ (paths go directly to NIL)
- Node 3 (black): $\text{bh}(3) = 0 + 1 = 1$ (counting the NIL below, which is black... wait, NIL is black but $\text{bh}$ does not count NIL). Let us reclarify.

Using the convention that $\text{bh}(x)$ counts the black nodes **below** $x$ (not including $x$ or NIL):

- Node 20 (black, children are NIL): $\text{bh}(20) = 0$
- Node 3 (black, children are NIL): $\text{bh}(3) = 0$
- Node 7 (black, children are NIL): $\text{bh}(7) = 0$
- Node 11 (black, children are NIL): $\text{bh}(11) = 0$
- Node 13 (red, children: 11(B) and NIL): $\text{bh}(13) = 0 + 1 = 1$
- Node 15 (black, children: 13(R) and 20(B)): $\text{bh}(15) = \text{bh}(13) = 1$ (13 is red, so no extra count), and checking: $\text{bh}(20) + 1 = 1$ (20 is black). Both give 1. Check.
- Node 5 (red, children: 3(B) and 7(B)): $\text{bh}(5) = \text{bh}(3) + 1 = 1$
- Node 10 (black, children: 5(R) and 15(B)): $\text{bh}(10) = \text{bh}(5) = 1$ (5 is red), and $\text{bh}(15) + 1 = 2$...

This discrepancy reveals a problem with our example tree --- it does not satisfy Property 5. Let us use a correct example instead.

**Corrected example:**

```
          10(B)          bh = 2
         /     \
       5(B)    15(B)     bh = 1
      /   \    /   \
    3(R) 7(R) 13(R) 20(R)   bh = 1
```

- Nodes 3, 7, 13, 20 (red, children are NIL): $\text{bh} = 0 + 1 = 1$. Wait, NIL is black, so the path from a red node to its NIL child has 1 black node (the NIL). But by CLRS convention, $\text{bh}$ counts black nodes on the path **not including** $x$ itself but **including** NIL. Under this convention: $\text{bh}(\text{NIL}) = 0$, and $\text{bh}(3) = 0 + 1 = 1$ if NIL is counted...

Let us adopt the **CLRS convention** precisely: $\text{bh}(x)$ = number of black nodes on any path from $x$ to a leaf, **not counting $x$**. NIL sentinels are leaves with $\text{bh} = 0$.

- Node 3 (red, children are NIL(B)): path from 3 to leaf = {NIL}, black count = 1. So $\text{bh}(3) = 1$.
- Node 5 (black, left child 3(R)): path from 5 to leaf through 3 = {3, NIL}. Black count = 1 (only NIL). So $\text{bh}(5) = 1$.
- Node 10 (black, left child 5(B)): path through 5 then 3 then NIL = {5, 3, NIL}. Black count = 2 (5 and NIL). So $\text{bh}(10) = 2$.

Checking the lemma: the subtree at node 10 has 7 internal nodes, and $2^{\text{bh}(10)} - 1 = 2^2 - 1 = 3$. Indeed $7 \geq 3$.

## Implementation

```python
"""
Black-height computation for red-black trees.

Demonstrates the black-height definition and verifies
that all paths from a node to leaves have equal black-node counts.
"""


# === Constants ===

RED = "R"
BLACK = "B"


# === Red-Black Node ===

class RBNode:
    """A red-black tree node."""

    def __init__(self, key, color=RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None

    def __repr__(self):
        return f"{self.key}({self.color})"


# Sentinel NIL node
NIL = RBNode(key=None, color=BLACK)
NIL.left = NIL
NIL.right = NIL


# === Black-Height Computation ===

def black_height(node):
    """Compute black-height of a node (CLRS convention).

    Returns the number of black nodes on any path from node
    to a descendant leaf, not counting node itself.
    Returns -1 if the tree violates Property 5.
    """
    if node is NIL:
        return 0

    left_bh = black_height(node.left)
    right_bh = black_height(node.right)

    if left_bh == -1 or right_bh == -1:
        return -1  # violation in subtree

    # Adjust for child color
    left_count = left_bh + (1 if node.left.color == BLACK else 0)
    right_count = right_bh + (1 if node.right.color == BLACK else 0)

    if left_count != right_count:
        print(f"  Property 5 violation at {node}: "
              f"left bh={left_count}, right bh={right_count}")
        return -1

    return left_count


# === Tree Builder (manual) ===

def build_example_tree():
    """Build a valid red-black tree for demonstration."""
    root = RBNode(10, BLACK)
    root.left = RBNode(5, BLACK)
    root.right = RBNode(15, BLACK)
    root.left.left = RBNode(3, RED)
    root.left.right = RBNode(7, RED)
    root.right.left = RBNode(13, RED)
    root.right.right = RBNode(20, RED)

    # Set NIL children
    for node in [root.left.left, root.left.right,
                 root.right.left, root.right.right]:
        node.left = NIL
        node.right = NIL
    root.left.left.left = NIL
    root.left.left.right = NIL

    return root


# === Display ===

def print_tree(node, level=0):
    """Print tree sideways with colors and black-heights."""
    if node is NIL:
        return
    print_tree(node.right, level + 1)
    bh = black_height(node)
    indent = "    " * level
    print(f"{indent}{node.key}({node.color}) bh={bh}")
    print_tree(node.left, level + 1)


if __name__ == "__main__":
    root = build_example_tree()
    print("Red-Black Tree with black-heights:")
    print_tree(root)
    print()
    print(f"Root black-height: {black_height(root)}")
    n = 7  # internal nodes
    bh = black_height(root)
    print(f"Internal nodes: {n}")
    print(f"Lemma check: 2^bh - 1 = {2**bh - 1} <= {n}: {2**bh - 1 <= n}")
```

**Output:**
```
Red-Black Tree with black-heights:
        20(R) bh=1
    15(B) bh=1
        13(R) bh=1
10(B) bh=2
        7(R) bh=1
    5(B) bh=1
        3(R) bh=1

Root black-height: 2
Internal nodes: 7
Lemma check: 2^bh - 1 = 3 <= 7: True
```

## Significance

Black-height serves two critical roles:

1. **Height bound proof**: The lemma $n \geq 2^{\text{bh}(x)} - 1$ combined with $\text{bh}(\text{root}) \geq h/2$ yields $h \leq 2\log_2(n+1)$. This proof is developed fully in the Height Bound section.

2. **Algorithm correctness**: During insertion and deletion fixup, the algorithms maintain black-height invariants. Every case analysis in the fixup procedures checks that black-heights remain consistent after recoloring and rotations.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 13](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
