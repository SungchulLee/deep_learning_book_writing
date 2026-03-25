# Red-Black Properties

AVL trees enforce balance by explicitly tracking height differences at every node. Red-black trees take a different approach: they assign a **color** (red or black) to each node and enforce a small set of coloring rules. These rules do not directly mention height, yet they guarantee that no root-to-leaf path is more than twice as long as any other, bounding the tree height to $O(\log n)$. The elegance of this approach is that maintaining the coloring rules during insertion and deletion requires fewer rotations on average than AVL trees.

## The Five Properties

A red-black tree is a binary search tree where every node carries a color bit --- red or black --- and the following five properties hold:

!!! info "Red-Black Tree Properties"
    1. **Node color**: Every node is either red or black.
    2. **Root property**: The root is black.
    3. **Leaf property**: Every leaf (NIL sentinel) is black.
    4. **Red property**: If a node is red, then both its children are black. (Equivalently, no two red nodes are adjacent on any path.)
    5. **Black-height property**: For each node, all simple paths from that node to descendant leaves contain the same number of black nodes.

Property 3 uses **NIL sentinels** --- external null nodes treated as black leaves. Every internal node has exactly two children (possibly NIL), which simplifies the statements and algorithms. In implementations, a single sentinel node `T.nil` is often shared across the tree.

## Understanding Each Property

### Property 1: Binary coloring

Each node stores one extra bit. This is the only additional space cost compared to a standard BST.

### Property 2: Black root

The root must be black. If an insertion or rotation produces a red root, simply recolor it to black. Changing the root to black adds one black node to every root-to-leaf path, preserving Property 5.

### Property 3: Black leaves (NIL sentinels)

External null pointers are modeled as black leaf nodes. This convention ensures that every internal node has exactly two children, eliminating null-pointer edge cases in the algorithms.

### Property 4: No two consecutive red nodes

A red node must have black children. Equivalently, on any path from root to leaf, red nodes cannot be adjacent. This limits how many red nodes can appear on a path: between any two red nodes there must be at least one black node.

### Property 5: Uniform black-height

Every path from a given node down to any leaf passes through the same number of black nodes. This number is the **black-height** of the node, denoted $\text{bh}(x)$.

## Why These Properties Guarantee Balance

The key insight comes from combining Properties 4 and 5:

- Property 5 ensures that every root-to-leaf path has the same number of black nodes, say $b$.
- Property 4 ensures that red nodes cannot be consecutive, so between consecutive black nodes there is at most one red node.
- Therefore, the shortest possible root-to-leaf path has $b$ nodes (all black), and the longest has at most $2b$ nodes (alternating red and black).

This gives the **longest-path/shortest-path ratio** of at most 2, which implies:

$$
h \leq 2 \cdot \text{bh}(\text{root})
$$

Combined with the bound $\text{bh}(\text{root}) \leq \log_2(n+1)$ (proved in the Height Bound section), this yields $h \leq 2\log_2(n+1)$, confirming $O(\log n)$ height.

## Example

Consider this red-black tree (R = red, B = black):

```
            8(B)
           /    \
         4(R)   12(R)
        /  \    /   \
      2(B) 6(B) 10(B) 14(B)
     / \  / \  / \   / \
    1  3  5  7 9 11 13  15
   (B)(B)(B)(B)(B)(B)(B)(B)
```

Verification of properties:

- **Property 1**: Every node is colored (check).
- **Property 2**: Root (8) is black (check).
- **Property 3**: All NIL children of leaves are black (implicit).
- **Property 4**: Red nodes 4 and 12 have only black children (check).
- **Property 5**: Every path from root to a NIL leaf passes through exactly 3 black nodes (check). For example: 8(B) -> 4(R) -> 2(B) -> 1(B) -> NIL has black nodes {8, 2, 1} = 3.

The black-height of the root is 3 (counting from root to leaf, including the leaf, but excluding the root by some conventions --- we adopt the convention that $\text{bh}(x)$ counts black nodes on any path from $x$ down to a leaf, not including $x$ itself). Under this convention, $\text{bh}(8) = 2$, and $h = 3 \leq 2 \cdot 2 + 1$.

## Comparison with AVL Trees

| Property | AVL Tree | Red-Black Tree |
|:--|:--|:--|
| Balance criterion | Height difference $\leq 1$ | Coloring rules (5 properties) |
| Height bound | $h \leq 1.44 \log_2 n$ | $h \leq 2 \log_2(n+1)$ |
| Extra storage per node | Height (integer) | Color (1 bit) |
| Rotations per insertion | $\leq 2$ | $\leq 2$ |
| Rotations per deletion | $O(\log n)$ | $\leq 3$ |
| Lookup speed | Slightly faster (shorter) | Slightly slower (taller) |
| Insert/delete speed | Slightly slower (more rotations) | Slightly faster (fewer rotations) |

Red-black trees allow taller trees (up to $2\log_2 n$ vs $1.44\log_2 n$) but compensate with fewer rotations during modifications. This trade-off makes red-black trees the preferred choice for language standard libraries (e.g., C++ `std::map`, Java `TreeMap`).

## Reference

- [Introduction to Algorithms (CLRS), Chapter 13](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
