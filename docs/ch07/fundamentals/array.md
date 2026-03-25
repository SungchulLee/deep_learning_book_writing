# Array Representation

When a binary tree is complete or nearly complete, storing nodes in a simple array eliminates the overhead of child pointers entirely. Instead of allocating separate node objects connected by references, we lay out the tree level by level in contiguous memory. This approach is the foundation of the binary heap data structure and offers excellent cache performance because parent-child navigation reduces to arithmetic on array indices.

## Index Formulas

The key insight behind the array representation is a bijection between tree positions and array indices. Consider a binary tree stored in an array $A$ of length $n$.

### Zero-Based Indexing

When the root occupies index $0$, the relationships for a node at index $i$ are:

$$
\text{left child}(i) = 2i + 1
$$

$$
\text{right child}(i) = 2i + 2
$$

$$
\text{parent}(i) = \left\lfloor \frac{i - 1}{2} \right\rfloor \quad \text{for } i > 0
$$

A child index is valid only when it is less than $n$, the total number of elements in the array.

### One-Based Indexing

When the root occupies index $1$ (a convention common in textbooks and heap implementations), the formulas simplify:

$$
\text{left child}(i) = 2i
$$

$$
\text{right child}(i) = 2i + 1
$$

$$
\text{parent}(i) = \left\lfloor \frac{i}{2} \right\rfloor \quad \text{for } i > 1
$$

The one-based convention makes the bit-shift interpretation transparent: multiplying by 2 is a left shift, and dividing by 2 is a right shift.

## Why It Works

The formulas above arise from a simple counting argument. At depth $d$, a complete binary tree has exactly $2^d$ nodes. The nodes at depth $d$ occupy array positions $2^d - 1$ through $2^{d+1} - 2$ (zero-based). Within each level, the left child of the $k$-th node at depth $d$ is the $(2k)$-th node at depth $d+1$. Translating this level-local offset into a global array index yields the formulas above.

!!! tip "When to Use Array Representation"
    The array layout works well when the tree is **complete** or **nearly complete**, meaning every level is fully occupied except possibly the last, which is filled from left to right. For sparse or highly unbalanced trees, the array wastes space because missing nodes still consume index slots. In such cases, the [linked representation](linked.md) is preferable.

## Space Comparison

| Representation | Space per node | Navigation cost |
|---|---|---|
| Linked (pointers) | Data + 2 pointers | Pointer dereference |
| Array | Data only | Index arithmetic |

For a complete binary tree with $n$ nodes, the array representation uses $\Theta(n)$ space with no pointer overhead, while the linked representation uses $\Theta(n)$ space plus $2n$ pointers.

## Example

Consider a complete binary tree with 7 nodes containing the values `[1, 2, 3, 4, 5, 6, 7]` stored level by level:

```
         1          depth 0, index 0
        / \
       2   3        depth 1, indices 1-2
      / \ / \
     4  5 6  7      depth 2, indices 3-6
```

The array stores these as `A = [1, 2, 3, 4, 5, 6, 7]`. To find the children of node at index 1 (value 2): left child is at $2(1)+1 = 3$ (value 4) and right child is at $2(1)+2 = 4$ (value 5). To find the parent of node at index 5 (value 6): parent is at $\lfloor(5-1)/2\rfloor = 2$ (value 3).

```python
"""
Array representation of a binary tree.

Demonstrates how a complete binary tree maps to a flat array
and how parent-child relationships reduce to index arithmetic.
"""


# === Index navigation (0-based) ===

def left_child(i: int) -> int:
    """Return the index of the left child of node i."""
    return 2 * i + 1


def right_child(i: int) -> int:
    """Return the index of the right child of node i."""
    return 2 * i + 2


def parent(i: int) -> int:
    """Return the index of the parent of node i (undefined for root)."""
    return (i - 1) // 2


# === Tree operations on the array ===

def get_children(tree: list, i: int) -> list:
    """Return the values of existing children of node i."""
    children = []
    l, r = left_child(i), right_child(i)
    if l < len(tree):
        children.append(tree[l])
    if r < len(tree):
        children.append(tree[r])
    return children


def print_tree_levels(tree: list) -> None:
    """Print the tree level by level."""
    if not tree:
        print("Empty tree")
        return
    level = 0
    i = 0
    while i < len(tree):
        level_size = 2 ** level
        level_nodes = tree[i : i + level_size]
        print(f"  Depth {level}: {level_nodes}")
        i += level_size
        level += 1


# === Main ===

if __name__ == "__main__":
    tree = [1, 2, 3, 4, 5, 6, 7]

    print("Array representation:", tree)
    print()
    print_tree_levels(tree)
    print()

    for idx in range(len(tree)):
        children = get_children(tree, idx)
        p = parent(idx) if idx > 0 else None
        print(
            f"  Node {tree[idx]} (index {idx}): "
            f"parent={'root' if p is None else tree[p]}, "
            f"children={children}"
        )
```

**Output:**
```
Array representation: [1, 2, 3, 4, 5, 6, 7]

  Depth 0: [1]
  Depth 1: [2, 3]
  Depth 2: [4, 5, 6, 7]

  Node 1 (index 0): parent=root, children=[2, 3]
  Node 2 (index 1): parent=1, children=[4, 5]
  Node 3 (index 2): parent=1, children=[6, 7]
  Node 4 (index 3): parent=2, children=[]
  Node 5 (index 4): parent=2, children=[]
  Node 6 (index 5): parent=3, children=[]
  Node 7 (index 6): parent=3, children=[]
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 6 - Heapsort](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
