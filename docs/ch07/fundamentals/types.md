# Full, Complete, Perfect Trees

Binary trees come in several structural varieties, and the distinctions among them matter for algorithm design. A perfect binary tree guarantees logarithmic height, a complete binary tree can be stored efficiently in an array, and a full binary tree constrains the relationship between internal nodes and leaves. Recognizing which type of tree an algorithm requires (or produces) helps predict its time and space complexity.

## Full Binary Tree

A **full binary tree** (also called a **proper** or **strictly binary** tree) is a binary tree in which every node has either 0 or 2 children. No node has exactly one child.

```
     Full:              Not full:
       1                   1
      / \                 / \
     2   3               2   3
    / \                  /
   4   5                4
```

**Key property**: A full binary tree with $n$ internal nodes has exactly $n + 1$ leaves.

??? note "Proof"
    Let $L$ denote the number of leaves and $I$ the number of internal nodes. Every internal node contributes exactly 2 edges (one to each child). The total number of edges is $2I$. Since a tree with $n$ total nodes has $n - 1$ edges, and $n = I + L$, we get $2I = I + L - 1$, so $L = I + 1$.

The total number of nodes in a full binary tree is:

$$
n = 2I + 1 = 2L - 1
$$

where $I$ is the number of internal nodes and $L$ is the number of leaves.

## Complete Binary Tree

A **complete binary tree** is a binary tree in which every level is fully filled except possibly the last level, which is filled from left to right.

```
     Complete:          Not complete:
         1                  1
        / \                / \
       2   3              2   3
      / \ /              / \   \
     4  5 6             4   5   7
```

**Key properties**:

- A complete binary tree with $n$ nodes has height $h = \lfloor \log_2 n \rfloor$.
- It can be stored efficiently in an [array](array.md) with no wasted space.
- The number of nodes satisfies $2^h \leq n \leq 2^{h+1} - 1$.

Complete binary trees are the shape used by **binary heaps**. The completeness property ensures that the heap maintains $\Theta(\log n)$ height, which keeps insertion and extraction efficient.

## Perfect Binary Tree

A **perfect binary tree** is a binary tree in which all internal nodes have exactly two children and all leaves are at the same depth.

```
     Perfect (h=2):
         1
        / \
       2   3
      / \ / \
     4  5 6  7
```

A perfect binary tree is both full and complete. It has the maximum number of nodes for a given height:

$$
n = 2^{h+1} - 1
$$

where $h$ is the height of the tree. Equivalently, the height of a perfect binary tree with $n$ nodes is:

$$
h = \log_2(n + 1) - 1
$$

The number of nodes at each depth $d$ is exactly $2^d$, and the number of leaves is:

$$
L = 2^h = \frac{n + 1}{2}
$$

This means roughly half of all nodes in a perfect binary tree are leaves.

## Comparison

| Property | Full | Complete | Perfect |
|---|---|---|---|
| Every node has 0 or 2 children | Yes | Not necessarily | Yes |
| All levels filled except possibly last | Not necessarily | Yes | Yes |
| All leaves at same depth | Not necessarily | Not necessarily | Yes |
| Can be stored in array efficiently | No | Yes | Yes |
| Height guarantee | $\Theta(\log n)$ to $\Theta(n)$ | $\Theta(\log n)$ | $\Theta(\log n)$ |

!!! warning "Terminology Varies Across Textbooks"
    Some authors use "complete" to mean what we call "perfect." Others use "full" to mean "complete." This book follows the CLRS and most algorithm textbook conventions: **full** = every node has 0 or 2 children, **complete** = all levels full except possibly the last (filled left to right), **perfect** = all levels full.

## Relationship Between Types

The three types form a hierarchy:

- Every **perfect** binary tree is also **complete** and **full**.
- A **complete** binary tree is not necessarily full (the last level may have nodes with only a left child at the boundary).
- A **full** binary tree is not necessarily complete (leaves can appear at different depths).

```python
"""
Classifying binary trees as full, complete, or perfect.

Provides functions to test whether a binary tree satisfies
each structural property, with examples of each type.
"""


# === Node definition ===

class Node:
    """A node in a binary tree."""

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right


# === Classification functions ===

def is_full(node):
    """Check if the tree is a full binary tree.

    Every node has either 0 or 2 children.
    """
    if node is None:
        return True
    if node.left is None and node.right is None:
        return True
    if node.left is not None and node.right is not None:
        return is_full(node.left) and is_full(node.right)
    return False


def _count_nodes(node):
    """Return the total number of nodes."""
    if node is None:
        return 0
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)


def is_complete(node, index=0, node_count=None):
    """Check if the tree is a complete binary tree.

    Uses the array-indexing property: for a complete tree with n nodes,
    every node's index (in level-order) must be less than n.
    """
    if node_count is None:
        node_count = _count_nodes(node)
    if node is None:
        return True
    if index >= node_count:
        return False
    return (is_complete(node.left, 2 * index + 1, node_count) and
            is_complete(node.right, 2 * index + 2, node_count))


def _height(node):
    """Return the height of the tree."""
    if node is None:
        return -1
    return 1 + max(_height(node.left), _height(node.right))


def is_perfect(node, depth=0, target_depth=None):
    """Check if the tree is a perfect binary tree.

    All leaves must be at the same depth, and every internal
    node must have exactly two children.
    """
    if target_depth is None:
        target_depth = _height(node)
    if node is None:
        return True
    if node.left is None and node.right is None:
        return depth == target_depth
    if node.left is None or node.right is None:
        return False
    return (is_perfect(node.left, depth + 1, target_depth) and
            is_perfect(node.right, depth + 1, target_depth))


# === Main ===

if __name__ == "__main__":
    # Perfect tree (also full and complete)
    perfect = Node(1,
        Node(2, Node(4), Node(5)),
        Node(3, Node(6), Node(7)))

    # Full but not complete (leaves at different depths)
    full_only = Node(1,
        Node(2, Node(4), Node(5)),
        Node(3))

    # Complete but not full (node 3 has only a left child at boundary)
    complete_only = Node(1,
        Node(2, Node(4), Node(5)),
        Node(3, Node(6)))

    trees = [
        ("Perfect tree", perfect),
        ("Full-only tree", full_only),
        ("Complete-only tree", complete_only),
    ]

    for name, tree in trees:
        print(f"{name}:")
        print(f"  Full:     {is_full(tree)}")
        print(f"  Complete: {is_complete(tree)}")
        print(f"  Perfect:  {is_perfect(tree)}")
        print()
```

**Output:**
```
Perfect tree:
  Full:     True
  Complete: True
  Perfect:  True

Full-only tree:
  Full:     True
  Complete: False
  Perfect:  False

Complete-only tree:
  Full:     False
  Complete: True
  Perfect:  False
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
