# Height and Depth

Understanding the height of a tree and the depth of individual nodes is essential for analyzing the time complexity of tree operations. Most search, insertion, and deletion algorithms on binary trees run in time proportional to the height, so knowing how to compute and bound the height directly determines whether an algorithm is efficient or degenerate.

## Depth of a Node

The **depth** of a node $v$ is the number of edges on the path from the root to $v$. Equivalently, depth counts how many ancestors $v$ has (excluding itself).

$$
\text{depth}(v) = \begin{cases} 0 & \text{if } v \text{ is the root} \\ 1 + \text{depth}(\text{parent}(v)) & \text{otherwise} \end{cases}
$$

The root always has depth 0. A child of the root has depth 1, a grandchild has depth 2, and so on.

## Height of a Node

The **height** of a node $v$ is the number of edges on the longest downward path from $v$ to a leaf. This captures how far below the deepest descendant lies.

$$
\text{height}(v) = \begin{cases} 0 & \text{if } v \text{ is a leaf} \\ 1 + \max(\text{height}(\text{left}(v)),\; \text{height}(\text{right}(v))) & \text{otherwise} \end{cases}
$$

For the recursive definition to work cleanly with empty subtrees, we adopt the convention that the height of a null (empty) subtree is $-1$:

$$
\text{height}(\text{null}) = -1
$$

This way, a leaf node with two null children has height $1 + \max(-1, -1) = 0$, which is consistent.

## Height of a Tree

The **height of a tree** is the height of its root node, or equivalently, the maximum depth of any node in the tree:

$$
h(T) = \text{height}(\text{root}) = \max_{v \in T} \text{depth}(v)
$$

## Height Bounds

For a binary tree with $n \geq 1$ nodes and height $h$:

- **Minimum height** (complete binary tree): $h = \lfloor \log_2 n \rfloor$
- **Maximum height** (degenerate/skewed tree): $h = n - 1$

This gives the bound:

$$
\lfloor \log_2 n \rfloor \leq h \leq n - 1
$$

A balanced binary tree keeps $h = \Theta(\log n)$, ensuring efficient operations. A degenerate tree where every internal node has exactly one child degrades to a linked list with $h = n - 1$, making operations $\Theta(n)$.

!!! note "Edge Count vs Node Count Convention"
    Some references define height as the number of **nodes** on the longest root-to-leaf path rather than the number of **edges**. Under that convention, height = (edge-based height) + 1, and a single-node tree has height 1 instead of 0. This book follows the edge-counting convention used by CLRS.

## Example

Consider the following binary tree:

```
         A          depth 0, height 3
        / \
       B   C        depth 1, heights 1 and 2
      /   / \
     D   E   F      depth 2, heights 0, 1, and 0
            /
           G        depth 3, height 0
```

| Node | Depth | Height |
|------|-------|--------|
| A    | 0     | 3      |
| B    | 1     | 1      |
| C    | 1     | 2      |
| D    | 2     | 0      |
| E    | 2     | 1      |
| F    | 2     | 0      |
| G    | 3     | 0      |

The height of the tree is $\text{height}(A) = 3$, and the maximum depth is also 3 (node G).

```python
"""
Height and depth computation for binary trees.

Demonstrates recursive computation of node depth, node height,
and tree height, along with the relationship between these measures.
"""


# === Node definition ===

class Node:
    """A node in a binary tree."""

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right


# === Depth computation ===

def depth(root, target, current_depth=0):
    """Return the depth of the target node, or -1 if not found."""
    if root is None:
        return -1
    if root.key == target:
        return current_depth
    left_result = depth(root.left, target, current_depth + 1)
    if left_result != -1:
        return left_result
    return depth(root.right, target, current_depth + 1)


# === Height computation ===

def height(node):
    """Return the height of the subtree rooted at node.

    Uses the convention that height(null) = -1, so a leaf has height 0.
    """
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))


# === Main ===

if __name__ == "__main__":
    # Build the example tree:
    #          A
    #         / \
    #        B   C
    #       /   / \
    #      D   E   F
    #             /
    #            G
    tree = Node("A",
        Node("B", Node("D")),
        Node("C",
            Node("E", None, Node("G")),
            Node("F")))

    print("Tree height:", height(tree))
    print()

    for label in ["A", "B", "C", "D", "E", "F", "G"]:
        d = depth(tree, label)
        print(f"  Node {label}: depth = {d}")
```

**Output:**
```
Tree height: 3

  Node A: depth = 0
  Node B: depth = 1
  Node C: depth = 1
  Node D: depth = 2
  Node E: depth = 2
  Node F: depth = 2
  Node G: depth = 3
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
