# Linked Representation

The linked representation is the most common way to implement binary trees in practice. Unlike the [array representation](array.md), which requires the tree to be complete, linked nodes can represent any tree shape -- balanced, skewed, or anything in between. Each node is a separate object in memory connected to its children (and optionally its parent) through references. This flexibility makes the linked representation the default choice for binary search trees, expression trees, and most tree algorithms covered in this book.

## Node Structure

A binary tree node in the linked representation stores three fields at minimum:

- **key** (or data): the value held at this node
- **left**: a reference to the left child (or null if no left child exists)
- **right**: a reference to the right child (or null if no right child exists)

Some implementations add a fourth field:

- **parent**: a reference to the parent node (or null for the root)

The parent pointer is optional but simplifies operations such as finding the successor of a node or walking up from a leaf to the root.

!!! note "Binary vs General Trees"
    For a **binary tree**, each node has at most two children, so `left` and `right` pointers suffice. For a **general tree** where nodes may have an arbitrary number of children, one common approach is the **left-child, right-sibling** representation: `left` points to the first child, and `right` points to the next sibling. This represents any rooted tree using only two pointers per node.

## Space Analysis

Each node in the linked representation requires:

$$
\text{Space per node} = \text{size(key)} + 2 \times \text{size(pointer)}
$$

With a parent pointer, this becomes $\text{size(key)} + 3 \times \text{size(pointer)}$. For a tree with $n$ nodes, the total space is $\Theta(n)$ regardless of the tree's shape. In contrast, the array representation uses $\Theta(n)$ space only for complete trees; a skewed tree of height $n - 1$ would require an array of size $2^n - 1$, wasting exponential space.

| Tree shape | Linked space | Array space |
|---|---|---|
| Complete ($h = \lfloor \log_2 n \rfloor$) | $\Theta(n)$ | $\Theta(n)$ |
| Skewed ($h = n - 1$) | $\Theta(n)$ | $\Theta(2^n)$ |

## Building a Tree

A tree is built by creating individual nodes and linking them through their `left` and `right` fields. The tree is accessed through a reference to the root node.

```python
"""
Linked representation of a binary tree.

Demonstrates node creation, tree construction, and basic
traversal using the linked (pointer-based) representation.
"""


# === Node definition ===

class Node:
    """A node in a linked binary tree.

    Attributes:
        key: The value stored at this node.
        left: Reference to the left child (None if absent).
        right: Reference to the right child (None if absent).
        parent: Reference to the parent node (None for root).
    """

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right
        self.parent = None

    def __repr__(self):
        return f"Node({self.key})"


# === Tree construction helpers ===

def build_tree(key, left=None, right=None):
    """Create a node and set parent pointers for children."""
    node = Node(key, left, right)
    if left is not None:
        left.parent = node
    if right is not None:
        right.parent = node
    return node


def tree_size(node):
    """Return the number of nodes in the subtree rooted at node."""
    if node is None:
        return 0
    return 1 + tree_size(node.left) + tree_size(node.right)


def tree_height(node):
    """Return the height of the subtree (edge count)."""
    if node is None:
        return -1
    return 1 + max(tree_height(node.left), tree_height(node.right))


# === Display ===

def print_tree(node, level=0, prefix="Root: "):
    """Print the tree structure with indentation."""
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.key))
        if node.left is not None or node.right is not None:
            print_tree(node.left, level + 1, "L--- ")
            print_tree(node.right, level + 1, "R--- ")


# === Main ===

if __name__ == "__main__":
    # Build a sample tree:
    #        10
    #       /  \
    #      5    15
    #     / \     \
    #    3   7    20
    tree = build_tree(10,
        build_tree(5,
            build_tree(3),
            build_tree(7)),
        build_tree(15,
            None,
            build_tree(20)))

    print_tree(tree)
    print()
    print(f"Size:   {tree_size(tree)} nodes")
    print(f"Height: {tree_height(tree)} edges")
    print()

    # Demonstrate parent pointers
    node_7 = tree.left.right
    print(f"Node {node_7.key}'s parent: {node_7.parent}")
    print(f"Node {node_7.parent.key}'s parent: {node_7.parent.parent}")
```

**Output:**
```
Root: 10
    L--- 5
        L--- 3
        R--- 7
    R--- 15
        L--- None
        R--- 20

Size:   6 nodes
Height: 2 edges

Node 7's parent: Node(5)
Node 5's parent: Node(10)
```

## Left-Child Right-Sibling Representation

For general (non-binary) trees, the **left-child right-sibling** encoding stores an arbitrary-degree tree using only two pointers per node:

- **left_child**: points to the node's first (leftmost) child
- **right_sibling**: points to the node's next sibling

This transforms any rooted forest into a binary tree. Given a general tree where a node can have $k$ children, walking down `left_child` reaches the first child, and then following `right_sibling` iterates through all siblings.

```
General tree:          Left-child right-sibling:
      A                     A
    / | \                  /
   B  C  D               B --> C --> D
  / \                    /
 E   F                  E --> F
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
