# Tree Terminology

Before studying tree algorithms, we need a precise vocabulary. Trees appear throughout computer science -- in file systems, parsing, decision-making, and search -- and every tree algorithm description relies on the terms defined here. This page establishes the core terminology that the rest of the chapter builds upon.

## Recursive Definition

A **rooted tree** $T$ is defined recursively:

- An **empty tree** has no nodes.
- A **non-empty tree** consists of a distinguished node $r$ called the **root**, together with zero or more non-empty subtrees $T_1, T_2, \ldots, T_k$, each of whose roots is connected to $r$ by an edge.

This recursive structure is what makes trees natural for recursive algorithms.

## Core Terms

Consider the following example tree:

```
            A             <- root
          / | \
         B  C  D          <- children of A
        / \    |
       E   F   G          <- E,F are children of B; G is child of D
          / \
         H   I            <- children of F
```

### Nodes and Edges

| Term | Definition |
|------|-----------|
| **Node** (vertex) | A fundamental unit of a tree that stores data. Nodes $A$ through $I$ above. |
| **Edge** | A connection between a parent and a child. A tree with $n$ nodes has exactly $n - 1$ edges. |
| **Root** | The topmost node with no parent. Node $A$ in the example. |

### Family Relationships

| Term | Definition |
|------|-----------|
| **Parent** | The node directly above a given node. $B$ is the parent of $E$ and $F$. |
| **Child** | A node directly below a given node. $E$ and $F$ are children of $B$. |
| **Sibling** | Nodes that share the same parent. $B$, $C$, and $D$ are siblings. |
| **Ancestor** | Any node on the path from a node to the root (inclusive). The ancestors of $H$ are $H$, $F$, $B$, $A$. |
| **Descendant** | Any node reachable by following edges downward from a node. The descendants of $B$ are $E$, $F$, $H$, $I$. |

### Classification of Nodes

| Term | Definition |
|------|-----------|
| **Leaf** (external node) | A node with no children. Nodes $E$, $H$, $I$, $C$, $G$ are leaves. |
| **Internal node** | A node with at least one child. Nodes $A$, $B$, $D$, $F$ are internal. |
| **Degree** of a node | The number of children of that node. $\text{degree}(A) = 3$, $\text{degree}(B) = 2$, $\text{degree}(C) = 0$. |

### Structural Terms

| Term | Definition |
|------|-----------|
| **Subtree** | The tree formed by a node and all its descendants. The subtree rooted at $B$ contains $\{B, E, F, H, I\}$. |
| **Path** | A sequence of nodes $v_1, v_2, \ldots, v_k$ where consecutive nodes are connected by edges. |
| **Level** | The set of all nodes at the same depth. Level 0 contains only the root. |
| **Depth** | The number of edges from the root to a node. See [Height and Depth](height_depth.md). |
| **Height** | The number of edges on the longest path from a node to a leaf. See [Height and Depth](height_depth.md). |

## Key Properties

Several fundamental properties follow directly from the recursive definition:

1. **Edge count**: A tree with $n$ nodes has exactly $n - 1$ edges.
2. **Unique path**: There is exactly one path between any two nodes in a tree.
3. **Connected and acyclic**: A tree is a connected graph with no cycles. Removing any edge disconnects the tree; adding any edge creates a cycle.

!!! tip "Binary Tree Specialization"
    In a **binary tree**, every node has at most two children, designated as the **left child** and the **right child**. The left and right subtrees are themselves binary trees (possibly empty). Most of this chapter focuses on binary trees, but the terminology above applies to trees of any degree.

## Degree of a Tree

The **degree of a tree** is the maximum degree of any node in the tree:

$$
\text{degree}(T) = \max_{v \in T} \text{degree}(v)
$$

A binary tree has degree at most 2. A ternary tree has degree at most 3.

## Example in Python

```python
"""
Tree terminology demonstration.

Illustrates core tree concepts: root, parent, child, leaf,
internal node, depth, height, degree, subtree size, and ancestors.
"""


# === Node definition ===

class TreeNode:
    """A node in a general tree (arbitrary number of children)."""

    def __init__(self, key):
        self.key = key
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        """Add a child and set its parent pointer."""
        child_node.parent = self
        self.children.append(child_node)

    def is_leaf(self):
        """A leaf has no children."""
        return len(self.children) == 0

    def is_root(self):
        """The root has no parent."""
        return self.parent is None

    def degree(self):
        """Number of children."""
        return len(self.children)

    def depth(self):
        """Number of edges from root to this node."""
        d = 0
        current = self.parent
        while current is not None:
            d += 1
            current = current.parent
        return d

    def ancestors(self):
        """Return list of ancestors from self to root."""
        result = [self.key]
        current = self.parent
        while current is not None:
            result.append(current.key)
            current = current.parent
        return result

    def __repr__(self):
        return f"TreeNode({self.key})"


# === Tree queries ===

def subtree_size(node):
    """Return the number of nodes in the subtree rooted at node."""
    if node is None:
        return 0
    count = 1
    for child in node.children:
        count += subtree_size(child)
    return count


def tree_height(node):
    """Return the height of the subtree rooted at node."""
    if node is None or node.is_leaf():
        return 0
    return 1 + max(tree_height(c) for c in node.children)


# === Main ===

if __name__ == "__main__":
    # Build the example tree
    A = TreeNode("A")
    B, C, D = TreeNode("B"), TreeNode("C"), TreeNode("D")
    E, F, G = TreeNode("E"), TreeNode("F"), TreeNode("G")
    H, I = TreeNode("H"), TreeNode("I")

    A.add_child(B); A.add_child(C); A.add_child(D)
    B.add_child(E); B.add_child(F)
    D.add_child(G)
    F.add_child(H); F.add_child(I)

    nodes = [A, B, C, D, E, F, G, H, I]

    print(f"{'Node':<6} {'Depth':<6} {'Degree':<7} {'Leaf?':<6} {'Subtree size'}")
    print("-" * 45)
    for node in nodes:
        print(f"{node.key:<6} {node.depth():<6} {node.degree():<7} "
              f"{'yes' if node.is_leaf() else 'no':<6} {subtree_size(node)}")

    print(f"\nTree height: {tree_height(A)}")
    print(f"Ancestors of H: {H.ancestors()}")
    print(f"Edge count: {len(nodes) - 1}")
```

**Output:**
```
Node   Depth  Degree  Leaf?  Subtree size
---------------------------------------------
A      0      3       no     9
B      1      2       no     5
C      1      0       yes    1
D      1      1       no     2
E      2      0       yes    1
F      2      2       no     3
G      2      0       yes    1
H      3      0       yes    1
I      3      0       yes    1

Tree height: 3
Ancestors of H: ['H', 'F', 'B', 'A']
Edge count: 8
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
