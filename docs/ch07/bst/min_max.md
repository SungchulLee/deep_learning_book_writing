# Min and Max

Finding the minimum and maximum keys in a BST is straightforward thanks to the [BST property](property.md). Since all keys in a left subtree are less than or equal to the root, and all keys in a right subtree are greater, the minimum lies at the end of the leftmost path and the maximum at the end of the rightmost path. These operations are fundamental building blocks for [deletion](deletion.md) (which needs the minimum of the right subtree) and [successor/predecessor](successor.md) queries.

## Finding the Minimum

To find the minimum key, start at the root and follow left child pointers until reaching a node with no left child. That node holds the minimum key.

```
         8
        / \
       3   10
      / \    \
     1   6   14
    ^
    minimum (follow left pointers)
```

**Why it works**: By the BST property, every node in the left subtree has a key $\leq$ the parent's key. Following left pointers repeatedly leads to the smallest key in the tree. When a node has no left child, there is no key smaller than it in the subtree, so it is the minimum.

The algorithm runs in $O(h)$ time, where $h$ is the height of the tree, since it visits at most one node per level.

## Finding the Maximum

Symmetrically, the maximum key is found by starting at the root and following right child pointers until reaching a node with no right child.

```
         8
        / \
       3   10
      / \    \
     1   6   14
               ^
               maximum (follow right pointers)
```

This also runs in $O(h)$ time.

## Formal Statement

!!! note "Theorem"
    In a BST with $n \geq 1$ nodes and height $h$, the minimum key can be found in $O(h)$ time by following left child pointers from the root, and the maximum key can be found in $O(h)$ time by following right child pointers from the root.

**Proof**: Consider the minimum. Let $x_0 = \text{root}, x_1 = x_0.\text{left}, x_2 = x_1.\text{left}, \ldots, x_k$ where $x_k.\text{left} = \text{null}$. By the BST property, $x_i.\text{key} \geq x_{i+1}.\text{key}$ for all $i$, so $x_k$ has the smallest key among all nodes on this path. Moreover, for any node $y$ not on this path, $y$ is in the right subtree of some $x_i$, so $y.\text{key} > x_i.\text{key} \geq x_k.\text{key}$. Therefore $x_k.\text{key}$ is the global minimum. The path has at most $h$ edges, so the algorithm takes $O(h)$ time. The argument for maximum is symmetric. $\square$

## Implementation

```python
"""
Finding minimum and maximum in a binary search tree.

Demonstrates both recursive and iterative approaches for
finding the smallest and largest keys using the BST property.
"""


# === Node definition ===

class Node:
    """A node in a binary search tree."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node({self.key})"


# === Minimum ===

def find_min_iterative(node):
    """Find the minimum key by following left pointers."""
    if node is None:
        return None
    while node.left is not None:
        node = node.left
    return node


def find_min_recursive(node):
    """Find the minimum key recursively."""
    if node is None:
        return None
    if node.left is None:
        return node
    return find_min_recursive(node.left)


# === Maximum ===

def find_max_iterative(node):
    """Find the maximum key by following right pointers."""
    if node is None:
        return None
    while node.right is not None:
        node = node.right
    return node


def find_max_recursive(node):
    """Find the maximum key recursively."""
    if node is None:
        return None
    if node.right is None:
        return node
    return find_max_recursive(node.right)


# === BST construction ===

def insert(root, key):
    """Insert a key into the BST."""
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    return root


def inorder(node):
    """Yield keys in sorted order."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)


# === Main ===

if __name__ == "__main__":
    root = None
    keys = [8, 3, 10, 1, 6, 14, 4, 7, 13]
    for k in keys:
        root = insert(root, k)

    print(f"BST keys (inorder): {list(inorder(root))}")
    print()

    min_node = find_min_iterative(root)
    max_node = find_max_iterative(root)
    print(f"Minimum (iterative): {min_node.key}")
    print(f"Maximum (iterative): {max_node.key}")
    print()

    min_node_r = find_min_recursive(root)
    max_node_r = find_max_recursive(root)
    print(f"Minimum (recursive): {min_node_r.key}")
    print(f"Maximum (recursive): {max_node_r.key}")
    print()

    # Min/max of subtrees
    right_subtree_min = find_min_iterative(root.right)
    left_subtree_max = find_max_iterative(root.left)
    print(f"Min of right subtree (rooted at {root.right.key}): {right_subtree_min.key}")
    print(f"Max of left subtree (rooted at {root.left.key}): {left_subtree_max.key}")
```

**Output:**
```
BST keys (inorder): [1, 3, 4, 6, 7, 8, 10, 13, 14]

Minimum (iterative): 1
Maximum (iterative): 14

Minimum (recursive): 1
Maximum (recursive): 14

Min of right subtree (rooted at 10): 10
Max of left subtree (rooted at 3): 7
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12.2](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
