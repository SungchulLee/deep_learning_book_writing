# Deletion

Deletion is the most involved BST operation because removing a node must preserve the [BST property](property.md) for the entire tree. Unlike [insertion](insertion.md), which always adds a leaf, deletion may require restructuring when the target node has children. The algorithm handles three cases based on how many children the node to be deleted has.

## The Three Cases

Let $z$ be the node to delete.

### Case 1: Leaf Node (No Children)

If $z$ has no children, simply remove it by setting its parent's corresponding child pointer to null.

```
Delete 4:
     5              5
    / \            / \
   3   7    ->    3   7
  /
 4
```

### Case 2: One Child

If $z$ has exactly one child, replace $z$ with its child. The child's subtree "moves up" to take $z$'s position.

```
Delete 3 (has only left child 1):
     5              5
    / \            / \
   3   7    ->    1   7
  /
 1
```

### Case 3: Two Children

If $z$ has two children, we cannot simply remove it without disconnecting the tree. Instead, we find $z$'s **inorder successor** $y$ -- the node with the smallest key in $z$'s right subtree -- and use it to replace $z$.

The inorder successor $y$ is found by going to $z$'s right child and then following left pointers until reaching a node with no left child. Since $y$ has no left child, removing $y$ from its current position falls under Case 1 or Case 2.

The procedure:

1. Find $y$ = [successor](successor.md) of $z$ (smallest node in $z$'s right subtree).
2. Replace $z$'s key with $y$'s key.
3. Delete $y$ from its original position (which is Case 1 or Case 2).

```
Delete 5 (has two children):
     5              6
    / \            / \
   3   8    ->    3   8
  / \ /         / \
 1  4 6        1   4
      \              \
       7              7
```

Here, the inorder successor of 5 is 6. We copy 6 into the root position and then delete the original node 6 (which has one child, 7).

!!! note "Successor vs Predecessor"
    Instead of the inorder successor, we could use the **inorder predecessor** (the largest node in the left subtree). Both approaches preserve the BST property. Some implementations alternate between the two to keep the tree more balanced on average.

## Transplant Helper

The CLRS approach uses a `transplant` subroutine that replaces one subtree with another. `transplant(T, u, v)` replaces the subtree rooted at $u$ with the subtree rooted at $v$ by updating $u$'s parent to point to $v$ instead.

## Complexity

Deletion visits at most two root-to-leaf paths (one to find the node, one to find its successor), so it runs in $O(h)$ time, where $h$ is the height of the tree. For a balanced BST, this is $O(\log n)$; for a degenerate tree, this is $O(n)$.

## Implementation

```python
"""
BST deletion with all three cases.

Implements the standard BST deletion algorithm: leaf removal,
single-child replacement, and two-child replacement using the
inorder successor.
"""


# === Node definition ===

class Node:
    """A node in a binary search tree."""

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Node({self.key})"


# === BST operations ===

def insert(root, key):
    """Insert a key into the BST."""
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    return root


def find_min(node):
    """Find the node with the minimum key in the subtree."""
    while node.left is not None:
        node = node.left
    return node


def delete(root, key):
    """Delete a node with the given key from the BST.

    Handles three cases:
      1. Leaf node: remove directly.
      2. One child: replace with child.
      3. Two children: replace with inorder successor.
    """
    if root is None:
        return None

    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # Found the node to delete
        # Case 1 and 2: zero or one child
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left

        # Case 3: two children
        # Find inorder successor (min of right subtree)
        successor = find_min(root.right)
        root.key = successor.key
        root.right = delete(root.right, successor.key)

    return root


# === Display ===

def inorder(node):
    """Yield keys in sorted order."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)


def print_tree(node, level=0, prefix="Root: "):
    """Print tree structure."""
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.key))
        if node.left is not None or node.right is not None:
            print_tree(node.left, level + 1, "L--- ")
            print_tree(node.right, level + 1, "R--- ")


# === Main ===

if __name__ == "__main__":
    root = None
    for k in [5, 3, 8, 1, 4, 6, 9, 7]:
        root = insert(root, k)

    print("Original tree:")
    print_tree(root)
    print(f"Inorder: {list(inorder(root))}")
    print()

    # Case 1: delete leaf
    root = delete(root, 4)
    print("After deleting 4 (leaf):")
    print(f"Inorder: {list(inorder(root))}")
    print()

    # Case 2: delete node with one child
    root = delete(root, 6)
    print("After deleting 6 (one child: 7):")
    print(f"Inorder: {list(inorder(root))}")
    print()

    # Case 3: delete node with two children
    root = delete(root, 5)
    print("After deleting 5 (two children, successor=7):")
    print_tree(root)
    print(f"Inorder: {list(inorder(root))}")
```

**Output:**
```
Original tree:
Root: 5
    L--- 3
        L--- 1
        R--- 4
    R--- 8
        L--- 6
            R--- 7
        R--- 9
Inorder: [1, 3, 4, 5, 6, 7, 8, 9]

After deleting 4 (leaf):
Inorder: [1, 3, 5, 6, 7, 8, 9]

After deleting 6 (one child: 7):
Inorder: [1, 3, 5, 7, 8, 9]

After deleting 5 (two children, successor=7):
Root: 7
    L--- 3
        L--- 1
    R--- 8
        R--- 9
Inorder: [1, 3, 7, 8, 9]
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12.3](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
