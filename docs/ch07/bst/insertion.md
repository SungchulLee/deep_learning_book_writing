# Insertion

Insertion is how a BST grows. Starting from an empty tree, we build the entire structure by inserting keys one at a time. Each insertion descends from the root, comparing the new key at each node to decide whether to go left or right, until it finds an empty position where the new node becomes a leaf. Because the new node always lands at a leaf position, insertion never restructures existing nodes -- it only extends the tree downward.

## Algorithm

To insert a key $k$ into a BST rooted at $r$:

1. If the tree is empty, create a new node with key $k$ and return it as the root.
2. Compare $k$ with $r.\text{key}$:
    - If $k \leq r.\text{key}$, recursively insert into the left subtree.
    - If $k > r.\text{key}$, recursively insert into the right subtree.
3. Return the (possibly updated) root.

The new node always becomes a leaf because the recursion continues until it reaches a null position.

## Visual Walkthrough

Inserting the key 5 into this BST:

```
Step 1: Start at root (8).        Step 2: 5 < 8, go left to 3.
         8                                  8
        / \                                / \
       3   10                             3   10
      / \                                / \
     1   6                              1   6

Step 3: 5 > 3, go right to 6.     Step 4: 5 < 6, go left (null).
         8                                  8
        / \                                / \
       3   10                             3   10
      / \                                / \
     1   6                              1   6
                                           /
                                          5   <- new leaf
```

## Preserving the BST Property

!!! note "Correctness"
    Insertion preserves the [BST property](property.md). At each step, the algorithm chooses the subtree that maintains the ordering invariant: keys less than or equal to the current node go left, and keys greater go right. The new leaf's position guarantees that every ancestor's BST property remains intact.

## Complexity

Insertion follows a single root-to-leaf path, performing $O(1)$ work at each node. The time complexity is $O(h)$, where $h$ is the height of the tree:

- **Balanced tree**: $O(\log n)$
- **Degenerate tree**: $O(n)$

See [Complexity](complexity.md) for a detailed analysis of how insertion order affects tree height.

## Recursive vs Iterative Implementation

The recursive approach is elegant and mirrors the algorithm description directly. The iterative approach avoids stack overhead and is preferred when the tree may be very deep (to prevent stack overflow).

```python
"""
BST insertion: recursive and iterative implementations.

Shows how insertion always creates a new leaf while preserving
the BST property, and demonstrates tree building from a sequence.
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


# === Recursive insertion ===

def insert_recursive(root, key):
    """Insert a key into the BST using recursion.

    Returns the root of the modified tree.
    New nodes are always created as leaves.
    """
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert_recursive(root.left, key)
    else:
        root.right = insert_recursive(root.right, key)
    return root


# === Iterative insertion ===

def insert_iterative(root, key):
    """Insert a key into the BST using iteration.

    Walks down the tree to find the correct null position,
    then attaches the new node.
    """
    new_node = Node(key)
    if root is None:
        return new_node

    parent = None
    current = root
    while current is not None:
        parent = current
        if key <= current.key:
            current = current.left
        else:
            current = current.right

    if key <= parent.key:
        parent.left = new_node
    else:
        parent.right = new_node

    return root


# === Display helpers ===

def inorder(node):
    """Yield keys in sorted order."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)


def print_tree(node, level=0, prefix="Root: "):
    """Print the tree structure with indentation."""
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.key))
        if node.left is not None or node.right is not None:
            print_tree(node.left, level + 1, "L--- ")
            print_tree(node.right, level + 1, "R--- ")


# === Main ===

if __name__ == "__main__":
    # Build a BST by inserting keys one at a time
    keys = [8, 3, 10, 1, 6, 14, 4, 7, 13]

    root = None
    for k in keys:
        root = insert_recursive(root, k)
        print(f"Insert {k}: inorder = {list(inorder(root))}")

    print()
    print("Final tree structure:")
    print_tree(root)

    # Verify iterative gives same result
    root2 = None
    for k in keys:
        root2 = insert_iterative(root2, k)
    print()
    print(f"Iterative inorder: {list(inorder(root2))}")
    print(f"Results match: {list(inorder(root)) == list(inorder(root2))}")
```

**Output:**
```
Insert 8: inorder = [8]
Insert 3: inorder = [3, 8]
Insert 10: inorder = [3, 8, 10]
Insert 1: inorder = [1, 3, 8, 10]
Insert 6: inorder = [1, 3, 6, 8, 10]
Insert 14: inorder = [1, 3, 6, 8, 10, 14]
Insert 4: inorder = [1, 3, 4, 6, 8, 10, 14]
Insert 7: inorder = [1, 3, 4, 6, 7, 8, 10, 14]
Insert 13: inorder = [1, 3, 4, 6, 7, 8, 10, 13, 14]

Final tree structure:
Root: 8
    L--- 3
        L--- 1
        R--- 6
            L--- 4
            R--- 7
    R--- 10
        R--- 14
            L--- 13

Iterative inorder: [1, 3, 4, 6, 7, 8, 10, 13, 14]
Results match: True
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12.3](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
