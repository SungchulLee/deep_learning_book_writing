# Binary Search Tree Property

The binary search tree (BST) property is the single invariant that transforms an ordinary binary tree into a powerful search structure. Just as binary search on a sorted array eliminates half the candidates at each comparison, the BST property allows us to eliminate an entire subtree at each node during a search. This connection between sorted order and tree structure is the foundation for all BST operations -- [search](search.md), [insertion](insertion.md), [deletion](deletion.md), and [successor/predecessor](successor.md) queries.

## Formal Definition

A binary tree satisfies the **BST property** if, for every node $x$:

$$
\text{For all nodes } y \text{ in the left subtree of } x: \quad y.\text{key} \leq x.\text{key}
$$

$$
\text{For all nodes } z \text{ in the right subtree of } x: \quad z.\text{key} > x.\text{key}
$$

Note that the property applies to **all** nodes in each subtree, not just the immediate children. A common mistake is to check only the direct children while ignoring deeper descendants.

!!! warning "Duplicate Keys"
    The convention for duplicate keys varies across textbooks. In this book, duplicates go into the **left subtree** (using $\leq$ for left and $>$ for right). Some implementations use $<$ for left and $\geq$ for right, or forbid duplicates entirely. The choice does not affect correctness as long as it is applied consistently.

## Visual Example

```
         8
        / \
       3   10
      / \    \
     1   6   14
        / \  /
       4  7 13
```

Consider node 8: every key in its left subtree ($\{1, 3, 4, 6, 7\}$) is less than or equal to 8, and every key in its right subtree ($\{10, 13, 14\}$) is greater than 8. This property holds recursively at every node in the tree.

The following tree violates the BST property even though each node's direct children satisfy the constraint:

```
         5
        / \
       3   8
      / \
     1   7      <- 7 > 5, so 7 is in the wrong subtree!
```

Node 3's right child is 7, which satisfies $7 > 3$. However, 7 is also in the left subtree of 5, and $7 > 5$ violates the BST property at the root.

## Inorder Traversal Produces Sorted Order

The most important consequence of the BST property is that an **inorder traversal** visits the nodes in non-decreasing order of their keys.

!!! note "Theorem"
    If $T$ is a binary search tree, then the inorder traversal of $T$ visits the keys in sorted (non-decreasing) order.

**Proof sketch**: By induction on the number of nodes. For a single node, the claim is trivial. For a tree rooted at $x$ with left subtree $L$ and right subtree $R$: by the BST property, all keys in $L$ are $\leq x.\text{key}$ and all keys in $R$ are $> x.\text{key}$. By the inductive hypothesis, inorder traversal of $L$ produces sorted output, and inorder traversal of $R$ produces sorted output. The inorder sequence $[L, x, R]$ therefore concatenates a sorted sequence of values $\leq x.\text{key}$, followed by $x.\text{key}$, followed by a sorted sequence of values $> x.\text{key}$. The result is sorted. $\square$

## Validating the BST Property

A correct validation must check that every node's key falls within the valid range inherited from its ancestors, not just compare a node with its immediate children.

```python
"""
Binary search tree property: definition, validation, and verification.

Demonstrates the BST invariant, shows how inorder traversal produces
sorted output, and provides correct BST validation using range checking.
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


# === BST validation ===

def is_valid_bst(node, min_key=float("-inf"), max_key=float("inf")):
    """Check whether the tree rooted at node satisfies the BST property.

    Each recursive call narrows the valid range [min_key, max_key]
    that the node's key must fall within.
    """
    if node is None:
        return True
    if node.key <= min_key or node.key > max_key:
        return False
    return (is_valid_bst(node.left, min_key, node.key) and
            is_valid_bst(node.right, node.key, max_key))


# === Inorder traversal ===

def inorder(node):
    """Yield keys in inorder (sorted) sequence."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)


# === Main ===

if __name__ == "__main__":
    # Valid BST
    valid_tree = Node(8,
        Node(3,
            Node(1),
            Node(6, Node(4), Node(7))),
        Node(10,
            None,
            Node(14, Node(13))))

    print("Valid BST:")
    print(f"  Inorder traversal: {list(inorder(valid_tree))}")
    print(f"  Is valid BST:      {is_valid_bst(valid_tree)}")
    print()

    # Invalid BST (7 in left subtree of 5 violates the property)
    invalid_tree = Node(5,
        Node(3, Node(1), Node(7)),
        Node(8))

    print("Invalid BST (7 in wrong subtree):")
    print(f"  Inorder traversal: {list(inorder(invalid_tree))}")
    print(f"  Is valid BST:      {is_valid_bst(invalid_tree)}")
```

**Output:**
```
Valid BST:
  Inorder traversal: [1, 3, 4, 6, 7, 8, 10, 13, 14]
  Is valid BST:      True

Invalid BST (7 in wrong subtree):
  Inorder traversal: [1, 3, 7, 5, 8]
  Is valid BST:      False
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 12](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
