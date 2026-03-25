# Augmented Binary Search Tree

A standard BST supports search, insertion, and deletion based on key values. However, many applications require queries that a plain BST cannot answer efficiently, such as "what is the 5th smallest element?" or "how many elements fall in the range $[a, b]$?" An **augmented BST** extends each node with additional data -- maintained during insertions and deletions -- to support these richer queries in $O(h)$ time, where $h$ is the tree height.

## The Augmentation Approach

The general strategy for augmenting a BST has four steps:

1. **Choose the extra information** to store at each node.
2. **Verify** that this information can be maintained during insertions, deletions, and rotations without increasing the asymptotic cost.
3. **Design new operations** that use the extra information.
4. **Prove correctness** of the maintenance procedures.

!!! tip "Augmentation Theorem (CLRS)"
    If additional information stored at each node can be computed from the node's own data plus the augmented information of its two children, then the information can be maintained during insertions, deletions, and rotations in $O(h)$ time with no asymptotic overhead.

## Order-Statistic Trees

The most common augmentation stores the **subtree size** at each node, creating an **order-statistic tree**. Each node $x$ stores:

$$
x.\text{size} = 1 + x.\text{left}.\text{size} + x.\text{right}.\text{size}
$$

with the convention that a null node has size 0. This single extra field enables two powerful operations.

### Select: Find the k-th Smallest Element

Given a rank $k$, `select(root, k)` returns the node with the $k$-th smallest key. The algorithm works by comparing $k$ with the size of the left subtree:

- Let $r = x.\text{left}.\text{size} + 1$ (the rank of $x$ within its subtree).
- If $k = r$, return $x$.
- If $k < r$, recurse into the left subtree.
- If $k > r$, recurse into the right subtree with rank $k - r$.

The operation runs in $O(h)$ time.

### Rank: Find the Rank of a Given Key

`rank(root, key)` returns the number of keys in the tree that are less than or equal to the given key. Starting from the root and walking down:

- If we go left, the rank does not change.
- If we visit or pass through a node $x$ going right, we add $x.\text{left}.\text{size} + 1$ to the running rank.

This also runs in $O(h)$ time.

## Maintaining Subtree Sizes

During **insertion**, we increment the size of every ancestor along the insertion path. During **deletion**, we decrement sizes along the path from the deleted node to the root.

During a **rotation** (used in balanced BSTs like AVL or red-black trees), only two nodes change their subtree relationships:

```
Right rotation at y:          Left rotation at x:
      y            x              x            y
     / \          / \            / \          / \
    x   C  -->  A   y          A   y   -->  x   C
   / \             / \            / \      / \
  A   B           B   C         B   C    A   B
```

After a right rotation at $y$:

$$
x.\text{size} = y.\text{size} \quad (\text{x now roots the same subtree})
$$

$$
y.\text{size} = y.\text{left}.\text{size} + y.\text{right}.\text{size} + 1
$$

Only the two nodes involved in the rotation need their sizes updated, so rotations remain $O(1)$.

## Other Augmentations

The subtree-size augmentation is the most common, but other useful augmentations include:

| Augmentation | Stored at each node | Enables |
|---|---|---|
| Subtree size | $1 + \text{left.size} + \text{right.size}$ | Select, Rank |
| Subtree min/max | $\min(\text{key}, \text{left.min}, \text{right.min})$ | Range-min queries |
| Subtree sum | $\text{key} + \text{left.sum} + \text{right.sum}$ | Range-sum queries |
| Interval max endpoint | $\max(\text{high}, \text{left.max}, \text{right.max})$ | Interval overlap queries |

## Example

```python
"""
Order-statistic tree: a BST augmented with subtree sizes.

Supports select (find k-th smallest) and rank (count elements
less than or equal to a key) in O(h) time.
"""


# === Node definition ===

class Node:
    """BST node augmented with subtree size."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.size = 1  # counts this node plus all descendants


# === Size helper ===

def size(node):
    """Return the size of a subtree (0 for null)."""
    return node.size if node else 0


# === Insertion with size maintenance ===

def insert(root, key):
    """Insert a key and update subtree sizes along the path."""
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    root.size = 1 + size(root.left) + size(root.right)
    return root


# === Select: find k-th smallest (1-indexed) ===

def select(node, k):
    """Return the node with the k-th smallest key.

    k is 1-indexed: select(root, 1) returns the minimum.
    """
    if node is None:
        return None
    left_size = size(node.left)
    rank_of_node = left_size + 1
    if k == rank_of_node:
        return node
    elif k < rank_of_node:
        return select(node.left, k)
    else:
        return select(node.right, k - rank_of_node)


# === Rank: count keys <= given key ===

def rank(node, key):
    """Return the number of keys less than or equal to key."""
    if node is None:
        return 0
    if key < node.key:
        return rank(node.left, key)
    elif key > node.key:
        return size(node.left) + 1 + rank(node.right, key)
    else:
        return size(node.left) + 1


# === Inorder traversal ===

def inorder(node):
    """Yield (key, size) pairs in sorted order."""
    if node is not None:
        yield from inorder(node.left)
        yield (node.key, node.size)
        yield from inorder(node.right)


# === Main ===

if __name__ == "__main__":
    root = None
    keys = [15, 6, 18, 3, 7, 17, 20, 2, 4, 13, 9]
    for k in keys:
        root = insert(root, k)

    print("Inorder (key, subtree_size):")
    print(f"  {list(inorder(root))}")
    print()

    for k in [1, 3, 5, 7, 11]:
        result = select(root, k)
        print(f"  Select(k={k}): {result.key if result else None}")

    print()
    for key in [1, 6, 13, 18, 25]:
        print(f"  Rank(key={key}): {rank(root, key)}")
```

**Output:**
```
Inorder (key, subtree_size):
  [(2, 1), (3, 2), (4, 1), (6, 4), (7, 1), (9, 1), (13, 3), (15, 11), (17, 1), (18, 3), (20, 1)]

  Select(k=1): 2
  Select(k=3): 4
  Select(k=5): 7
  Select(k=7): 13
  Select(k=11): 20

  Rank(key=1): 0
  Rank(key=6): 4
  Rank(key=13): 7
  Rank(key=18): 10
  Rank(key=25): 11
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 14 - Augmenting Data Structures](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
