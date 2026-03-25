# Successor and Predecessor

Many BST applications require finding not just a specific key but the **next** or **previous** key in sorted order.  Database range queries, iterator implementations, and order-statistic operations all depend on efficiently locating the in-order successor and predecessor of a given node.  Because a BST's in-order traversal yields keys in sorted order, successor and predecessor follow directly from the tree's structure without requiring a full traversal.

## In-Order Successor

The **in-order successor** of a node $x$ is the node with the smallest key greater than $x.key$.  The algorithm splits into two cases depending on whether $x$ has a right subtree.

**Case 1: $x$ has a right child.**  The successor is the leftmost node in $x$'s right subtree -- the minimum of that subtree.

**Case 2: $x$ has no right child.**  The successor is the lowest ancestor of $x$ whose left subtree contains $x$.  Starting from $x$, walk upward until you find a node that is a left child of its parent.  That parent is the successor.

$$
\text{successor}(x) =
\begin{cases}
\text{minimum}(x.\text{right}) & \text{if } x.\text{right} \neq \text{nil} \\
\text{lowest ancestor } y \text{ such that } x \text{ is in } y.\text{left subtree} & \text{otherwise}
\end{cases}
$$

??? example "Finding the successor"
    Consider the BST:

    ```
            15
           /  \
         10    20
        /  \   / \
       5   12 17  25
          /
         11
    ```

    - **Successor of 12:** Node 12 has no right child.  Walk up: 12 is the right child of 10, so continue.  10 is the left child of 15.  The successor is **15**.
    - **Successor of 10:** Node 10 has a right child (12).  The leftmost node in the right subtree of 10 is **11**.
    - **Successor of 15:** Node 15 has a right child (20).  The leftmost node in the right subtree of 15 is **17**.
    - **Successor of 25:** Node 25 has no right child and no ancestor whose left subtree contains it (beyond what we have already traversed).  The successor is **nil** -- 25 is the maximum.

## In-Order Predecessor

The **in-order predecessor** of a node $x$ is the node with the largest key smaller than $x.key$.  It mirrors the successor logic:

**Case 1: $x$ has a left child.**  The predecessor is the rightmost node in $x$'s left subtree -- the maximum of that subtree.

**Case 2: $x$ has no left child.**  The predecessor is the lowest ancestor of $x$ whose right subtree contains $x$.

$$
\text{predecessor}(x) =
\begin{cases}
\text{maximum}(x.\text{left}) & \text{if } x.\text{left} \neq \text{nil} \\
\text{lowest ancestor } y \text{ such that } x \text{ is in } y.\text{right subtree} & \text{otherwise}
\end{cases}
$$

## Implementation

Two implementations are provided: one using parent pointers (matching the textbook algorithm) and one without parent pointers (using a top-down search).

### With Parent Pointers

```python
"""BST successor and predecessor with parent pointers."""


# === Node Definition ===

class Node:
    """BST node with parent pointer."""

    def __init__(self, key: int):
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None
        self.parent: Node | None = None


# === Successor and Predecessor ===

def tree_minimum(x: Node) -> Node:
    """Return the node with the minimum key in the subtree rooted at x."""
    while x.left is not None:
        x = x.left
    return x


def tree_maximum(x: Node) -> Node:
    """Return the node with the maximum key in the subtree rooted at x."""
    while x.right is not None:
        x = x.right
    return x


def successor(x: Node) -> Node | None:
    """Return the in-order successor of node x, or None if x is the maximum."""
    if x.right is not None:
        return tree_minimum(x.right)
    y = x.parent
    while y is not None and x == y.right:
        x = y
        y = y.parent
    return y


def predecessor(x: Node) -> Node | None:
    """Return the in-order predecessor of node x, or None if x is the minimum."""
    if x.left is not None:
        return tree_maximum(x.left)
    y = x.parent
    while y is not None and x == y.left:
        x = y
        y = y.parent
    return y
```

### Without Parent Pointers

When nodes lack parent pointers, the successor is found by searching from the root.

```python
"""BST successor without parent pointers (top-down search)."""


# === Top-Down Successor ===

def successor_no_parent(root: Node | None, key: int) -> Node | None:
    """Find the in-order successor of the node with the given key."""
    successor_node = None
    current = root
    while current is not None:
        if key < current.key:
            successor_node = current  # candidate successor
            current = current.left
        elif key > current.key:
            current = current.right
        else:
            # Found the node with the target key
            if current.right is not None:
                return tree_minimum(current.right)
            return successor_node
    return None  # key not found


# === Demonstration ===

def insert(root: Node | None, key: int) -> Node:
    """Insert a key into the BST, maintaining parent pointers."""
    new_node = Node(key)
    if root is None:
        return new_node
    parent = None
    current = root
    while current is not None:
        parent = current
        if key < current.key:
            current = current.left
        else:
            current = current.right
    new_node.parent = parent
    if key < parent.key:
        parent.left = new_node
    else:
        parent.right = new_node
    return root


if __name__ == "__main__":
    root = None
    for k in [15, 10, 20, 5, 12, 17, 25, 11]:
        root = insert(root, k)

    # Find successors using parent pointers
    def find_node(root, key):
        while root and root.key != key:
            root = root.left if key < root.key else root.right
        return root

    for key in [5, 10, 11, 12, 15, 17, 20, 25]:
        node = find_node(root, key)
        succ = successor(node)
        pred = predecessor(node)
        succ_key = succ.key if succ else None
        pred_key = pred.key if pred else None
        print(f"key={key:2d}  successor={succ_key}  predecessor={pred_key}")
```

## Complexity

Both successor and predecessor run in $O(h)$ time, where $h$ is the height of the tree.  In the worst case (a degenerate tree), $h = n - 1$ so the time is $O(n)$.  For a balanced BST, $h = O(\log n)$.

No additional space is used beyond the stack frame (for the iterative versions shown, space is $O(1)$).

| Operation | Time | Space |
|---|---|---|
| Successor (with parent) | $O(h)$ | $O(1)$ |
| Predecessor (with parent) | $O(h)$ | $O(1)$ |
| Successor (top-down) | $O(h)$ | $O(1)$ |

!!! note "Amortized cost of iterating through all successors"
    Calling `successor` $n$ times to iterate through the entire BST in sorted order takes $O(n)$ total time, not $O(nh)$.  Each edge in the tree is traversed at most twice (once downward, once upward), giving an amortized cost of $O(1)$ per successor call.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
