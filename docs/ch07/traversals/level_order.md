# Level-Order Traversal

While preorder, inorder, and postorder traversals explore the tree depth-first, **level-order traversal** visits nodes breadth-first: all nodes at depth 0 (the root), then all nodes at depth 1, then depth 2, and so on.  This is equivalent to running BFS on the tree, using a queue to process nodes in the order they are discovered.

## Algorithm

Level-order traversal uses a FIFO queue:

1. Enqueue the root.
2. While the queue is not empty:
    - Dequeue a node and visit it.
    - Enqueue its left child (if it exists).
    - Enqueue its right child (if it exists).

The queue ensures that all nodes at depth $d$ are processed before any node at depth $d+1$, because a node at depth $d$ enqueues its children (at depth $d+1$) after all other depth-$d$ nodes are already in the queue.

??? example "Level-order traversal"
    Consider the tree:

    ```
            1
           / \
          2   3
         / \   \
        4   5   6
    ```

    | Queue state | Dequeue | Enqueue |
    |---|---|---|
    | [1] | 1 | 2, 3 |
    | [2, 3] | 2 | 4, 5 |
    | [3, 4, 5] | 3 | 6 |
    | [4, 5, 6] | 4 | -- |
    | [5, 6] | 5 | -- |
    | [6] | 6 | -- |

    Visit order: **1, 2, 3, 4, 5, 6** (level by level, left to right).

## Implementation

### Basic Level-Order

```python
"""Level-order (BFS) traversal of a binary tree."""

from __future__ import annotations
from collections import deque


# === Node Definition ===

class TreeNode:
    """Binary tree node."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right


# === Basic Level-Order ===

def level_order(root: TreeNode | None) -> list[int]:
    """Return node values in level-order."""
    if root is None:
        return []
    result: list[int] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

### Level-by-Level Grouping

A common variant groups nodes by their level, returning a list of lists.  This is achieved by processing all nodes at the current level in one batch before moving to the next.

```python
# === Level-by-Level Grouping ===

def level_order_grouped(root: TreeNode | None) -> list[list[int]]:
    """Return node values grouped by level."""
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level_size = len(queue)
        level: list[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

The key technique is capturing `len(queue)` at the start of each level.  This tells us exactly how many nodes belong to the current level; all subsequently enqueued nodes belong to the next level.

### Zigzag (Spiral) Order

Zigzag traversal alternates direction at each level: left-to-right at even depths, right-to-left at odd depths.

```python
# === Zigzag Level-Order ===

def zigzag_level_order(root: TreeNode | None) -> list[list[int]]:
    """Return node values in zigzag (spiral) level order."""
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    left_to_right = True
    while queue:
        level_size = len(queue)
        level: list[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if not left_to_right:
            level.reverse()
        result.append(level)
        left_to_right = not left_to_right
    return result
```

## Complexity

| Aspect | Complexity |
|---|---|
| Time | $O(n)$ -- each node enqueued and dequeued exactly once |
| Space | $O(w)$ where $w$ is the maximum width of the tree |

The maximum width $w$ occurs at the widest level.  For a complete binary tree of height $h$, the last level has $2^h$ nodes, so $w = O(n)$ in the worst case.  For a skewed tree, $w = O(1)$.

!!! note "Space comparison with DFS traversals"
    DFS traversals (preorder, inorder, postorder) use $O(h)$ space where $h$ is the height.  Level-order uses $O(w)$ space where $w$ is the maximum width.  For balanced trees, $h = O(\log n)$ while $w = O(n)$, making DFS more space-efficient.  For skewed trees, $h = O(n)$ while $w = O(1)$, making level-order more space-efficient.

## Applications

- **Printing a tree level by level** for debugging or visualization.
- **Finding the minimum depth** of a tree (the first leaf encountered by BFS is at the minimum depth).
- **Serialization and deserialization** of binary trees (level-order encoding is commonly used).
- **Connecting nodes at the same level** (e.g., populating "next right" pointers).
- **Computing the width** of each level for maximum-width queries.

## Demonstration

```python
# === Demonstration ===

if __name__ == "__main__":
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3, None, TreeNode(6)))

    print(f"Level-order:  {level_order(tree)}")
    print(f"By level:     {level_order_grouped(tree)}")
    print(f"Zigzag:       {zigzag_level_order(tree)}")
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
