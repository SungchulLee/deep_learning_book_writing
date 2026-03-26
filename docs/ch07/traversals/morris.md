# Morris Traversal

Both recursive and [iterative](iterative.md) tree traversals use $O(h)$ auxiliary space, either via the call stack or an explicit stack.  **Morris traversal** (Morris, 1979) eliminates this space overhead entirely.  It achieves inorder traversal in $O(1)$ auxiliary space by temporarily modifying the tree structure itself, creating **threaded links** from the rightmost node of each left subtree back to its inorder successor.  These threads let the traversal return to ancestor nodes without a stack.  After the traversal completes, all modifications are undone, leaving the tree in its original state.

## Core Idea: Threaded Binary Trees

In a standard binary tree, many right-child pointers are `nil`.  Morris traversal exploits these null pointers by temporarily pointing them to the inorder successor.  Specifically, for a node $x$ with a left subtree, the traversal sets:

$$
\text{rightmost node of } x.\text{left} \;\rightarrow\; x
$$

This thread allows the traversal to return to $x$ after finishing $x$'s left subtree, without using a stack.

## Algorithm

The algorithm maintains a single pointer `current`, starting at the root.  At each step:

**Case 1: `current.left` is `nil`.**  Visit `current` and move to `current.right`.

**Case 2: `current.left` is not `nil`.**  Find the inorder predecessor of `current` — the rightmost node in `current`'s left subtree.

- **Sub-case 2a:** The predecessor's right child is `nil`.  This means the left subtree has not been traversed yet.  Create a thread: set `predecessor.right = current`.  Move to `current.left`.
- **Sub-case 2b:** The predecessor's right child is `current`.  This means the left subtree has been fully traversed and we have returned via the thread.  Remove the thread: set `predecessor.right = nil`.  Visit `current` and move to `current.right`.

The traversal terminates when `current` becomes `nil`.

??? example "Step-by-step Morris traversal"
    Consider the tree:

    ```
          4
         / \
        2   5
       / \
      1   3
    ```

    | Step | current | Action | Thread created/removed |
    |------|---------|--------|------------------------|
    | 1 | 4 | Left exists. Predecessor of 4 is 3. 3.right is nil | Create thread: 3 -> 4. Move to 2 |
    | 2 | 2 | Left exists. Predecessor of 2 is 1. 1.right is nil | Create thread: 1 -> 2. Move to 1 |
    | 3 | 1 | No left child | **Visit 1**. Move to 1.right = 2 (thread) |
    | 4 | 2 | Left exists. Predecessor of 2 is 1. 1.right = 2 (thread found) | Remove thread. **Visit 2**. Move to 3 |
    | 5 | 3 | No left child | **Visit 3**. Move to 3.right = 4 (thread) |
    | 6 | 4 | Left exists. Predecessor of 4 is 3. 3.right = 4 (thread found) | Remove thread. **Visit 4**. Move to 5 |
    | 7 | 5 | No left child | **Visit 5**. Move to nil |

    Inorder result: **1, 2, 3, 4, 5**

## Implementation

```python
"""Morris traversal: O(1) space inorder traversal of a binary tree."""

from __future__ import annotations


# === Node Definition ===

class TreeNode:
    """Binary tree node."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right


# === Morris Inorder Traversal ===

def morris_inorder(root: TreeNode | None) -> list[int]:
    """Inorder traversal using O(1) auxiliary space."""
    result: list[int] = []
    current = root
    while current is not None:
        if current.left is None:
            # Case 1: no left subtree — visit and move right
            result.append(current.val)
            current = current.right
        else:
            # Find the inorder predecessor
            predecessor = current.left
            while predecessor.right is not None and predecessor.right is not current:
                predecessor = predecessor.right

            if predecessor.right is None:
                # Case 2a: create thread and move left
                predecessor.right = current
                current = current.left
            else:
                # Case 2b: remove thread, visit, and move right
                predecessor.right = None
                result.append(current.val)
                current = current.right
    return result


# === Morris Preorder Traversal ===

def morris_preorder(root: TreeNode | None) -> list[int]:
    """Preorder traversal using O(1) auxiliary space."""
    result: list[int] = []
    current = root
    while current is not None:
        if current.left is None:
            result.append(current.val)
            current = current.right
        else:
            predecessor = current.left
            while predecessor.right is not None and predecessor.right is not current:
                predecessor = predecessor.right

            if predecessor.right is None:
                # Visit current BEFORE moving left (preorder)
                result.append(current.val)
                predecessor.right = current
                current = current.left
            else:
                predecessor.right = None
                current = current.right
    return result


# === Demonstration ===

if __name__ == "__main__":
    tree = TreeNode(4,
        TreeNode(2, TreeNode(1), TreeNode(3)),
        TreeNode(5))

    print(f"Morris inorder:  {morris_inorder(tree)}")   # [1, 2, 3, 4, 5]
    print(f"Morris preorder: {morris_preorder(tree)}")  # [4, 2, 1, 3, 5]
```

## Why the Total Work is O(n)

Although the algorithm repeatedly searches for predecessors, an amortized argument shows the total work is linear.  Each node becomes the `current` pointer at most twice: once when the thread from its predecessor is created, and once when that thread is removed.  Nodes may also be touched during predecessor searches — but each such walk follows a chain of right-child pointers, and every edge in the tree is traversed at most twice across all predecessor searches combined (once to create a thread, once to detect and remove it).

Since the tree has $n - 1$ edges, the aggregate cost of all predecessor searches is $O(n)$.

## Complexity

These observations yield the following complexity bounds:

| Aspect | Complexity |
|---|---|
| Time | $O(n)$ |
| Auxiliary space | $O(1)$ |

The $O(1)$ space is the defining advantage of Morris traversal.  The price is temporary modification of the tree, which is fully restored before the traversal completes.

!!! warning "Thread safety"
    Morris traversal modifies the tree during execution.  It is not safe to run concurrently with other operations on the same tree.  If thread safety is required, use a stack-based iterative traversal instead.

## Morris Preorder vs Morris Inorder

The only difference between Morris preorder and inorder is **when** the node is visited:

| Traversal | Visit timing |
|---|---|
| Morris inorder | Visit when the thread is **removed** (Case 2b) |
| Morris preorder | Visit when the thread is **created** (Case 2a) |

Morris postorder is more complex because postorder requires visiting children before parents, which conflicts with the natural threading direction.  The standard approach reverses the right spine of each left subtree during the thread-removal step; this technique is rarely needed in practice.

## Reference

- Morris, J. H. (1979). Traversing binary trees simply and cheaply. *Information Processing Letters*, 9(5), 197–200.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
