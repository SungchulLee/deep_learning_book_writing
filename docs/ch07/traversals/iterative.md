# Iterative Traversals

Recursive tree traversals are elegant but use $O(h)$ stack frames implicitly, where $h$ is the tree height.  For very deep or unbalanced trees, this risks stack overflow.  **Iterative traversals** replace the call stack with an explicit stack data structure, achieving the same visit order with controlled memory usage and often better constant-factor performance due to reduced function-call overhead.

## Iterative Preorder

Preorder visits the current node **before** its children: root, left subtree, right subtree.

The key insight is that we push the right child first so that the left child is popped (and processed) first.

```python
"""Iterative tree traversals using an explicit stack."""

from __future__ import annotations


# === Node Definition ===

class TreeNode:
    """Binary tree node."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right


# === Iterative Preorder ===

def preorder(root: TreeNode | None) -> list[int]:
    """Preorder traversal: root -> left -> right."""
    if root is None:
        return []
    result: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```

!!! tip "Why push right before left?"
    The stack is LIFO.  Pushing right first and left second ensures that the left child is popped next, maintaining the preorder sequence (root, left, right).

## Iterative Inorder

Inorder visits the left subtree, then the current node, then the right subtree.  The iterative version uses a pointer to track the "go left as far as possible" phase.

```python
# === Iterative Inorder ===

def inorder(root: TreeNode | None) -> list[int]:
    """Inorder traversal: left -> root -> right."""
    result: list[int] = []
    stack: list[TreeNode] = []
    current = root
    while current or stack:
        # Go left as far as possible
        while current:
            stack.append(current)
            current = current.left
        # Visit the node
        current = stack.pop()
        result.append(current.val)
        # Move to the right subtree
        current = current.right
    return result
```

The algorithm maintains two invariants:

1. Every node on the stack has not yet been visited, but its left subtree is being (or has been) explored.
2. The `current` pointer tracks the next node whose left subtree must be fully explored before visiting it.

## Iterative Postorder

Postorder visits the left subtree, then the right subtree, then the current node.  This is the most complex to implement iteratively because the root must be visited **last**.

### Two-Stack Approach

A clean approach uses two stacks.  The first stack drives a modified preorder (root, right, left), and the second stack reverses the result to obtain postorder (left, right, root).

```python
# === Iterative Postorder (two stacks) ===

def postorder_two_stacks(root: TreeNode | None) -> list[int]:
    """Postorder traversal using two stacks."""
    if root is None:
        return []
    stack1 = [root]
    stack2: list[int] = []
    while stack1:
        node = stack1.pop()
        stack2.append(node.val)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    return stack2[::-1]
```

### One-Stack Approach

A single-stack version tracks the previously visited node to determine whether we are returning from the left or right subtree.

```python
# === Iterative Postorder (one stack) ===

def postorder_one_stack(root: TreeNode | None) -> list[int]:
    """Postorder traversal using a single stack."""
    if root is None:
        return []
    result: list[int] = []
    stack: list[TreeNode] = []
    current = root
    last_visited: TreeNode | None = None
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        peek = stack[-1]
        if peek.right and peek.right != last_visited:
            current = peek.right
        else:
            result.append(peek.val)
            last_visited = stack.pop()
    return result
```

!!! note "The `last_visited` trick"
    After visiting a node's left subtree, we peek at the stack top.  If it has an unvisited right child, we move there.  Otherwise, we visit the node itself and mark it as `last_visited` so we do not re-enter its right subtree.

## Comparison

| Traversal | Recursive stack depth | Iterative stack size | Implementation complexity |
|---|---|---|---|
| Preorder | $O(h)$ | $O(h)$ | Simple |
| Inorder | $O(h)$ | $O(h)$ | Moderate |
| Postorder | $O(h)$ | $O(h)$ (one-stack) or $O(n)$ (two-stack) | Complex |

All three iterative traversals use $O(h)$ space with the one-stack approach, matching the recursive versions.  The two-stack postorder uses $O(n)$ space since `stack2` holds all $n$ node values.

## Demonstration

```python
# === Demonstration ===

if __name__ == "__main__":
    # Build tree:      1
    #                 / \
    #                2   3
    #               / \
    #              4   5
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3))

    print(f"Preorder:  {preorder(tree)}")           # [1, 2, 4, 5, 3]
    print(f"Inorder:   {inorder(tree)}")            # [4, 2, 5, 1, 3]
    print(f"Postorder: {postorder_two_stacks(tree)}") # [4, 5, 2, 3, 1]
    print(f"Postorder: {postorder_one_stack(tree)}")  # [4, 5, 2, 3, 1]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
