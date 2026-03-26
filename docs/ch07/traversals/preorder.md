# Preorder Traversal

Preorder traversal visits every node in a binary tree by processing the **root first**, then recursively visiting the left subtree, and finally the right subtree.  This "root before children" ordering makes preorder traversal the natural choice whenever a parent must be handled before its descendants — for example, when copying a tree, serializing its structure, or generating a prefix expression from an expression tree.

## Definition

Given a binary tree rooted at $r$, the **preorder traversal** visits nodes in the order:

$$
\text{visit}(r), \quad \text{preorder}(r.\text{left}), \quad \text{preorder}(r.\text{right})
$$

The base case is an empty subtree (null pointer), which produces no output.  For a tree with $n$ nodes, the traversal produces a sequence of exactly $n$ visits.

## Recursive Algorithm

The recursive formulation mirrors the definition directly.  Each call processes the current node, then delegates to left and right children.

```python
"""Preorder traversal of a binary tree: recursive and iterative approaches."""

from __future__ import annotations


# === Node Definition ===

class TreeNode:
    """Binary tree node with an integer value."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right


# === Recursive Preorder ===

def preorder_recursive(root: TreeNode | None) -> list[int]:
    """Return the preorder traversal of the tree rooted at *root*."""
    if root is None:
        return []
    return [root.val] + preorder_recursive(root.left) + preorder_recursive(root.right)
```

The recursive version is concise but uses $O(h)$ space on the call stack, where $h$ is the height of the tree.  For a skewed tree, $h = n - 1$, so the worst-case space is $O(n)$.

## Iterative Algorithm

An explicit stack replaces the call stack.  The key insight is that the right child must be pushed **before** the left child so that the left child is popped (and visited) first.

```python
# === Iterative Preorder ===

def preorder_iterative(root: TreeNode | None) -> list[int]:
    """Iterative preorder traversal using an explicit stack."""
    if root is None:
        return []
    stack: list[TreeNode] = [root]
    result: list[int] = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push right first so left is processed first
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result
```

??? example "Trace of iterative preorder"
    Consider the tree:

    ```
          1
         / \
        2   3
       / \
      4   5
    ```

    | Step | Stack (top→right) | Pop | Output so far |
    |------|-------------------|-----|---------------|
    | 0 | [1] | — | [] |
    | 1 | [3, 2] | 1 | [1] |
    | 2 | [3, 5, 4] | 2 | [1, 2] |
    | 3 | [3, 5] | 4 | [1, 2, 4] |
    | 4 | [3] | 5 | [1, 2, 4, 5] |
    | 5 | [] | 3 | [1, 2, 4, 5, 3] |

    Result: **1, 2, 4, 5, 3**

## Complexity

| Aspect | Recursive | Iterative |
|--------|-----------|-----------|
| Time | $O(n)$ | $O(n)$ |
| Space | $O(h)$ call stack | $O(h)$ explicit stack |

Both approaches visit every node exactly once, giving $O(n)$ time.  The space is $O(h)$ in both cases — $O(\log n)$ for a balanced tree and $O(n)$ in the worst case.

!!! tip "Choosing between recursive and iterative"
    The recursive version is simpler and sufficient for most use cases.  Prefer the iterative version when the tree may be very deep (risking stack overflow) or when you need fine-grained control over traversal state.

## Applications

Preorder traversal appears in several practical scenarios:

- **Tree copying:** processing the root before its children ensures the parent node exists before children are attached.
- **Serialization and deserialization:** preorder combined with null markers uniquely encodes a binary tree.
- **Prefix expression evaluation:** an expression tree traversed in preorder yields the prefix (Polish) notation.
- **Directory listing:** a file system tree printed with indentation uses preorder ordering — the directory name appears before its contents.

## Demonstration

```python
# === Demonstration ===

if __name__ == "__main__":
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3))

    print(f"Recursive preorder: {preorder_recursive(tree)}")  # [1, 2, 4, 5, 3]
    print(f"Iterative preorder: {preorder_iterative(tree)}")  # [1, 2, 4, 5, 3]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
- Knuth, D. E. (1997). *The Art of Computer Programming*, Volume 1, Section 2.3.1. Addison-Wesley.
