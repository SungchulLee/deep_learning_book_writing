# Treap Split and Merge

While standard BST operations (search, insert, delete) work on treaps using rotations, the **split** and **merge** primitives offer a cleaner alternative.  Split divides a treap into two treaps based on a key threshold; merge combines two treaps into one.  Together, they form a powerful toolkit: insertion reduces to a split followed by two merges, and deletion reduces to two splits and a merge.  Both operations run in expected $O(\log n)$ time because the treap has expected $O(\log n)$ [height](priorities.md).

## Merge

`merge(L, R)` takes two treaps $L$ and $R$ where **every key in $L$ is less than every key in $R$** and returns a single treap containing all elements from both.

The algorithm exploits heap order on priorities:

1. If $L$ is empty, return $R$.  If $R$ is empty, return $L$.
2. If $L.\text{priority} > R.\text{priority}$ (using a max-heap convention), $L$'s root should be the root of the merged tree.  Recursively merge $L.\text{right}$ with $R$ and attach the result as $L$'s right child.
3. Otherwise, $R$'s root should be the root.  Recursively merge $L$ with $R.\text{left}$ and attach the result as $R$'s left child.

$$
\text{merge}(L, R) = \begin{cases}
R & \text{if } L = \text{nil} \\
L & \text{if } R = \text{nil} \\
(L.\text{key},\; L.\text{left},\; \text{merge}(L.\text{right}, R)) & \text{if } L.\text{pri} > R.\text{pri} \\
(R.\text{key},\; \text{merge}(L, R.\text{left}),\; R.\text{right}) & \text{otherwise}
\end{cases}
$$

Each recursive call descends one level in either $L$ or $R$, so the total work is $O(h_L + h_R) = O(\log n)$ expected.

## Split

`split(T, k)` divides treap $T$ into two treaps $(L, R)$ where $L$ contains all keys $\le k$ and $R$ contains all keys $> k$.

1. If $T$ is empty, return $(\text{nil}, \text{nil})$.
2. If $T.\text{key} \le k$, then $T$ and its left subtree belong to $L$.  Recursively split $T.\text{right}$ by $k$ to get $(L', R)$.  Set $T.\text{right} = L'$ and return $(T, R)$.
3. If $T.\text{key} > k$, then $T$ and its right subtree belong to $R$.  Recursively split $T.\text{left}$ by $k$ to get $(L, R')$.  Set $T.\text{left} = R'$ and return $(L, T)$.

Each recursive call descends one level, so split runs in $O(\log n)$ expected time.

## Implementation

```python
"""Treap split and merge operations."""

from __future__ import annotations

import random


# === Node Definition ===

class TreapNode:
    """Treap node with key and random priority."""

    def __init__(self, key: int):
        self.key = key
        self.priority = random.random()
        self.left: TreapNode | None = None
        self.right: TreapNode | None = None


# === Merge ===

def merge(left: TreapNode | None, right: TreapNode | None) -> TreapNode | None:
    """Merge two treaps where all keys in *left* < all keys in *right*."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        left.right = merge(left.right, right)
        return left
    else:
        right.left = merge(left, right.left)
        return right


# === Split ===

def split(node: TreapNode | None, key: int
          ) -> tuple[TreapNode | None, TreapNode | None]:
    """Split treap into (L, R) where L has keys <= key, R has keys > key."""
    if node is None:
        return None, None
    if node.key <= key:
        left, right = split(node.right, key)
        node.right = left
        return node, right
    else:
        left, right = split(node.left, key)
        node.left = right
        return left, node


# === Insert via Split + Merge ===

def insert(root: TreapNode | None, key: int) -> TreapNode:
    """Insert a key into the treap using split and merge."""
    left, right = split(root, key)
    new_node = TreapNode(key)
    return merge(merge(left, new_node), right)


# === Delete via Split + Merge ===

def delete(root: TreapNode | None, key: int) -> TreapNode | None:
    """Delete a key from the treap using split and merge."""
    left, right = split(root, key)
    left_without, _ = split(left, key - 1)
    return merge(left_without, right)


# === Inorder Traversal ===

def inorder(node: TreapNode | None) -> list[int]:
    """Collect keys in sorted order."""
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


# === Demonstration ===

if __name__ == "__main__":
    root: TreapNode | None = None
    for k in [5, 3, 8, 1, 4, 7, 9]:
        root = insert(root, k)
    print(f"After inserts: {inorder(root)}")  # [1, 3, 4, 5, 7, 8, 9]

    root = delete(root, 5)
    print(f"After delete 5: {inorder(root)}")  # [1, 3, 4, 7, 8, 9]
```

## Insert and Delete via Split/Merge

Using split and merge, insertion and deletion become simple compositions:

**Insert(T, k):**

1. Split $T$ at $k$ into $(L, R)$.
2. Create a new single-node treap $N$ with key $k$.
3. Return $\text{merge}(\text{merge}(L, N), R)$.

**Delete(T, k):**

1. Split $T$ at $k$ into $(L, R)$ (keys $\le k$ in $L$, keys $> k$ in $R$).
2. Split $L$ at $k - 1$ into $(L', M)$ (the node with key $k$ is the root of $M$).
3. Return $\text{merge}(L', R)$.

## Complexity

| Operation | Expected time |
|-----------|---------------|
| Merge | $O(\log n)$ |
| Split | $O(\log n)$ |
| Insert (via split/merge) | $O(\log n)$ |
| Delete (via split/merge) | $O(\log n)$ |

!!! tip "Advantages of split/merge over rotations"
    The split/merge approach avoids explicit rotation logic and naturally extends to **implicit treaps** (where keys are implicit array indices), enabling efficient sequence operations like reversals and range queries in $O(\log n)$ time.

## Reference

- Aragon, C. R., & Seidel, R. (1989). Randomized search trees. *30th IEEE Symposium on Foundations of Computer Science*, 540–545.
- Blelloch, G. E., & Reid-Miller, M. (1998). Fast set operations using treaps. *10th ACM Symposium on Parallel Algorithms and Architectures*, 16–26.
