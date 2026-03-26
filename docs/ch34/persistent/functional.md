# Functional Data Structures

In imperative programming, data structures are typically mutable: an insert modifies the structure in place. **Functional data structures** are immutable by design -- every operation returns a new version, and the old version persists automatically. This makes them inherently persistent without any extra bookkeeping, and they are naturally thread-safe since no mutation can occur.

## Structural Sharing

The key insight that makes functional data structures efficient is **structural sharing**: a new version reuses most of the old version's memory, copying only the nodes that change. For a balanced tree of $n$ nodes, an insert touches $O(\log n)$ nodes on a root-to-leaf path, so the new version shares all but $O(\log n)$ nodes with its predecessor.

$$
S_{\text{per operation}} = O(\log n) \text{ for balanced trees}
$$

Without structural sharing, every operation would require $O(n)$ copying, making persistence impractical.

## Functional Lists (Cons Lists)

The simplest functional data structure is the **cons list**, built from pairs (head, tail) where the tail points to the rest of the list. Prepending a new element creates a single new cons cell pointing to the existing list:

$$
T_{\text{prepend}} = O(1), \quad S_{\text{prepend}} = O(1)
$$

Multiple versions can share the same tail, forming a tree of list versions.

!!! warning "Append is expensive"
    Appending to a cons list requires copying the entire spine: $O(n)$ time and space. Functional code therefore favors prepend over append, reversing the final result if order matters.

## Functional Trees

A functional balanced BST (e.g., a functional red-black tree or AVL tree) supports insert, delete, and search while maintaining immutability:

| Operation | Time | Extra Space |
|---|---|---|
| Search | $O(\log n)$ | $O(1)$ |
| Insert | $O(\log n)$ | $O(\log n)$ |
| Delete | $O(\log n)$ | $O(\log n)$ |

Each insert or delete copies the root-to-modification path and shares the remaining subtrees. This is identical to the path-copying technique but arises naturally from the programming style rather than as an explicit persistence mechanism.

## Amortization and Persistence

A subtle issue arises when combining amortized data structures with persistence. In ephemeral structures, amortized analysis works because expensive operations are paid for by prior cheap operations. In persistent structures, a single version can be the source of multiple future operations, potentially replaying the expensive step without the accumulated "credit."

Okasaki (1998) addresses this with **lazy evaluation**: deferred computations are memoized so that an expensive step is performed at most once regardless of how many times the version is accessed.

$$
T_{\text{amortized}}^{\text{persistent}} = T_{\text{amortized}}^{\text{ephemeral}} \text{ with lazy evaluation}
$$

Without laziness, persistent versions of amortized structures may lose their amortized bounds.

## Functional Queues

A standard queue requires $O(1)$ enqueue and dequeue, which is trivial with mutable pointers but non-obvious with immutability. The **banker's queue** uses two lists:

- **Front list** $F$: dequeue takes from the head.
- **Rear list** $R$: enqueue prepends to $R$.

When $F$ is empty, reverse $R$ and swap: $F' = \text{reverse}(R)$, $R' = []$. The reversal costs $O(n)$ but happens at most once per $n$ enqueues, giving $O(1)$ amortized cost per operation.

With lazy evaluation, the reversal is deferred and memoized, preserving the $O(1)$ amortized bound even under persistence.

## Implementation

```python
"""
Functional Data Structures -- cons list and functional BST.

All operations return new versions without mutating existing ones.
Structural sharing keeps space overhead proportional to the path length.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator


# === Cons List ================================================================

@dataclass(frozen=True)
class ConsList:
    """Immutable singly linked list (cons cell)."""
    head: object
    tail: ConsList | None = None

    def prepend(self, value: object) -> ConsList:
        """Return a new list with *value* at the front."""
        return ConsList(value, self)

    def __iter__(self) -> Iterator:
        node = self
        while node is not None:
            yield node.head
            node = node.tail

    def to_list(self) -> list:
        return list(self)


# === Functional BST ===========================================================

@dataclass(frozen=True)
class FuncNode:
    """Immutable BST node."""
    key: int
    left: FuncNode | None = None
    right: FuncNode | None = None


def func_insert(root: FuncNode | None, key: int) -> FuncNode:
    """Return a new tree with *key* inserted (path copying)."""
    if root is None:
        return FuncNode(key)
    if key < root.key:
        return FuncNode(root.key, func_insert(root.left, key), root.right)
    elif key > root.key:
        return FuncNode(root.key, root.left, func_insert(root.right, key))
    return root  # key already present


def func_inorder(root: FuncNode | None) -> list[int]:
    """In-order traversal of an immutable BST."""
    if root is None:
        return []
    return func_inorder(root.left) + [root.key] + func_inorder(root.right)


# === Main =====================================================================

if __name__ == "__main__":
    # Cons list: multiple versions sharing tails
    v1 = ConsList(3)
    v2 = v1.prepend(2)
    v3 = v2.prepend(1)
    v4 = v1.prepend(9)  # branches from v1, not v3

    print("v1:", v1.to_list())
    print("v2:", v2.to_list())
    print("v3:", v3.to_list())
    print("v4:", v4.to_list())
    print(f"v3.tail.tail is v4.tail? {v3.tail.tail is v4.tail}  (shared tail)")

    # Functional BST: versions with structural sharing
    print()
    trees = [None]
    for k in [5, 3, 7, 2, 4]:
        trees.append(func_insert(trees[-1], k))

    for i, t in enumerate(trees):
        print(f"tree v{i}: {func_inorder(t)}")
```

**Output:**

```
v1: [3]
v2: [2, 3]
v3: [1, 2, 3]
v4: [9, 3]
v3.tail.tail is v4.tail? True  (shared tail)

tree v0: []
tree v1: [5]
tree v2: [3, 5]
tree v3: [3, 5, 7]
tree v4: [2, 3, 5, 7]
tree v5: [2, 3, 4, 5, 7]
```

The cons list demonstrates structural sharing: `v3` and `v4` both share the tail node containing `3`. The functional BST shows that every historical version remains intact after each insert.

## Reference

- Okasaki, C. *Purely Functional Data Structures.* Cambridge University Press, 1998
- Driscoll, J.R., Sarnak, N., Sleator, D.D., and Tarjan, R.E. "Making Data Structures Persistent." *JCSS*, 1989
