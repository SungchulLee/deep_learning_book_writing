# Persistent BSTs

A binary search tree (BST) supports search, insert, and delete in $O(h)$ time where $h$ is the tree height. In the standard (ephemeral) version, each insert or delete mutates the tree, destroying the previous shape. A **persistent BST** retains all prior versions so that any historical state can be queried or, in the fully persistent case, modified to spawn new versions.

## Why Path Copying Works for Trees

Trees have a key structural property that makes persistence efficient: every node has exactly one parent. When a node changes, only its ancestors need new copies -- the rest of the tree can be shared. For a balanced BST of $n$ nodes with height $h = O(\log n)$, an insert touches at most $h + 1$ nodes on a root-to-leaf path, so path copying creates only $O(\log n)$ new nodes per operation.

## Path-Copying Insert

To insert key $k$ into version $v$:

1. Walk from the root of version $v$ down to the insertion point, copying each visited node.
2. Create a new leaf for $k$.
3. Link the copied ancestors together and return the new root as version $v+1$.

The old root and all its shared subtrees remain intact, so version $v$ is still accessible.

**Time and space per insert:**

$$
T_{\text{insert}} = O(h), \quad S_{\text{insert}} = O(h)
$$

For a balanced BST, $h = O(\log n)$, giving $O(\log n)$ time and $O(\log n)$ extra space per version.

## Path-Copying Search

Searching version $v$ follows the standard BST search starting from the root of version $v$. No copying is needed since search does not modify the tree:

$$
T_{\text{search}} = O(h)
$$

## Path-Copying Delete

Deletion follows the same path-copying principle. Find the node to delete, copy the root-to-node path, and adjust pointers in the copies to remove the node (handling the standard BST deletion cases: leaf, one child, two children with in-order successor). The cost is:

$$
T_{\text{delete}} = O(h), \quad S_{\text{delete}} = O(h)
$$

## Complexity Summary

| Operation | Time | Extra Space |
|---|---|---|
| Search(version $v$, key $k$) | $O(\log n)$ | $O(1)$ |
| Insert(version $v$, key $k$) | $O(\log n)$ | $O(\log n)$ |
| Delete(version $v$, key $k$) | $O(\log n)$ | $O(\log n)$ |

These bounds assume a balanced BST. Without balancing, the worst case is $O(n)$ per operation.

## Implementation

```python
"""
Persistent BST -- path-copying implementation.

Each insert or delete creates a new root sharing unchanged subtrees
with previous versions. All historical versions remain accessible.
"""

from __future__ import annotations
from dataclasses import dataclass


# === Node =====================================================================

@dataclass
class Node:
    """Immutable BST node. Children may be shared across versions."""
    key: int
    left: Node | None = None
    right: Node | None = None


# === Persistent Operations ====================================================

def insert(root: Node | None, key: int) -> Node:
    """Return a new root with *key* inserted (path-copying)."""
    if root is None:
        return Node(key)
    if key < root.key:
        return Node(root.key, insert(root.left, key), root.right)
    elif key > root.key:
        return Node(root.key, root.left, insert(root.right, key))
    else:
        return root  # duplicate key: no change

def search(root: Node | None, key: int) -> bool:
    """Search for *key* starting from *root*."""
    if root is None:
        return False
    if key == root.key:
        return True
    elif key < root.key:
        return search(root.left, key)
    else:
        return search(root.right, key)

def inorder(root: Node | None) -> list[int]:
    """Return the in-order traversal as a list."""
    if root is None:
        return []
    return inorder(root.left) + [root.key] + inorder(root.right)


# === Main =====================================================================

if __name__ == "__main__":
    # Build version 0 (empty)
    versions: list[Node | None] = [None]

    # Insert keys, creating new versions
    for key in [5, 3, 7, 1, 4, 6, 8]:
        new_root = insert(versions[-1], key)
        versions.append(new_root)

    print(f"Number of versions: {len(versions)}")
    print(f"Latest (v{len(versions)-1}):  {inorder(versions[-1])}")
    print(f"After 3 inserts (v3): {inorder(versions[3])}")
    print(f"After 1 insert  (v1): {inorder(versions[1])}")
    print(f"Empty original  (v0): {inorder(versions[0])}")

    # Verify sharing: v7's right subtree root is the same object as v6's
    print(f"\nSharing check: v7.left is v6.left? {versions[7].left is versions[6].left}")
```

**Output:**

```
Number of versions: 8
Latest (v7):  [1, 3, 4, 5, 6, 7, 8]
After 3 inserts (v3): [3, 5, 7]
After 1 insert  (v1): [5]
Empty original  (v0): []

Sharing check: v7.left is v6.left? True
```

The sharing check confirms that path copying reuses unchanged subtrees: the left subtree of version 7 is the exact same object as the left subtree of version 6, since inserting 8 (which goes right) does not modify the left side.

## Reference

- Driscoll, J.R., Sarnak, N., Sleator, D.D., and Tarjan, R.E. "Making Data Structures Persistent." *JCSS*, 1989
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
