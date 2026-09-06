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

## Exercises

**Exercise 1.**
Describe how to make a BST partially persistent using path copying. What is the time and space cost per insert operation?

??? success "Solution to Exercise 1"
    For partial persistence (old versions are read-only, only the latest is modified): on each insert, the new key is placed at a leaf position. Copy all nodes from the root to the new leaf, linking each copied node's unchanged child to the original subtree. The new root becomes the latest version; the old root still points to the old tree. Time per insert: $O(h)$ where $h$ is the tree height (same as ephemeral BST). Space per insert: $O(h)$ new nodes (one per level on the root-to-leaf path). For a balanced BST (AVL or red-black), $h = O(\log n)$, so both time and space per insert are $O(\log n)$. After $m$ insertions, total space is $O(n + m \log n)$. $\square$

---

**Exercise 2.**
Explain the difference between partial persistence and full persistence. Give an example where full persistence is necessary but partial persistence is insufficient.

??? success "Solution to Exercise 2"
    **Partial persistence**: all versions are queryable, but only the latest version can be modified. Versions form a linear sequence: $v_0 \to v_1 \to v_2 \to \cdots$. **Full persistence**: any version can be modified to produce a new version, creating a branching version tree. Example requiring full persistence: a version control system where a user checks out an old commit (version $v_3$) and makes a new edit, creating version $v_3'$ that branches off from $v_3$ rather than continuing from the latest $v_{10}$. With partial persistence, editing $v_3$ is impossible -- only $v_{10}$ can be modified. Another example: an undo tree in a text editor where the user can undo to any point and create a new branch of edits, rather than a linear undo/redo stack. $\square$

---

**Exercise 3.**
A persistent red-black tree inserts a key that triggers a rebalancing rotation. How many additional nodes must be copied compared to a simple path-copying insert without rotation?

??? success "Solution to Exercise 3"
    A red-black tree insert may trigger up to $O(\log n)$ recolorings (which are pointer-field changes on existing nodes) and at most 2 rotations. Each recoloring along the path requires copying the recolored node (which is already on the root-to-leaf path, so it would be copied anyway). Each rotation involves rearranging parent-child pointers among 2--3 nodes. If the rotated nodes are on the insertion path, they are already being copied. If a rotation involves a node's sibling (off the path), that sibling must also be copied -- at most 1 additional node per rotation. With at most 2 rotations, the additional nodes are at most 2. Total nodes copied: $O(\log n) + O(1) = O(\log n)$. The rotations do not change the asymptotic cost. $\square$

---

**Exercise 4.**
Design a persistent BST that supports the operation "count the number of keys in the range $[a, b]$ at version $v$" in $O(\log n)$ time. What augmentation is needed?

??? success "Solution to Exercise 4"
    Augment each node with a `size` field storing the number of nodes in its subtree. This is maintained during insertions: when copying nodes on the insertion path, update each copied node's size as the sum of its children's sizes plus 1. To count keys in $[a, b]$ at version $v$: define `rank(x, v)` as the number of keys $\le x$ in version $v$, computed by traversing the tree for version $v$ in $O(\log n)$ using the size fields. The count in $[a, b]$ is $\text{rank}(b, v) - \text{rank}(a-1, v)$, requiring two $O(\log n)$ traversals. The size augmentation adds $O(1)$ space per node and $O(1)$ time per node during updates, so the persistent BST's asymptotic complexities remain $O(\log n)$ per operation. $\square$

---

**Exercise 5.**
Compare persistent BSTs implemented via path copying versus fat nodes. Under what conditions does each approach use less total space after $m$ operations on a tree of $n$ elements?

??? success "Solution to Exercise 5"
    **Path copying**: each update copies $O(\log n)$ nodes. Total space after $m$ operations: $O(n + m \log n)$. Each version has its own root pointer. **Fat nodes**: each update stores $O(1)$ field changes in the modified nodes (with amortized $O(1)$ space per update if the node has bounded in-degree). Total space: $O(n + m)$. However, queries must search through modification logs at each node, adding $O(\log m)$ time per node access (binary search on timestamps). Path copying uses less space when $m$ is small ($m \ll n$), since the initial $O(n)$ dominates. Fat nodes use less space when $m$ is large ($m \gg n / \log n$), since $O(m)$ vs. $O(m \log n)$ becomes significant. The crossover is at $m \approx n / \log n$. Path copying is simpler to implement and has no query-time overhead, making it the preferred choice in competitive programming and most practical applications. $\square$
