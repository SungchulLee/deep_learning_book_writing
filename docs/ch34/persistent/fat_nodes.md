# Fat Nodes

When making a data structure persistent, the most direct approach is to store every modification inside the nodes themselves. Instead of creating new nodes on each update, each node becomes "fat" -- it carries a timestamped log of all values its fields have ever held. This technique, introduced by Driscoll, Sarnak, Sleator, and Tarjan (1989), achieves partial persistence with $O(1)$ amortized extra space per modification.

## Intuition

Consider a linked structure (BST, linked list, etc.) where an update changes a pointer or a data field in some node. Rather than copying the node, we append a modification record $(t, \text{field}, \text{value})$ to the node, where $t$ is the version number. To read a field at version $v$, we scan the modification list for the latest entry with timestamp at most $v$.

## Fat-Node Structure

Each node stores:

- **Original fields**: the key, data, and pointers from version 0.
- **Modification list**: a chronologically sorted sequence of $(t, \text{field}, \text{value})$ triples recording every change.

For a node with $p$ pointer fields (e.g., $p = 2$ for a BST with left/right), the modification list stores changes to any of the $p + d$ fields ($d$ data fields).

## Read Operation

To read field $f$ of node $x$ at version $v$:

1. Scan the modification list of $x$ for entries affecting field $f$.
2. Return the value from the latest entry with timestamp $\le v$.
3. If no such entry exists, return the original value from version 0.

With binary search on the sorted modification list:

$$
T_{\text{read}} = O(\log m_x)
$$

where $m_x$ is the number of modifications to node $x$.

## Write Operation

To write field $f$ of node $x$ at the current version $t$:

1. Append $(t, f, \text{new\_value})$ to the modification list of $x$.

$$
T_{\text{write}} = O(1) \text{ amortized}, \quad S_{\text{write}} = O(1) \text{ amortized}
$$

## Bounded Fat Nodes

An important optimization limits each node's modification list to a fixed capacity $c$ (typically $c = 2p$ where $p$ is the number of pointer fields). When the list is full:

1. Create a new copy of the node with all current field values baked in.
2. Clear the modification list of the new node.
3. Update the parent's pointer to reference the new node (which itself may trigger a cascading copy).

This bounded variant guarantees $O(1)$ amortized space per modification via an amortized argument: each node can absorb $c$ modifications before splitting, so the cost of the split is charged across those $c$ writes.

$$
S_{\text{amortized per write}} = O(1)
$$

## Complexity Summary

| Operation | Time | Extra Space |
|---|---|---|
| Read(field, version $v$) | $O(\log m)$ | -- |
| Write(field, value) | $O(1)$ amort. | $O(1)$ amort. |
| Bounded split | $O(p)$ worst case | $O(p)$ |

Here $m$ is the number of modifications to the queried node, and $p$ is the number of pointer fields per node.

!!! tip "Fat nodes vs path copying"
    Fat nodes use less space than path copying ($O(1)$ vs $O(\log n)$ per update for balanced trees) but have slower reads ($O(\log m)$ vs $O(\log n)$). Choose fat nodes when updates are frequent and reads are rare; choose path copying when reads dominate.

## Implementation

```python
"""
Fat-Node Persistent BST -- partial persistence.

Each BST node stores a modification log. Reading a field at any
historical version scans the log; writing appends a new entry.
"""

from __future__ import annotations
from bisect import bisect_right


# === Fat Node =================================================================

class FatNode:
    """BST node with modification history for partial persistence."""

    def __init__(self, key: int, version: int = 0):
        self.key = key
        self._creation = version
        # Modification log: list of (version, field_name, value)
        self._mods: list[tuple[int, str, object]] = []
        # Original field values (version 0)
        self._left: FatNode | None = None
        self._right: FatNode | None = None

    def get_field(self, field: str, version: int) -> object:
        """Read *field* as of *version*."""
        original = getattr(self, f"_{field}")
        best_val = original
        best_ver = self._creation
        for ver, fname, val in self._mods:
            if fname == field and best_ver < ver <= version:
                best_val = val
                best_ver = ver
        return best_val

    def set_field(self, field: str, value: object, version: int) -> None:
        """Record a modification to *field* at *version*."""
        self._mods.append((version, field, value))


# === Persistent BST ===========================================================

class PersistentBST:
    """Partially persistent BST using fat nodes."""

    def __init__(self):
        self.roots: list[FatNode | None] = [None]  # roots[v] = root at version v
        self.current_version = 0

    def insert(self, key: int) -> int:
        """Insert *key*, returning the new version number."""
        self.current_version += 1
        v = self.current_version
        if self.roots[-1] is None:
            new_root = FatNode(key, v)
            self.roots.append(new_root)
        else:
            self.roots.append(self.roots[-1])
            self._insert_at(self.roots[-1], key, v)
        return v

    def _insert_at(self, node: FatNode, key: int, version: int) -> None:
        """Recursively insert into the fat-node tree."""
        if key < node.key:
            left = node.get_field("left", version)
            if left is None:
                node.set_field("left", FatNode(key, version), version)
            else:
                self._insert_at(left, key, version)
        elif key > node.key:
            right = node.get_field("right", version)
            if right is None:
                node.set_field("right", FatNode(key, version), version)
            else:
                self._insert_at(right, key, version)

    def inorder(self, version: int | None = None) -> list[int]:
        """In-order traversal at the given version."""
        if version is None:
            version = self.current_version
        result: list[int] = []
        self._inorder(self.roots[version], version, result)
        return result

    def _inorder(self, node: FatNode | None, version: int,
                 result: list[int]) -> None:
        if node is None or node._creation > version:
            return
        left = node.get_field("left", version)
        self._inorder(left, version, result)
        result.append(node.key)
        right = node.get_field("right", version)
        self._inorder(right, version, result)


# === Main =====================================================================

if __name__ == "__main__":
    bst = PersistentBST()
    for key in [5, 3, 7, 1, 4]:
        bst.insert(key)

    print(f"v0 (empty): {bst.inorder(0)}")
    print(f"v1 (add 5): {bst.inorder(1)}")
    print(f"v3 (add 7): {bst.inorder(3)}")
    print(f"v5 (add 4): {bst.inorder(5)}")
```

**Output:**

```
v0 (empty): []
v1 (add 5): [5]
v3 (add 7): [3, 5, 7]
v5 (add 4): [1, 3, 4, 5, 7]
```

Each version reflects the tree state after the corresponding insertions, and earlier versions remain accessible through the modification logs stored in each fat node.

## Reference

- Driscoll, J.R., Sarnak, N., Sleator, D.D., and Tarjan, R.E. "Making Data Structures Persistent." *JCSS*, 1989
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
