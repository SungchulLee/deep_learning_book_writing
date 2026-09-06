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

## Exercises

**Exercise 1.**
Explain the fat node technique for partial persistence. How does a query at version $t$ find the correct field value at a given node?

??? success "Solution to Exercise 1"
    In the fat node technique, each node stores a list of (timestamp, value) pairs for each mutable field. When a field is modified at version $t$, the pair $(t, \text{new\_value})$ is appended to that field's log. The original value is stored with timestamp 0 (or the creation time). To query a field at version $t$: binary search the log for the largest timestamp $\le t$, returning the corresponding value. This gives the field's value as it was at version $t$. Since the log is sorted by timestamp (modifications are applied in version order for partial persistence), binary search takes $O(\log m)$ where $m$ is the number of modifications to that field. For a BST search at version $t$, each node access costs $O(\log m)$ instead of $O(1)$, making total search time $O(h \log m)$. $\square$

---

**Exercise 2.**
Prove that fat nodes achieve $O(1)$ amortized space per modification for a data structure where each node has bounded in-degree (at most $p$ pointers pointing to it).

??? success "Solution to Exercise 2"
    The DSST framework assigns each node $2p$ extra modification slots (where $p$ is the max in-degree). When a field of a node is modified, the change is written into one of its free slots. If the node's slots are full, the node is "copied out": a new node is created with the latest values, and all pointers to the old node are updated to point to the new node. These pointer updates are themselves modifications to the pointing nodes, handled recursively. The amortized analysis uses a potential function $\Phi = c \cdot (\text{total occupied slots})$ for an appropriate constant $c$. Each modification fills one slot (cost 1, potential increases by $c$). A copy-out empties $2p$ slots (potential decreases by $2pc$) but requires updating $\le p$ pointers (cost $\le p$, each filling one slot). Choosing $c \ge 1$ makes the amortized cost $O(1)$ per modification. $\square$

---

**Exercise 3.**
Compare the query-time overhead of fat nodes versus path copying for reading version $t$ of a BST with $n$ nodes and $m$ total modifications.

??? success "Solution to Exercise 3"
    **Path copying**: each version has its own root pointer. A query at version $t$ traverses the tree from version $t$'s root, following pointers that lead to shared or copied nodes. Each node access is $O(1)$ (just follow a pointer). Total query time: $O(h)$ where $h$ is the tree height. **Fat nodes**: a single tree structure exists, with modification logs at each node. A query at version $t$ must binary-search the log at each visited node to find the correct field values. Each node access costs $O(\log m_v)$ where $m_v$ is the number of modifications at that node. Total query time: $O(h \cdot \log m)$ in the worst case (if modifications are concentrated). In practice, $m_v$ varies across nodes, and most nodes have few modifications, so the average cost is closer to $O(h)$. Path copying has strictly better query performance at the cost of higher space. $\square$

---

**Exercise 4.**
A fat-node persistent linked list stores 1000 versions, each with a single modification. What is the total space usage? How does it compare to storing 1000 separate copies of the list?

??? success "Solution to Exercise 4"
    Let the list have $n$ nodes. Each of the 1000 modifications adds one (timestamp, value) pair to one node's log. Total space: $O(n)$ for the original list + $O(1000)$ for the modification entries = $O(n + 1000)$. If $n = 10{,}000$, total space is roughly $11{,}000$ units. Storing 1000 separate copies would require $1000 \times n = 10{,}000{,}000$ units -- a 1000x increase. Fat nodes are dramatically more space-efficient when modifications are sparse relative to the data structure size. The tradeoff is that each node access now requires searching the modification log, adding $O(\log 1000) \approx 10$ overhead per access. $\square$

---

**Exercise 5.**
Can fat nodes support full persistence (modifying any version, not just the latest)? What fundamental difficulty arises, and how did Driscoll et al. address it?

??? success "Solution to Exercise 5"
    Fat nodes as described support only partial persistence: modifications must be applied in timestamp order to the latest version. For full persistence, a modification to version $t$ (which may not be the latest) creates a branch. The difficulty is that the modification log at each node is no longer linearly ordered by time -- the version history forms a tree, and binary search on a linear log does not work. Driscoll et al. addressed this by augmenting the fat-node technique with a "node-splitting" strategy and an access function that navigates the version tree. Each node's modification log is organized by the version tree structure, and field lookup at version $t$ requires finding the correct branch. The amortized space bound increases: $O(1)$ per modification still holds, but the constant factor is larger. The query overhead also increases because navigating the version DAG adds complexity. In practice, path copying is preferred for full persistence due to its simpler implementation. $\square$
