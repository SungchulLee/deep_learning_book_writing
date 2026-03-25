# Persistent Segment Trees

Sometimes we need to answer queries about past states of a data structure. For example, given an array that undergoes a sequence of updates, we might want to query the sum over a range as it was after the $k$-th update. A **persistent segment tree** retains all previous versions of the tree, allowing queries on any historical version — without copying the entire tree for each update.

## Path Copying

The key technique is **path copying**: when updating a single position, only the nodes along the root-to-leaf path change. A persistent update creates new copies of just those $O(\log n)$ nodes, while unchanged subtrees are shared with the previous version. Each version is identified by its root pointer.

!!! note "Structural Sharing"
    If the tree has $n$ leaves and height $h = O(\log n)$, each update creates exactly $h + 1$ new nodes. All other nodes are shared with the previous version. Over $q$ updates, the total space is $O(n + q \log n)$, compared to $O(nq)$ if we copied the entire tree each time.

## Node-Based Representation

Unlike an array-based segment tree, a persistent segment tree uses **pointer-based nodes** because array indices would conflict across versions. Each node stores:

- `value`: the aggregate for its range.
- `left`, `right`: references to child nodes (which may belong to older versions).

## Persistent Update

To create version $v+1$ from version $v$ by setting position $i$ to a new value:

1. Start from the root of version $v$.
2. If the current node is a leaf: create a new leaf with the updated value.
3. Otherwise: create a new internal node. If $i$ falls in the left child's range, recursively update the left child and reuse the old right child (and vice versa).
4. The new node's value is recomputed from its (possibly new) children.
5. The root of this new path becomes the root of version $v+1$.

## Implementation

```python
"""
Persistent segment tree using path copying.

Each update creates a new version by copying only the O(log n)
nodes on the root-to-leaf path, sharing all unchanged subtrees
with previous versions.
"""


# === Node Definition ===

class Node:
    """Immutable node in the persistent segment tree."""

    __slots__ = ('value', 'left', 'right')

    def __init__(self, value: int = 0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


# === Persistent Segment Tree ===

class PersistentSegTree:
    """Persistent segment tree supporting point updates and range queries.

    Each update produces a new root (version). Queries can be
    performed on any version.
    """

    def __init__(self, data: list):
        """Build version 0 from the input array."""
        self.n = len(data)
        self.roots = []
        if self.n > 0:
            root = self._build(data, 0, self.n - 1)
            self.roots.append(root)

    def _build(self, data: list, lo: int, hi: int) -> Node:
        """Recursively build the initial tree."""
        if lo == hi:
            return Node(value=data[lo])
        mid = (lo + hi) // 2
        left = self._build(data, lo, mid)
        right = self._build(data, mid + 1, hi)
        return Node(value=left.value + right.value, left=left, right=right)

    def update(self, version: int, idx: int, val: int) -> int:
        """Create a new version by setting position idx to val.

        Returns the index of the new version.
        """
        new_root = self._update(self.roots[version], 0, self.n - 1, idx, val)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _update(self, node: Node, lo: int, hi: int,
                idx: int, val: int) -> Node:
        """Path-copy update: create new nodes along the path to idx."""
        if lo == hi:
            return Node(value=val)
        mid = (lo + hi) // 2
        if idx <= mid:
            new_left = self._update(node.left, lo, mid, idx, val)
            return Node(value=new_left.value + node.right.value,
                        left=new_left, right=node.right)
        else:
            new_right = self._update(node.right, mid + 1, hi, idx, val)
            return Node(value=node.left.value + new_right.value,
                        left=node.left, right=new_right)

    def query(self, version: int, l: int, r: int) -> int:
        """Range sum query on a specific version."""
        return self._query(self.roots[version], 0, self.n - 1, l, r)

    def _query(self, node: Node, lo: int, hi: int,
               l: int, r: int) -> int:
        """Recursive range query."""
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return node.value
        mid = (lo + hi) // 2
        return (self._query(node.left, lo, mid, l, r)
                + self._query(node.right, mid + 1, hi, l, r))

    def version_count(self) -> int:
        """Return the number of versions stored."""
        return len(self.roots)


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    pst = PersistentSegTree(data)

    print(f"Version 0 (original): {data}")
    print(f"  Sum [0,4]: {pst.query(0, 0, 4)}")
    print(f"  Sum [1,3]: {pst.query(0, 1, 3)}")
    print()

    # Version 1: set position 2 to 50
    v1 = pst.update(0, 2, 50)
    print(f"Version 1 (set a[2]=50):")
    print(f"  Sum [0,4]: {pst.query(v1, 0, 4)}")
    print(f"  Sum [1,3]: {pst.query(v1, 1, 3)}")
    print()

    # Version 2: set position 0 to 100 (based on version 1)
    v2 = pst.update(v1, 0, 100)
    print(f"Version 2 (set a[0]=100, from v1):")
    print(f"  Sum [0,4]: {pst.query(v2, 0, 4)}")
    print()

    # Original version is still accessible
    print(f"Version 0 still intact:")
    print(f"  Sum [0,4]: {pst.query(0, 0, 4)}")
    print()

    print(f"Total versions: {pst.version_count()}")
```

**Output:**
```
Version 0 (original): [1, 3, 5, 7, 9]
  Sum [0,4]: 25
  Sum [1,3]: 15

Version 1 (set a[2]=50):
  Sum [0,4]: 70
  Sum [1,3]: 60

Version 2 (set a[0]=100, from v1):
  Sum [0,4]: 169

Version 0 still intact:
  Sum [0,4]: 25

Total versions: 3
```

## Complexity

| Operation | Time | Space per version |
|-----------|------|-------------------|
| Build (version 0) | $O(n)$ | $O(n)$ nodes |
| Point update (new version) | $O(\log n)$ | $O(\log n)$ new nodes |
| Range query | $O(\log n)$ | $O(1)$ |
| Total after $q$ updates | — | $O(n + q \log n)$ nodes |

The space efficiency comes from structural sharing: each update creates only $O(\log n)$ new nodes.

## Applications

- **K-th smallest in a range.** Build a persistent segment tree on sorted values. Version $i$ represents the state after inserting the first $i$ elements. The difference between two versions answers k-th smallest queries.
- **Version control.** Any application needing undo/redo on aggregate data.
- **Offline queries.** Answer queries that refer to different time points in the update sequence.

## Reference

- Driscoll, J. R., Sarnak, N., Sleator, D. D., & Tarjan, R. E. (1989). Making Data Structures Persistent. *Journal of Computer and System Sciences*, 38(1), 86-124.
