# Persistent Arrays

A standard array supports read and write in $O(1)$ time, but every write destroys the previous state. In many applications -- undo systems, version control, functional programming -- we need to keep old versions accessible. A **persistent array** augments the basic array so that every modification produces a new version while all older versions remain queryable.

## Persistence Models

There are two levels of persistence relevant to arrays:

- **Partial persistence**: old versions can be read but not modified. Only the latest version is mutable.
- **Full persistence**: every version (past or present) can be both read and modified, spawning new versions.

For an array of size $n$, the naive approach copies the entire array on every write, giving $O(n)$ per update and $O(n)$ extra space per version. The techniques below reduce these costs.

## Copy-on-Write Arrays

The simplest practical persistent array uses **copy-on-write** (COW). A new version initially shares the underlying storage with its parent. Only when a cell is modified does the system copy the affected portion.

For a flat array, a write to version $v$ at index $i$:

1. Copy the entire backing array to create version $v+1$.
2. Modify position $i$ in the new copy.

$$
T_{\text{write}} = O(n), \quad T_{\text{read}} = O(1), \quad S_{\text{per version}} = O(n)
$$

This is acceptable when writes are rare relative to reads, but the linear write cost motivates more sophisticated techniques.

## Fat-Node Arrays

Each array cell stores a list of (version, value) pairs sorted by version number. A read at version $v$ and index $i$ binary-searches the list at position $i$ for the latest entry whose version is at most $v$.

$$
T_{\text{read}}(i, v) = O(\log m_i)
$$

where $m_i$ is the number of modifications to cell $i$. A write appends a new (version, value) pair:

$$
T_{\text{write}} = O(1) \text{ amortized}, \quad S_{\text{per write}} = O(1)
$$

The total space across all versions is $O(n + M)$ where $M$ is the total number of writes.

## Backer's Trick

Backer's trick (also called the "rerooting" trick) achieves $O(1)$ amortized access for the most recently used version while maintaining full persistence. The idea stores the full array for one "current" version and records diffs (index, old-value) on edges of the version tree.

To access version $v$:

1. Walk the version tree from $v$ to the current root, collecting diffs.
2. Apply diffs in reverse to transform the root array into version $v$.
3. Reroot the version tree at $v$ so that subsequent accesses to $v$ are $O(1)$.

The amortized cost per access is $O(1)$ when accesses exhibit temporal locality (repeatedly querying the same or nearby versions).

## Complexity Summary

| Technique | Read | Write | Space per Version |
|---|---|---|---|
| Full copy | $O(1)$ | $O(n)$ | $O(n)$ |
| Fat nodes | $O(\log m_i)$ | $O(1)$ amort. | $O(1)$ amort. |
| Backer's trick | $O(1)$ amort.* | $O(1)$ amort. | $O(1)$ amort. |

\* Amortized $O(1)$ with rerooting; worst case $O(n)$ for a cold version.

## Implementation

```python
"""
Persistent Array -- fat-node implementation.

Each cell stores a history of (version, value) pairs. Reads at any
version use binary search; writes append a new entry in O(1).
"""

from bisect import bisect_right


# === Fat-Node Persistent Array ================================================

class PersistentArray:
    """Array supporting O(1) amortized writes and O(log m) reads per cell."""

    def __init__(self, initial: list):
        self.version = 0
        # Each cell stores a sorted list of (version, value)
        self._history: list[list[tuple[int, object]]] = [
            [(0, val)] for val in initial
        ]

    def read(self, index: int, version: int | None = None) -> object:
        """Read cell at *index* as of *version* (default: latest)."""
        if version is None:
            version = self.version
        cell = self._history[index]
        # Binary search for the last entry with ver <= version
        pos = bisect_right(cell, (version, float("inf"))) - 1
        if pos < 0:
            raise ValueError(f"No data at index {index} for version {version}")
        return cell[pos][1]

    def write(self, index: int, value: object) -> int:
        """Write *value* at *index*, returning the new version number."""
        self.version += 1
        self._history[index].append((self.version, value))
        return self.version

    def __len__(self) -> int:
        return len(self._history)


# === Main =====================================================================

if __name__ == "__main__":
    arr = PersistentArray([10, 20, 30, 40, 50])

    # Version 0: original
    print("v0:", [arr.read(i, 0) for i in range(5)])

    # Version 1: write index 2
    v1 = arr.write(2, 99)
    print(f"v{v1}:", [arr.read(i, v1) for i in range(5)])

    # Version 2: write index 0
    v2 = arr.write(0, 77)
    print(f"v{v2}:", [arr.read(i, v2) for i in range(5)])

    # Old versions still accessible
    print("v0 (again):", [arr.read(i, 0) for i in range(5)])
    print(f"v{v1} (again):", [arr.read(i, v1) for i in range(5)])
```

**Output:**

```
v0: [10, 20, 30, 40, 50]
v1: [10, 20, 99, 40, 50]
v2: [77, 20, 99, 40, 50]
v0 (again): [10, 20, 30, 40, 50]
v1 (again): [10, 20, 99, 40, 50]
```

The output confirms that writes create new versions without destroying previous states, and every historical version remains accessible.

## Reference

- Driscoll, J.R., Sarnak, N., Sleator, D.D., and Tarjan, R.E. "Making Data Structures Persistent." *JCSS*, 1989
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
