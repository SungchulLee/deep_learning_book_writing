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

## Exercises

**Exercise 1.**
Explain the tradeoff between full copying and path copying for making arrays persistent. What are the time and space complexities of each approach?

??? success "Solution to Exercise 1"
    **Full copying**: on each write, copy the entire array of size $n$. Read any version in $O(1)$, write creates a new version in $O(n)$ time and $O(n)$ space. After $m$ writes, total space is $O(nm)$. **Path copying via a balanced binary tree over the array indices**: represent the array as a complete binary tree of height $\lceil \log_2 n \rceil$ with values at the leaves. A write copies only the $O(\log n)$ nodes on the root-to-leaf path, sharing the rest. Read requires traversing $O(\log n)$ nodes. After $m$ writes, total space is $O(n + m \log n)$. The tradeoff: full copying has $O(1)$ read but $O(n)$ write; path copying has $O(\log n)$ read and write. Path copying is preferable when $m$ is large relative to $n$. $\square$

---

**Exercise 2.**
Describe how to implement a persistent array using a segment tree. What are the complexities for point read, point write, and range query across versions?

??? success "Solution to Exercise 2"
    Build a segment tree over the $n$ array positions. For a point write at index $i$ in version $v$: create a new root for version $v+1$, copy only the $O(\log n)$ nodes on the path from root to leaf $i$, pointing to shared children from version $v$ for unchanged subtrees. Point read at version $v$: traverse the segment tree root for version $v$ down to the leaf in $O(\log n)$. Range query $[l, r]$ at version $v$: traverse the version-$v$ tree in $O(\log n)$, combining results from at most $O(\log n)$ nodes. Each new version creates $O(\log n)$ new nodes. After $m$ updates, total space is $O(n + m \log n)$. $\square$

---

**Exercise 3.**
A persistent array stores stock prices over time. Version $t$ represents the price array at time $t$. Design a query that, given two time points $t_1$ and $t_2$ and a stock index $i$, returns the price change. What is the time complexity?

??? success "Solution to Exercise 3"
    Store the price array as a persistent segment tree. Each write (price update) creates a new version. To compute the price change for stock $i$ between times $t_1$ and $t_2$: perform a point read on version $t_1$ at index $i$ to get $p_1$, and a point read on version $t_2$ at index $i$ to get $p_2$. Return $p_2 - p_1$. Time: $O(\log n)$ per read, so $O(\log n)$ total (two reads, each $O(\log n)$). Space: $O(n + T \log n)$ where $T$ is the total number of updates across all time steps. This approach naturally supports historical queries without maintaining separate snapshots. $\square$

---

**Exercise 4.**
Prove that a persistent array implemented via path copying on a balanced binary tree has amortized $O(\log n)$ space per update.

??? success "Solution to Exercise 4"
    Each update modifies one leaf and copies the $O(\log n)$ nodes on the path from that leaf to the root. All other nodes are shared with the previous version via pointers. Therefore, each update creates exactly $\lceil \log_2 n \rceil + 1$ new nodes (one per tree level). The initial version requires $O(n)$ nodes (the full tree). After $m$ updates, the total number of nodes is $O(n + m \log n)$. The space per update is $O(\log n)$ (not amortized -- it is worst-case per update). This is optimal for tree-based approaches because any path from root to leaf has length $\Theta(\log n)$, and a single-position update must modify at least the nodes on this path to create a new root distinguishable from the old root. $\square$

---

**Exercise 5.**
Compare persistent arrays with the "copy-on-write" technique used in modern operating systems for process forking. What structural similarity and what practical difference exist?

??? success "Solution to Exercise 5"
    **Structural similarity**: both share unchanged data between versions and copy only what is modified. In copy-on-write (CoW) forking, the parent and child process share all memory pages, and a page is duplicated only when either process writes to it. In a persistent array via path copying, old and new versions share all unmodified subtrees, and only the path to the modified element is duplicated. Both achieve $O(\text{changes})$ space per version rather than $O(n)$. **Practical difference**: CoW operates at the page granularity (typically 4 KB), while persistent arrays operate at the element granularity. CoW is managed by the OS/hardware MMU transparently, while persistent arrays are an explicit data structure. CoW is not fully persistent -- after a page is copied and modified, the original page in the child is lost unless explicitly preserved. Persistent arrays retain all versions permanently by design. $\square$
