# Quick Find

Quick Find is the simplest implementation of the Union-Find ADT. It prioritizes fast membership queries at the expense of slow merges. The core idea is to store the set representative directly in a flat array, so FIND is a single array lookup -- $O(1)$ -- but UNION must update all entries belonging to one set, costing $O(n)$ per merge. This trade-off makes Quick Find a useful starting point for understanding Union-Find, but impractical for large inputs.

## Data Structure

Maintain an array `id[]` of size $n$, where `id[x]` stores the representative (or "component ID") of the set containing $x$.

**Invariant**: two elements $x$ and $y$ are in the same set if and only if `id[x] == id[y]`.

**Initialization**: `id[x] = x` for all $x$, meaning each element is its own representative.

## Operations

### FIND(x)

Simply return `id[x]`. Since the representative is stored directly, this takes $O(1)$ time.

### UNION(x, y)

To merge the sets containing $x$ and $y$:

1. Let `id_x = id[x]` and `id_y = id[y]`.
2. If `id_x == id_y`, the elements are already in the same set -- done.
3. Otherwise, scan the entire array and change every entry equal to `id_y` to `id_x`.

This scan takes $O(n)$ time regardless of the set sizes.

## Implementation

```python
"""
Quick Find implementation of Union-Find.

Uses a flat array where id[x] stores the representative directly,
giving O(1) FIND but O(n) UNION.
"""


# === Quick Find ===

class QuickFind:
    """Union-Find with O(1) find and O(n) union."""

    def __init__(self, n):
        """Create n singleton sets {0}, {1}, ..., {n-1}."""
        self.id = list(range(n))
        self.count = n  # number of distinct sets

    def find(self, x):
        """Return the representative of x's set in O(1)."""
        return self.id[x]

    def union(self, a, b):
        """
        Merge the sets containing a and b.

        Scans the entire array to update all entries, taking O(n) time.
        Returns True if a and b were in different sets.
        """
        id_a = self.id[a]
        id_b = self.id[b]
        if id_a == id_b:
            return False
        # Change all entries with id_b to id_a
        for i in range(len(self.id)):
            if self.id[i] == id_b:
                self.id[i] = id_a
        self.count -= 1
        return True

    def connected(self, a, b):
        """Check whether a and b are in the same set in O(1)."""
        return self.id[a] == self.id[b]


# === Example ===

if __name__ == "__main__":
    qf = QuickFind(6)
    print(f"Initial id array: {qf.id}")

    qf.union(0, 1)
    print(f"After union(0,1): {qf.id}")

    qf.union(2, 3)
    print(f"After union(2,3): {qf.id}")

    qf.union(0, 3)
    print(f"After union(0,3): {qf.id}")

    print(f"connected(0,3): {qf.connected(0, 3)}")
    print(f"connected(0,4): {qf.connected(0, 4)}")
    print(f"Components: {qf.count}")
```

**Output:**
```
Initial id array: [0, 1, 2, 3, 4, 5]
After union(0,1): [0, 0, 2, 3, 4, 5]
After union(2,3): [0, 0, 2, 2, 4, 5]
After union(0,3): [0, 0, 0, 0, 4, 5]
connected(0,3): True
connected(0,4): False
Components: 3
```

## Execution Trace

Starting with 6 elements, the array evolves as follows:

| Operation | id[0] | id[1] | id[2] | id[3] | id[4] | id[5] | Components |
|-----------|-------|-------|-------|-------|-------|-------|------------|
| Init | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| union(0,1) | 0 | 0 | 2 | 3 | 4 | 5 | 5 |
| union(2,3) | 0 | 0 | 2 | 2 | 4 | 5 | 4 |
| union(0,3) | 0 | 0 | 0 | 0 | 4 | 5 | 3 |

When `union(0,3)` executes: `id_a = id[0] = 0` and `id_b = id[3] = 2`. The algorithm scans the entire array and changes every `2` to `0`, affecting both index 2 and index 3.

## Complexity Analysis

| Operation | Time | Explanation |
|-----------|------|-------------|
| MAKE-SET | $O(1)$ | Initialize one array entry |
| FIND | $O(1)$ | Single array lookup |
| UNION | $O(n)$ | Full array scan |

For a sequence of $n - 1$ UNION operations (to merge all $n$ elements into one set), the total cost is

$$
\sum_{i=1}^{n-1} O(n) = O(n^2)
$$

This quadratic cost makes Quick Find unsuitable for large inputs. Kruskal's algorithm on a graph with $V$ vertices and $E$ edges would take $O(E \cdot V)$ with Quick Find -- much worse than the $O(E \log E)$ achieved with an optimized Union-Find.

## Limitations

- **UNION is too expensive**: the $O(n)$ per-operation cost dominates for any non-trivial sequence of merges.
- **No incremental improvement**: unlike the forest-based approaches (Quick Union), repeated operations do not improve the structure.
- **Poor scalability**: $O(n^2)$ for $n$ merges makes it impractical for graphs with more than a few thousand vertices.

The next page introduces Quick Union, which uses a tree (forest) structure to potentially reduce UNION cost, at the expense of making FIND slower in the worst case.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 21](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. & Wayne, K. *Algorithms*, 4th ed., Section 1.5.
