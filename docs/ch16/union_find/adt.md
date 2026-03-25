# Disjoint Set ADT

Many problems require tracking a collection of non-overlapping groups and answering two questions efficiently: "Do elements $x$ and $y$ belong to the same group?" and "Merge the groups containing $x$ and $y$." These groups arise naturally as connected components in Kruskal's algorithm, equivalence classes in unification problems, and regions in image segmentation. The **disjoint set** (or **Union-Find**) abstract data type provides a clean interface for these operations.

## Definition

A **disjoint set data structure** maintains a collection $\mathcal{S} = \{S_1, S_2, \ldots, S_k\}$ of disjoint dynamic sets. Each set $S_i$ is identified by a **representative** -- a distinguished member of the set. The data structure supports three operations:

### MAKE-SET(x)

Creates a new set $\{x\}$ containing only element $x$. The representative of this singleton set is $x$ itself.

**Precondition**: $x$ does not already belong to any set in $\mathcal{S}$.

**Postcondition**: $\mathcal{S} \leftarrow \mathcal{S} \cup \{\{x\}\}$.

### FIND(x)

Returns the representative of the unique set containing $x$.

**Precondition**: $x$ belongs to some set in $\mathcal{S}$.

**Postcondition**: returns the representative of $S_i$ where $x \in S_i$. The collection $\mathcal{S}$ is unchanged (though the internal structure may be modified for efficiency).

**Key property**: $\text{FIND}(x) = \text{FIND}(y)$ if and only if $x$ and $y$ are in the same set.

### UNION(x, y)

Merges the two sets containing $x$ and $y$ into a single set. The representative of the merged set can be any member.

**Precondition**: $x$ and $y$ belong to (possibly the same) set(s) in $\mathcal{S}$.

**Postcondition**: if $x \in S_i$ and $y \in S_j$ with $S_i \ne S_j$, then $\mathcal{S} \leftarrow (\mathcal{S} \setminus \{S_i, S_j\}) \cup \{S_i \cup S_j\}$.

## Interface in Python

The following defines the ADT interface without committing to a specific internal representation. The subsequent pages in this section build up progressively faster implementations.

```python
"""
Disjoint Set abstract data type.

Defines the interface for Union-Find operations without
committing to a specific implementation strategy.
"""


# === ADT interface ===

class DisjointSetADT:
    """Abstract interface for the disjoint set data structure."""

    def __init__(self, n):
        """Create n singleton sets {0}, {1}, ..., {n-1}."""
        self.parent = list(range(n))  # each element is its own representative

    def find(self, x):
        """Return the representative of the set containing x."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a, b):
        """
        Merge the sets containing a and b.

        Returns True if a and b were in different sets,
        False if they were already in the same set.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        self.parent[root_b] = root_a  # naive: just point one root to the other
        return True

    def connected(self, a, b):
        """Check whether a and b are in the same set."""
        return self.find(a) == self.find(b)


# === Example ===

if __name__ == "__main__":
    uf = DisjointSetADT(5)
    print(f"Initially: 0 and 1 connected? {uf.connected(0, 1)}")

    uf.union(0, 1)
    uf.union(2, 3)
    print(f"After union(0,1) and union(2,3): 0 and 1 connected? {uf.connected(0, 1)}")
    print(f"0 and 3 connected? {uf.connected(0, 3)}")

    uf.union(1, 3)
    print(f"After union(1,3): 0 and 3 connected? {uf.connected(0, 3)}")
```

**Output:**
```
Initially: 0 and 1 connected? False
After union(0,1) and union(2,3): 0 and 1 connected? True
0 and 3 connected? False
After union(1,3): 0 and 3 connected? True
```

## Complexity of the Naive Implementation

The naive implementation above (no optimizations) has the following costs:

| Operation | Time |
|-----------|------|
| MAKE-SET | $O(1)$ |
| FIND | $O(n)$ worst case |
| UNION | $O(n)$ worst case (due to FIND) |

The worst case occurs when UNION always appends to the same chain, creating a linked list of depth $n$. The subsequent pages introduce two optimizations -- **union by rank** and **path compression** -- that reduce the amortized cost of FIND and UNION to $O(\alpha(n))$, where $\alpha$ is the inverse Ackermann function.

## Representation Choices

Two classical implementations exist:

| Approach | FIND | UNION | Concept |
|----------|------|-------|---------|
| **Quick Find** | $O(1)$ | $O(n)$ | Flat array: `id[x]` stores the set representative directly |
| **Quick Union** | $O(n)$ worst case | $O(n)$ worst case | Forest: `parent[x]` stores $x$'s parent; follow chain to root |

The next two pages examine these approaches in detail, followed by the optimizations that make the forest-based approach nearly optimal.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 21](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *JACM*, 22(2), 215--225.
