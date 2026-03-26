# Quick Union

In the quick-find approach to disjoint sets, every element stores its root directly, making `find` run in $O(1)$ but `union` in $O(n)$ because every member of one component must be updated. Quick union flips this trade-off: each element stores only its **parent** in a rooted tree, so `union` simply links one root to another in $O(1)$, while `find` walks up the tree to the root. The price is that `find` now takes time proportional to the tree height, which can degrade to $O(n)$ in the worst case without further optimizations.

## Tree Representation

Quick union represents each disjoint set as a **rooted tree**. Every element $x$ maintains a parent pointer $\text{parent}[x]$. The root of a tree is the unique element satisfying $\text{parent}[r] = r$. Two elements belong to the same set if and only if they share the same root.

Initially, every element is its own root:

$$
\text{parent}[x] = x \quad \text{for all } x \in \{0, 1, \dots, n-1\}
$$

## Core Operations

### Find

To determine which set an element $x$ belongs to, follow parent pointers from $x$ until reaching a root:

$$
\text{find}(x) = \begin{cases} x & \text{if } \text{parent}[x] = x \\ \text{find}(\text{parent}[x]) & \text{otherwise} \end{cases}
$$

The cost of `find` is $O(d)$ where $d$ is the depth of $x$ in its tree.

### Union

To merge the sets containing elements $a$ and $b$, find their respective roots and make one root point to the other:

$$
\text{union}(a, b): \quad r_a = \text{find}(a),\; r_b = \text{find}(b),\; \text{parent}[r_b] \leftarrow r_a
$$

This takes $O(1)$ beyond the cost of the two `find` calls.

## Worst-Case Analysis

Without any balancing strategy, a sequence of $n - 1$ unions can produce a degenerate chain of height $n - 1$. For example, performing $\text{union}(0,1),\, \text{union}(1,2),\, \dots,\, \text{union}(n{-}2, n{-}1)$ in the naive version creates a single path $0 \to 1 \to 2 \to \cdots \to n{-}1$. A subsequent `find` on element $0$ traverses all $n$ nodes.

Over a sequence of $m$ operations on $n$ elements, the naive quick union therefore has worst-case cost $O(mn)$.

## Optimizations

Two key optimizations bring the amortized cost per operation down to nearly $O(1)$:

**Union by rank.** Attach the shorter tree under the root of the taller tree. Each node stores a rank (an upper bound on its height). When two roots have equal rank, the new root's rank increases by one. This guarantees tree height is at most $O(\log n)$.

**Path compression.** During `find`, make every visited node point directly to the root. This flattens the tree for future queries. A lighter variant called **path splitting** makes each visited node point to its grandparent, achieving similar amortized performance with a simpler loop.

With both optimizations, any sequence of $m$ operations on $n$ elements runs in $O(m \, \alpha(n))$ time, where $\alpha$ is the inverse Ackermann function, which grows so slowly that it is effectively constant ($\alpha(n) \le 4$ for all practical $n$).

## Implementation

The following implementation combines union by rank with path splitting, the form most commonly used in practice.

```python
"""
Quick Union with union-by-rank and path splitting.

Demonstrates the quick union approach to disjoint sets, where each
element stores a parent pointer and find walks up the tree to the root.
Two optimizations keep trees nearly flat:
  - Union by rank: attach the shorter tree under the taller root.
  - Path splitting: during find, re-link each node to its grandparent.
"""

# === Union-Find Class ===

class UnionFind:
    """Disjoint-set data structure using quick union."""

    def __init__(self, n: int):
        """Create n singleton sets {0}, {1}, ..., {n-1}."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Return the root of the set containing x (with path splitting)."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path splitting
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets containing a and b. Return False if already same set."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Union by rank: attach smaller-rank tree under larger-rank root
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """Check whether a and b belong to the same set."""
        return self.find(a) == self.find(b)


# === Demonstration ===

if __name__ == "__main__":
    uf = UnionFind(6)

    # Build three pairs
    print(uf.union(0, 1))  # True  — {0,1}
    print(uf.union(2, 3))  # True  — {2,3}
    print(uf.union(4, 5))  # True  — {4,5}

    # Merge two pairs
    print(uf.union(1, 3))  # True  — {0,1,2,3}

    # Query connectivity
    print(f"0 and 3 connected: {uf.connected(0, 3)}")  # True
    print(f"0 and 4 connected: {uf.connected(0, 4)}")  # False
```

**Output:**

```
True
True
True
True
0 and 3 connected: True
0 and 4 connected: False
```

The first three unions each create a new pair. The fourth union merges the sets containing $1$ and $3$, joining $\{0,1\}$ and $\{2,3\}$ into a single component. The queries confirm that $0$ and $3$ are now connected while $0$ and $4$ remain in separate components.

## Complexity Summary

| Operation | Naive Quick Union | With Rank + Path Compression |
|-----------|:-----------------:|:----------------------------:|
| `find`    | $O(n)$            | $O(\alpha(n))$ amortized     |
| `union`   | $O(n)$            | $O(\alpha(n))$ amortized     |
| Space     | $O(n)$            | $O(n)$                       |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 21: Data Structures for Disjoint Sets.
- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215--225.
