# Template Libraries

Competitive programmers maintain personal libraries of pre-written, pre-tested code for common data structures and algorithms. During a contest, copying a trusted template eliminates implementation time and reduces the risk of bugs in well-understood components. This section covers what to include in a template library, how to organize it, and the design principles that make templates reliable under contest pressure.

## Why Templates Matter

In a typical 5-hour contest, implementation time dominates for problems whose algorithmic insight is straightforward. A segment tree implementation takes 15--20 minutes to write from scratch and is easy to get wrong under pressure. A tested template reduces this to 30 seconds of pasting and adapting.

Templates are valuable for:

- **Complex data structures**: Segment trees, Fenwick trees, disjoint set union, convex hull trick.
- **Standard algorithms**: Dijkstra, KMP, FFT, maximum flow.
- **Boilerplate**: Fast I/O setup, modular arithmetic, graph input parsing.

## Template Design Principles

### Correctness First

A template with a subtle bug is worse than no template at all, because the programmer trusts it and wastes time debugging elsewhere. Every template should be:

- **Verified** on at least 3--5 different problems.
- **Stress-tested** against a brute-force reference.
- **Edge-case tested** on empty inputs, single elements, and maximum sizes.

### Minimal and Readable

A contest template should be as short as possible while remaining readable. Avoid:

- Unnecessary generality (e.g., a segment tree that supports 10 different operations when you only ever use sum and max).
- Clever tricks that save one line but take a minute to understand under stress.
- Inheritance, virtual functions, or complex OOP -- flat classes or standalone functions are better.

### Easy to Adapt

A good template has clearly marked sections where the user plugs in problem-specific logic.

```python
"""
Template library for competitive programming.

Provides reusable implementations of common data structures
and algorithms, designed for quick adaptation during contests.
"""

import sys
from collections import defaultdict, deque
from heapq import heappush, heappop

# ===================================================================
# Fast I/O Template
# ===================================================================

input = sys.stdin.readline

def read_int():
    """Read a single integer."""
    return int(input())

def read_ints():
    """Read a line of space-separated integers."""
    return list(map(int, input().split()))

# ===================================================================
# Modular Arithmetic Template
# ===================================================================

MOD = 10**9 + 7

def mod_pow(base, exp, mod=MOD):
    """Fast modular exponentiation."""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        exp >>= 1
        base = base * base % mod
    return result

def mod_inv(a, mod=MOD):
    """Modular inverse using Fermat's little theorem (mod must be prime)."""
    return mod_pow(a, mod - 2, mod)

# ===================================================================
# Disjoint Set Union (Union-Find) Template
# ===================================================================

class DSU:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find root with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        """Union by rank. Return True if newly connected."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

# ===================================================================
# Segment Tree Template
# ===================================================================

class SegTree:
    """Segment tree for range queries and point updates.

    Customize the combine function and identity for different
    query types (sum, min, max, gcd, etc.).
    """

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        # Build
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, pos, value):
        """Set position pos to value (0-indexed)."""
        pos += self.n
        self.tree[pos] = value
        while pos > 1:
            pos >>= 1
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]

    def query(self, l, r):
        """Query the range [l, r) (0-indexed, exclusive end)."""
        res = 0
        l += self.n
        r += self.n
        while l < r:
            if l & 1:
                res += self.tree[l]
                l += 1
            if r & 1:
                r -= 1
                res += self.tree[r]
            l >>= 1
            r >>= 1
        return res

# ===================================================================
# Graph Template
# ===================================================================

def dijkstra(adj, src, n):
    """Dijkstra's algorithm with binary heap.

    Args:
        adj: adjacency list as dict of {node: [(neighbor, weight), ...]}
        src: source node
        n: number of nodes

    Returns:
        List of shortest distances from src.
    """
    dist = [float('inf')] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(heap, (dist[v], v))
    return dist

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    # DSU example
    dsu = DSU(5)
    dsu.union(0, 1)
    dsu.union(2, 3)
    dsu.union(1, 3)
    print(f"0 and 3 connected: {dsu.find(0) == dsu.find(3)}")

    # Segment tree example
    data = [1, 3, 5, 7, 9]
    st = SegTree(data)
    print(f"Sum [0,3): {st.query(0, 3)}")
    print(f"Sum [1,5): {st.query(1, 5)}")

    # Modular arithmetic example
    print(f"2^10 mod (10^9+7): {mod_pow(2, 10)}")
    print(f"Inverse of 3 mod (10^9+7): {mod_inv(3)}")
```

**Output:**
```
0 and 3 connected: True
Sum [0,3): 9
Sum [1,5): 24
2^10 mod (10^9+7): 1024
Inverse of 3 mod (10^9+7): 333333336
```

## Template Organization

Organize templates by category for quick lookup during contests:

| Category | Templates |
|---|---|
| Data structures | Segment tree, Fenwick tree, DSU, sparse table, trie |
| Graph | BFS, DFS, Dijkstra, Bellman--Ford, Kruskal, max flow |
| String | KMP, Z-algorithm, suffix array, Aho--Corasick |
| Math | Modular arithmetic, FFT/NTT, matrix exponentiation, sieve |
| Geometry | Convex hull, point-in-polygon, line intersection |
| Boilerplate | Fast I/O, random number generation, coordinate compression |

## Testing Your Templates

Before a contest season, verify each template:

1. **Unit test**: Does each function produce correct output for known inputs?
2. **Stress test**: Does it agree with a brute-force solution on random inputs?
3. **Performance test**: Does it handle maximum-size inputs within time limits?
4. **Integration test**: Does it compose correctly with other templates (e.g., segment tree inside Dijkstra)?

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
