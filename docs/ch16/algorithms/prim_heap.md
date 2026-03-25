# Prim with Heap

The basic array-based Prim's algorithm runs in $O(V^2)$, which is efficient for dense graphs but wasteful for sparse ones. By replacing the linear scan for the minimum-key vertex with a binary min-heap (priority queue), the EXTRACT-MIN and DECREASE-KEY operations both take $O(\log V)$ time, bringing the total complexity down to $O(E \log V)$. This page presents the heap-based implementation in detail.

## Why a Heap Helps

In each iteration of Prim's algorithm, we need to:

1. **Find** the non-tree vertex with the smallest key (EXTRACT-MIN).
2. **Update** keys of neighbors when a lighter connecting edge is discovered (DECREASE-KEY).

With an unsorted array, EXTRACT-MIN takes $O(V)$ and DECREASE-KEY takes $O(1)$. Over $V$ extractions and up to $E$ decreases, the total is $O(V^2 + E) = O(V^2)$.

With a binary min-heap, both operations take $O(\log V)$. Over $V$ extractions and up to $E$ decreases, the total becomes $O((V + E) \log V) = O(E \log V)$ for connected graphs (since $E \ge V - 1$).

## Implementation

The Python implementation below uses `heapq`, which provides a min-heap. Since `heapq` does not support DECREASE-KEY directly, we use the **lazy deletion** strategy: push a new entry with the updated key and mark old entries as stale when the vertex is extracted.

```python
"""
Prim's MST algorithm using a binary min-heap.

Uses Python's heapq with lazy deletion to handle
the lack of a native DECREASE-KEY operation.
"""

import heapq
from collections import defaultdict


# === Graph representation ===

def build_adjacency_list(n, edges):
    """Build adjacency list from edge list."""
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# === Prim's algorithm with heap ===

def prim(n, edges, start=0):
    """
    Compute the MST using Prim's algorithm with a binary heap.

    Parameters
    ----------
    n : int
        Number of vertices (labeled 0 to n-1).
    edges : list of (u, v, w)
        Edge list with integer endpoints and numeric weight.
    start : int
        Starting vertex (default 0).

    Returns
    -------
    mst_edges : list of (u, v, w)
        Edges in the MST.
    total_weight : int or float
        Total weight of the MST.
    """
    adj = build_adjacency_list(n, edges)
    in_tree = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n

    key[start] = 0
    # heap entries: (key, vertex)
    heap = [(0, start)]
    mst_edges = []
    total_weight = 0

    while heap:
        k, u = heapq.heappop(heap)
        if in_tree[u]:
            continue  # lazy deletion: skip stale entries
        in_tree[u] = True
        total_weight += k
        if parent[u] != -1:
            mst_edges.append((parent[u], u, k))

        for v, w in adj[u]:
            if not in_tree[v] and w < key[v]:
                key[v] = w
                parent[v] = u
                heapq.heappush(heap, (w, v))

    return mst_edges, total_weight


# === Example ===

if __name__ == "__main__":
    #   0 ---4--- 1
    #   |  \      |
    #   1    3    2
    #   |      \  |
    #   2 ---5--- 3
    edges = [
        (0, 1, 4),
        (0, 2, 1),
        (1, 2, 3),
        (1, 3, 2),
        (2, 3, 5),
    ]
    mst, weight = prim(4, edges, start=0)
    print(f"MST edges: {mst}")
    print(f"Total weight: {weight}")
```

**Output:**
```
MST edges: [(0, 2, 1), (2, 1, 3), (1, 3, 2)]
Total weight: 6
```

## Lazy Deletion Explained

Python's `heapq` module does not provide a DECREASE-KEY operation. Instead of modifying existing heap entries, we push a new entry with the updated key. When we extract a vertex that has already been added to the tree (`in_tree[u] == True`), we simply skip it. This approach is called **lazy deletion**.

The trade-off: the heap may contain up to $O(E)$ entries instead of $O(V)$. Each heap operation costs $O(\log E) = O(\log V)$ (since $E \le V^2$), so the asymptotic complexity remains $O(E \log V)$.

## Complexity Analysis

**Time**: the algorithm performs at most $E$ `heappush` operations and at most $E$ `heappop` operations (one per stale or valid entry). Each costs $O(\log E) = O(\log V)$. Total:

$$
T(V, E) = O(E \log V)
$$

**Space**: $O(V + E)$ for the adjacency list and heap.

## Comparison with Fibonacci Heap

A Fibonacci heap supports DECREASE-KEY in $O(1)$ amortized time and EXTRACT-MIN in $O(\log V)$ amortized time. This reduces Prim's total to:

$$
T(V, E) = O(E + V \log V)
$$

For sparse graphs where $E = O(V)$, the Fibonacci heap gives $O(V \log V)$, improving over the binary heap's $O(V \log V)$ by a constant. For dense graphs where $E = \Theta(V^2)$, the Fibonacci heap gives $O(V^2)$, matching the simple array implementation. The Fibonacci heap is theoretically superior but rarely used in practice due to large constant factors and implementation complexity.

| Implementation | Time | Space | Practical speed |
|---------------|------|-------|-----------------|
| Array | $O(V^2)$ | $O(V)$ | Best for dense |
| Binary heap | $O(E \log V)$ | $O(V + E)$ | Best general-purpose |
| Fibonacci heap | $O(E + V \log V)$ | $O(V + E)$ | Rarely faster in practice |

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [1584. Min Cost to Connect All Points -- LeetCode](https://leetcode.com/problems/min-cost-to-connect-all-points/)
