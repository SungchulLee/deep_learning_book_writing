# Euler Tour on Trees

The Euler tour technique flattens a rooted tree into a linear array so that every
subtree corresponds to a contiguous range. This reduction lets us answer subtree
queries, path queries, and LCA queries with standard range data structures.

## Intuition

A DFS traversal visits every node twice — once on entry and once on exit. Recording
these events produces a sequence of length $2n$ in which the entire subtree of any
node $v$ sits inside a contiguous segment. That segment can then be fed to a
Fenwick tree, segment tree, or sparse table.

## Definitions

Given a rooted tree $T$ with $n$ nodes, an **Euler tour** is constructed by a DFS
that records each node at entry and exit:

- $\text{tin}[v]$: the time when $v$ is first visited (entry).
- $\text{tout}[v]$: the time when $v$'s subtree is fully explored (exit).

The tour array $E$ of length $2n$ stores node labels in visitation order.

### Subtree Property

**Lemma.** Node $u$ is in the subtree of $v$ if and only if

$$
\text{tin}[v] \le \text{tin}[u] \le \text{tout}[u] \le \text{tout}[v]
$$

This means the subtree of $v$ maps to the range $[\text{tin}[v], \text{tout}[v]]$ in
the tour array.

### Flat Array Variant

For subtree-aggregate queries (sum, min, etc.), a common variant uses a **flat
array** $A$ of length $n$ where $A[\text{tin}[v]] = \text{value}(v)$. Then:

$$
\text{subtree\_sum}(v) = \sum_{i=\text{tin}[v]}^{\text{tout}[v]} A[i]
$$

computable in $O(\log n)$ with a Fenwick tree after $O(n)$ preprocessing.

## Euler Tour for LCA

A second variant records the node at every edge traversal (both down and up),
yielding a sequence of length $2n - 1$. The **depth array** $D$ stores the depth
of each entry.

**Theorem.** The LCA of nodes $u$ and $v$ is the node with minimum depth in

$$
D[\min(\text{tin}[u], \text{tin}[v]) \,..\, \max(\text{tin}[u], \text{tin}[v])]
$$

This reduces LCA to a **Range Minimum Query** (RMQ), solvable in
$O(n)$ preprocessing and $O(1)$ per query with a sparse table.

## Worked Example

Consider the tree rooted at $1$:

```
        1
       / \
      2   3
     / \
    4   5
```

DFS order (entry/exit):

| Event | Time | Node | Depth |
|-------|------|------|-------|
| Enter | 0 | 1 | 0 |
| Enter | 1 | 2 | 1 |
| Enter | 2 | 4 | 2 |
| Exit  | 3 | 4 | 2 |
| Enter | 4 | 5 | 2 |
| Exit  | 5 | 5 | 2 |
| Exit  | 6 | 2 | 1 |
| Enter | 7 | 3 | 1 |
| Exit  | 8 | 3 | 1 |
| Exit  | 9 | 1 | 0 |

- $\text{tin} = [0, 1, 7, 2, 4]$ and $\text{tout} = [9, 6, 8, 3, 5]$ (0-indexed nodes $1$-$5$).
- Subtree of node $2$: range $[1, 6]$, which covers nodes $2, 4, 5$.
- LCA of $4$ and $5$: minimum depth in tour positions $[2, 5]$ gives node $2$.

## Implementation

```python
"""Euler tour on a rooted tree with subtree and LCA support."""

import sys
from collections import defaultdict

# === Constants ===
sys.setrecursionlimit(300_000)


# === Build Euler tour ===
def euler_tour(adj, root, n):
    """Compute tin, tout, and the tour array via iterative DFS.

    Parameters
    ----------
    adj : dict[int, list[int]]
        Adjacency list (undirected tree).
    root : int
        Root node.
    n : int
        Number of nodes.

    Returns
    -------
    tin : list[int]
        Entry times.
    tout : list[int]
        Exit times.
    tour : list[int]
        Euler tour sequence.
    depth : list[int]
        Depth of each node.
    """
    tin = [0] * n
    tout = [0] * n
    depth = [0] * n
    tour = []
    timer = 0

    # Iterative DFS using (node, parent, entered) triples
    stack = [(root, -1, False)]
    while stack:
        v, par, entered = stack.pop()
        if entered:
            tout[v] = timer
            tour.append(v)
            timer += 1
            continue
        tin[v] = timer
        tour.append(v)
        timer += 1
        stack.append((v, par, True))
        for u in reversed(adj[v]):
            if u != par:
                depth[u] = depth[v] + 1
                stack.append((u, v, False))

    return tin, tout, tour, depth


# === Subtree query helper ===
def subtree_range(tin, tout, v):
    """Return the range [l, r] in the flat array for the subtree of v."""
    return tin[v], tout[v]


# === Demo ===
if __name__ == "__main__":
    n = 5
    adj = defaultdict(list)
    edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    tin, tout, tour, depth = euler_tour(adj, 0, n)
    print("tin: ", tin)
    print("tout:", tout)
    print("tour:", tour)
    print("depth:", depth)

    # Subtree of node 1
    l, r = subtree_range(tin, tout, 1)
    subtree_nodes = [nd for nd in range(n) if l <= tin[nd] <= r]
    print(f"Subtree of node 1: {subtree_nodes}")
```

## Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| Build Euler tour | $O(n)$ | $O(n)$ |
| Subtree query (with Fenwick) | $O(\log n)$ | $O(n)$ |
| LCA query (with sparse table) | $O(1)$ after $O(n)$ preprocessing | $O(n)$ |

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
- Bender, M. A. & Farach-Colton, M. (2000). *The LCA Problem Revisited*
