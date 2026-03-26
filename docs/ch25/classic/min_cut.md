# Randomized Min-Cut (Karger's Algorithm)

Finding the minimum cut of an undirected graph — the smallest set of edges
whose removal disconnects the graph — is a classical problem in network
reliability and graph partitioning. Deterministic algorithms based on
max-flow run in $O(n^3)$ or better, but Karger's randomized contraction
algorithm achieves the same goal with a remarkably simple approach:
repeatedly contract random edges until only two vertices remain.

## Problem Statement

Given an undirected multigraph $G = (V, E)$ with $n = |V|$ vertices, a
**cut** is a partition of $V$ into two non-empty sets $S$ and $\bar{S}$.
The **size** of the cut is the number of edges crossing the partition.
The **minimum cut** (min-cut) is the cut of smallest size.

## Karger's Contraction Algorithm

### Edge Contraction

**Contracting** an edge $(u, v)$ merges $u$ and $v$ into a single
super-vertex. All edges between $u$ and $v$ are removed, and edges
to other vertices are preserved (creating a multigraph with possible
parallel edges). Self-loops are removed.

### Algorithm

1. While the graph has more than 2 vertices:
    - Choose an edge uniformly at random.
    - Contract it.
2. The remaining 2 super-vertices define a cut. Return the number
   of edges between them.

### One Run

Each run produces a cut (not necessarily the minimum). The key insight is
that the probability of finding the min-cut in a single run is surprisingly
high.

## Probability Analysis

**Theorem.** A single run of Karger's algorithm returns the min-cut with
probability at least $\binom{n}{2}^{-1} = \frac{2}{n(n-1)}$.

**Proof.** Let $C$ be a min-cut of size $k$. At any step with $t$ vertices
remaining, the graph has at least $kt/2$ edges (since every vertex has
degree $\ge k$). The probability of contracting a min-cut edge is at most:

$$
\frac{k}{kt/2} = \frac{2}{t}
$$

The probability of *not* contracting any min-cut edge through all $n - 2$
contractions is:

$$
\prod_{t=n}^{3} \left(1 - \frac{2}{t}\right)
= \prod_{t=3}^{n} \frac{t-2}{t}
= \frac{1 \cdot 2}{(n-1) \cdot n}
= \frac{2}{n(n-1)}
$$

$\square$

## Boosting Success Probability

Run the algorithm $T$ times and return the smallest cut found. The
probability of *missing* the min-cut in all $T$ runs is:

$$
\left(1 - \frac{2}{n(n-1)}\right)^T \le e^{-2T / n(n-1)}
$$

Setting $T = \binom{n}{2} \ln n = \frac{n(n-1)}{2} \ln n$ gives failure
probability at most $1/n$:

$$
e^{-\ln n} = \frac{1}{n}
$$

**Total time:** $O(n^2 \cdot T) = O(n^4 \log n)$.

## Karger-Stein Improvement

The Karger-Stein algorithm improves the runtime by observing that early
contractions are safer (low probability of cutting a min-cut edge). It
contracts down to $\lceil n/\sqrt{2} \rceil + 1$ vertices, then branches
into two independent recursive calls.

$$
T(n) = O(n^2 \log^3 n)
$$

This is faster than both the naive $O(n^4 \log n)$ and deterministic
max-flow approaches for dense graphs.

## Implementation

```python
"""
Karger's randomized min-cut algorithm.

Finds the minimum cut of an undirected graph by repeatedly
contracting random edges.
"""

import random
import copy


# === Graph Representation ===

def make_graph(n, edges):
    """Create an adjacency-list multigraph.

    Returns a dict mapping vertex -> list of neighbors (with repeats
    for parallel edges).
    """
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# === Edge Contraction ===

def contract(adj, u, v):
    """Contract edge (u, v) by merging v into u.

    Removes self-loops and redirects all of v's edges to u.
    """
    # Move all v's neighbors to u
    for w in adj[v]:
        if w != u:
            adj[u].append(w)
            adj[w] = [u if x == v else x for x in adj[w]]

    # Remove self-loops
    adj[u] = [x for x in adj[u] if x != u]

    # Remove v
    del adj[v]


# === Karger's Algorithm (Single Run) ===

def karger_once(adj):
    """One run of Karger's contraction algorithm.

    Returns the cut size (number of edges between the final 2 vertices).
    """
    adj = copy.deepcopy(adj)

    while len(adj) > 2:
        # Pick a random edge
        u = random.choice(list(adj.keys()))
        if not adj[u]:
            break
        v = random.choice(adj[u])
        contract(adj, u, v)

    # The cut size is the number of edges between the two remaining vertices
    vertices = list(adj.keys())
    if len(vertices) < 2:
        return float("inf")
    return len(adj[vertices[0]])


# === Repeated Karger ===

def karger_min_cut(n, edges, trials=None):
    """Find the min-cut by running Karger's algorithm multiple times.

    Args:
        n: number of vertices.
        edges: list of (u, v) edge tuples.
        trials: number of repetitions (default: n^2 * ln(n) / 2).

    Returns:
        The minimum cut size found.
    """
    import math
    if trials is None:
        trials = max(int(n * n * math.log(n) / 2), 10)

    adj = make_graph(n, edges)
    min_cut = float("inf")

    for _ in range(trials):
        cut = karger_once(adj)
        min_cut = min(min_cut, cut)

    return min_cut


# === Main ===

if __name__ == "__main__":
    random.seed(42)

    # Simple graph with known min-cut = 2
    #   0 -- 1
    #   |    |
    #   2 -- 3
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    n = 4
    print(f"Graph: {n} vertices, edges = {edges}")
    print(f"Min-cut (50 trials): {karger_min_cut(n, edges, trials=50)}")

    # Graph with min-cut = 1 (bridge)
    #   0 -- 1 -- 2
    edges2 = [(0, 1), (1, 2)]
    print(f"\nBridge graph: min-cut = {karger_min_cut(3, edges2, trials=50)}")

    # Denser graph
    edges3 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    print(f"\nDenser graph: min-cut = {karger_min_cut(5, edges3, trials=100)}")
```

**Output:**
```
Graph: 4 vertices, edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
Min-cut (50 trials): 2

Bridge graph: min-cut = 1

Denser graph: min-cut = 2
```

## Complexity Summary

| Algorithm | Time | Success probability |
|---|---|---|
| Single Karger run | $O(n^2)$ | $\ge 2/(n(n-1))$ |
| Repeated Karger | $O(n^4 \log n)$ | $\ge 1 - 1/n$ |
| Karger-Stein | $O(n^2 \log^3 n)$ | $\ge 1 - 1/n$ |
| Deterministic (Stoer-Wagner) | $O(nm + n^2 \log n)$ | $1$ |

## Reference

- Karger, D. R. "Global Min-Cuts in RNC, and Other Ramifications of a Simple Min-Cut Algorithm." *SODA*, 1993.
- Karger, D. R. & Stein, C. "A New Approach to the Minimum Cut Problem." *JACM*, 1996.
