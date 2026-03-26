# Traveling Salesman Problem

Given a list of cities and the cost of traveling between each pair, can a
salesman visit every city exactly once and return home for at most a given
budget?  The Traveling Salesman Problem (TSP) is one of the most studied
NP-complete problems, with applications spanning logistics, circuit design,
and genome sequencing.

## Problem Definition

**Input.**  A complete graph $G = (V, E)$ on $n$ vertices with a weight
function $w : E \to \mathbb{Z}_{\ge 0}$, and a budget $B \in \mathbb{Z}_{\ge 0}$.

**Question.**  Does there exist a Hamiltonian cycle (a cycle visiting every
vertex exactly once) of total weight at most $B$?

!!! example "Small Instance"
    Four cities with distances:

    | | A | B | C | D |
    |---|---|---|---|---|
    | A | 0 | 10 | 15 | 20 |
    | B | 10 | 0 | 35 | 25 |
    | C | 15 | 35 | 0 | 30 |
    | D | 20 | 25 | 30 | 0 |

    The tour $A \to B \to D \to C \to A$ costs $10 + 25 + 30 + 15 = 80$.
    With $B = 80$, the answer is **YES**.

## TSP Is in NP

A certificate is a permutation $\pi$ of the vertices.  The verifier computes

$$
\sum_{i=1}^{n-1} w(\pi(i), \pi(i+1)) + w(\pi(n), \pi(1))
$$

and checks whether the total is at most $B$.  This takes $O(n)$ time, so
TSP $\in$ NP.

## NP-Hardness via Reduction from Hamiltonian Cycle

The **Hamiltonian Cycle** problem (given a graph, does it contain a cycle
visiting every vertex exactly once?) is NP-complete.  We reduce it to TSP.

### Construction

Given a graph $G = (V, E)$ with $n$ vertices, build a complete weighted graph
$G'$ on the same vertex set:

$$
w(u, v) =
\begin{cases}
1 & \text{if } (u, v) \in E \\
2 & \text{if } (u, v) \notin E
\end{cases}
$$

Set the budget $B = n$.

### Correctness

- **If $G$ has a Hamiltonian cycle,** that cycle uses only edges of weight $1$
  in $G'$, giving total cost $n \le B$.  Answer: YES.
- **If TSP answers YES on $(G', B)$,** the tour has cost at most $n$.  Since
  each of the $n$ edges costs at least $1$, every edge must cost exactly $1$,
  meaning every edge belongs to $E$.  Thus $G$ has a Hamiltonian cycle.

The reduction runs in $O(n^2)$ time.

!!! note "Decision vs. Optimization"
    The decision version ("is there a tour of cost $\le B$?") is NP-complete.
    The optimization version ("find the minimum-cost tour") is NP-hard.  A
    polynomial-time algorithm for the decision version would solve the
    optimization version via binary search on $B$.

## Exact Algorithms

### Brute Force

Enumerate all $(n - 1)!$ permutations.  Running time: $O(n!)$.

### Held--Karp Algorithm (Dynamic Programming)

Define $\text{dp}[S][j]$ as the minimum cost of a path starting at vertex $0$,
visiting every vertex in the subset $S \subseteq V$ exactly once, and ending at
vertex $j \in S$.

**Base case:**

$$
\text{dp}[\{0, j\}][j] = w(0, j) \quad \text{for all } j \ne 0
$$

**Recurrence:**

$$
\text{dp}[S][j] = \min_{k \in S \setminus \{j\}} \bigl(\text{dp}[S \setminus \{j\}][k] + w(k, j)\bigr)
$$

**Answer:**

$$
\min_{j \ne 0} \bigl(\text{dp}[V][j] + w(j, 0)\bigr)
$$

Time complexity: $O(2^n \cdot n^2)$.  Space: $O(2^n \cdot n)$.

```python
"""
Held-Karp algorithm for TSP using bitmask DP.

Time : O(2^n * n^2)
Space: O(2^n * n)
"""

import math


# === Held-Karp DP ===
def tsp_held_karp(dist: list[list[int]]) -> int:
    """Return the minimum Hamiltonian cycle cost starting from vertex 0."""
    n = len(dist)
    full_mask = (1 << n) - 1
    dp = [[math.inf] * n for _ in range(1 << n)]
    dp[1][0] = 0  # start at vertex 0

    for mask in range(1, 1 << n):
        for u in range(n):
            if dp[mask][u] == math.inf:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                cost = dp[mask][u] + dist[u][v]
                if cost < dp[new_mask][v]:
                    dp[new_mask][v] = cost

    return min(dp[full_mask][u] + dist[u][0] for u in range(1, n))


# === Example ===
if __name__ == "__main__":
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    print(f"Minimum tour cost: {tsp_held_karp(dist)}")  # 80
```

## Approximation

For the **metric TSP** (where edge weights satisfy the triangle inequality),
Christofides' algorithm achieves a $\frac{3}{2}$-approximation.  No
polynomial-time algorithm can achieve a ratio better than $\frac{123}{122}$
for metric TSP unless P = NP.

For general (non-metric) TSP, no constant-factor approximation exists unless
P = NP.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 34.
- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
