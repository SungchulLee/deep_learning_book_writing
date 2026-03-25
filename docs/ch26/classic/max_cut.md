# Max-Cut Approximation

Given an undirected graph, the **Max-Cut** problem asks for a partition of the
vertices into two sets that maximizes the number (or total weight) of edges
crossing the partition. Max-Cut is NP-hard, yet a simple randomized algorithm
achieves a 1/2-approximation, and a greedy local-search algorithm matches
this guarantee deterministically.

## Problem Definition

Given an undirected graph $G = (V, E)$ with edge weights $w_e \ge 0$, find a
partition $(S, \bar{S})$ of $V$ maximizing

$$
w(S, \bar{S}) = \sum_{\substack{(u,v) \in E \\ u \in S,\, v \in \bar{S}}} w_{uv}
$$

For unweighted graphs, $w_e = 1$ for all edges, and the objective counts the
number of crossing edges.

## Randomized 1/2-Approximation

**Intuition.** Place each vertex independently into $S$ with probability 1/2.
Each edge crosses the cut with probability exactly 1/2, so on average half the
total weight is captured.

!!! tip "Theorem"
    The randomized algorithm achieves $\mathbb{E}[w(S, \bar{S})] = W/2$, where
    $W = \sum_{e \in E} w_e$. Since $\text{OPT} \le W$, this gives a
    1/2-approximation in expectation.

**Proof.** For each edge $e = (u, v)$, define indicator $X_e = 1$ if $e$
crosses the cut. Then $\Pr[X_e = 1] = 2 \cdot \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{2}$.
By linearity of expectation,

$$
\mathbb{E}[w(S, \bar{S})] = \sum_{e \in E} w_e \cdot \Pr[X_e = 1]
= \frac{1}{2} \sum_{e \in E} w_e = \frac{W}{2}
\ge \frac{\text{OPT}}{2} \qquad \blacksquare
$$

## Greedy Local Search

**Intuition.** Start with an arbitrary partition. If moving any single vertex
to the other side increases the cut, do it. Repeat until no improving move
exists.

**Algorithm:**

1. Start with $S = \emptyset$, $\bar{S} = V$
2. For each vertex $v$, compute the gain from moving $v$: the cut-weight
   increase if $v$ switches sides
3. If any vertex has positive gain, move the one with maximum gain
4. Repeat until no positive-gain move exists

!!! tip "Theorem"
    The local-search algorithm returns a cut of weight at least $W/2$.

**Proof.** At termination, for every vertex $v$, moving $v$ does not increase
the cut. Let $d_{\text{in}}(v)$ be the weight of edges from $v$ to vertices on
the same side, and $d_{\text{out}}(v)$ be the weight to the other side.
The no-improvement condition gives $d_{\text{out}}(v) \ge d_{\text{in}}(v)$
for all $v$. Summing over all vertices:

$$
\sum_v d_{\text{out}}(v) \ge \sum_v d_{\text{in}}(v)
$$

Since each crossing edge contributes to $d_{\text{out}}$ for both endpoints
and each non-crossing edge contributes to $d_{\text{in}}$ for both endpoints:

$$
2 \cdot w(S, \bar{S}) \ge 2 \cdot (W - w(S, \bar{S}))
$$

$$
w(S, \bar{S}) \ge W/2 \ge \text{OPT}/2 \qquad \blacksquare
$$

## The Goemans-Williamson Algorithm

The celebrated SDP-based algorithm of Goemans and Williamson (1995) achieves
an approximation ratio of $\alpha_{\text{GW}} \approx 0.878$, which is optimal
assuming the Unique Games Conjecture. The algorithm relaxes the integer program
to a semidefinite program, then rounds the solution using random hyperplane
rounding.

## Implementation

```python
"""
Max-Cut: randomized and greedy local-search approximation algorithms.
"""

import random


# === Randomized 1/2-approximation =============================================

def max_cut_random(n, edges):
    """
    Randomized Max-Cut: each vertex goes to S with probability 1/2.

    Returns (cut_weight, set_S).
    Expected approximation ratio: 1/2.
    """
    S = set()
    for v in range(n):
        if random.random() < 0.5:
            S.add(v)

    cut = sum(w for u, v, w in edges if (u in S) != (v in S))
    return cut, S


# === Greedy Local Search ======================================================

def max_cut_local_search(n, edges):
    """
    Local-search Max-Cut: repeatedly move vertices to increase cut.

    Returns (cut_weight, set_S).
    Guaranteed approximation ratio: 1/2.
    """
    # Build adjacency with weights
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Start: all vertices in bar_S
    in_S = [False] * n

    improved = True
    while improved:
        improved = False
        for v in range(n):
            gain = 0
            for u, w in adj[v]:
                if in_S[v] == in_S[u]:
                    gain += w  # Would become crossing
                else:
                    gain -= w  # Would stop crossing
            if gain > 0:
                in_S[v] = not in_S[v]
                improved = True

    S = {v for v in range(n) if in_S[v]}
    cut = sum(w for u, v, w in edges if in_S[u] != in_S[v])
    return cut, S


# === Demo =====================================================================

if __name__ == "__main__":
    n = 5
    edges = [
        (0, 1, 3), (0, 2, 2), (1, 2, 1),
        (1, 3, 4), (2, 4, 5), (3, 4, 2),
    ]
    W = sum(w for _, _, w in edges)

    random.seed(42)
    cut_rand, S_rand = max_cut_random(n, edges)
    print(f"Random:       cut={cut_rand}, S={S_rand}")

    cut_local, S_local = max_cut_local_search(n, edges)
    print(f"Local search: cut={cut_local}, S={S_local}")
    print(f"Total weight: {W}, lower bound (W/2): {W / 2}")
```

## Summary

| Algorithm | Ratio | Time |
|---|---|---|
| Random | $1/2$ (expected) | $O(m)$ |
| Local Search | $1/2$ | $O(nm)$ per pass |
| Goemans-Williamson | $\approx 0.878$ | Polynomial (SDP) |

## Reference

- Goemans, M. X. and Williamson, D. P. "Improved Approximation Algorithms for
  Maximum Cut." *JACM*, 1995.
- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001.
