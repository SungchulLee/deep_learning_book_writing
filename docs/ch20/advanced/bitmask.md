# Bitmask DP

When a problem asks to optimize over all subsets of a small set, brute-force enumeration seems unavoidable. Bitmask DP tames the exponential blow-up by encoding each subset as a binary integer and using bitwise operations to transition between states. Because an $n$-bit integer can represent any subset of $\{0, 1, \ldots, n-1\}$, the DP table is simply an array of size $2^n$, and adding or removing an element reduces to flipping a single bit.

## Subset Representation

A subset $S \subseteq \{0, 1, \ldots, n-1\}$ is represented by an $n$-bit integer called a **mask**, where bit $i$ is 1 if and only if $i \in S$.

| Subset | Binary | Mask |
|--------|--------|------|
| $\emptyset$ | `0000` | 0 |
| $\{0\}$ | `0001` | 1 |
| $\{1, 3\}$ | `1010` | 10 |
| $\{0, 1, 2, 3\}$ | `1111` | 15 |

Common bitwise operations for subset manipulation:

| Operation | Expression | Meaning |
|-----------|-----------|---------|
| Check if $i \in S$ | `mask & (1 << i)` | Test bit $i$ |
| Add $i$ to $S$ | `mask | (1 << i)` | Set bit $i$ |
| Remove $i$ from $S$ | `mask & ~(1 << i)` | Clear bit $i$ |
| Toggle $i$ | `mask ^ (1 << i)` | Flip bit $i$ |
| Full set | `(1 << n) - 1` | All $n$ bits set |
| Subset size | `bin(mask).count('1')` | Population count |

!!! tip "When to use bitmask DP"
    Bitmask DP is the right choice when (1) the problem involves subsets of a ground set, (2) the ground set has at most about 20 elements, and (3) the optimal solution for a subset can be built from optimal solutions for smaller subsets.

## General Framework

Define a state $dp[\text{mask}]$ (or $dp[\text{mask}][i]$ when the last element chosen matters) where mask encodes which elements have been selected. The key insight is that subset inclusion provides a natural partial order: every mask with $k$ bits set depends only on masks with $k-1$ bits set, so iterating masks in increasing order respects all dependencies.

**Generic recurrence** (minimization variant):

$$
dp[\text{mask}] = \min_{i \in \text{mask}} \bigl( dp[\text{mask} \setminus \{i\}] + \text{cost}(i, \text{mask}) \bigr)
$$

Here $\text{mask} \setminus \{i\}$ denotes removing element $i$ from the subset, implemented as `mask & ~(1 << i)`.

**Base case.** $dp[0] = 0$ (empty subset, zero cost).

**Time complexity.** $O(2^n \cdot n)$ --- iterate over all $2^n$ masks and check up to $n$ bits in each.

**Space complexity.** $O(2^n)$ or $O(2^n \cdot n)$ if an additional dimension tracks the last element.

## Example: Traveling Salesman Problem

The TSP asks for the minimum-cost Hamiltonian cycle visiting all $n$ cities exactly once. This formulation, due to Held and Karp (1962), works for both symmetric and asymmetric distance matrices.

**State.** $dp[\text{mask}][i]$ = minimum cost to visit exactly the cities in mask, ending at city $i$.

**Recurrence.** For each city $j \in \text{mask}$ with $j \neq i$:

$$
dp[\text{mask}][i] = \min_{j \in \text{mask} \setminus \{i\}} \bigl( dp[\text{mask} \setminus \{i\}][j] + \text{dist}(j, i) \bigr)
$$

**Base case.** $dp[\{0\}][0] = 0$ (start at city 0).

**Answer.** $\min_{i} \bigl( dp[(1 \ll n) - 1][i] + \text{dist}(i, 0) \bigr)$, completing the cycle back to city 0.

## Example: Assignment Problem

Given $n$ workers and $n$ tasks with cost matrix $C$, assign each worker to exactly one task to minimize total cost.

**State.** $dp[\text{mask}]$ = minimum cost to assign tasks in mask to the first $|\text{mask}|$ workers.

**Recurrence.** Let $k = |\text{mask}|$ (number of tasks assigned so far):

$$
dp[\text{mask}] = \min_{j \in \text{mask}} \bigl( dp[\text{mask} \setminus \{j\}] + C[k-1][j] \bigr)
$$

**Base case.** $dp[0] = 0$.

## Implementation

```python
"""
Bitmask DP: Traveling Salesman Problem and Assignment Problem.

Demonstrates the Held-Karp algorithm for TSP and a subset-based
approach for the assignment problem, both using bitmask DP.
"""

import math


# ===================================================================
# TSP via bitmask DP (Held-Karp algorithm)
# ===================================================================
def tsp_bitmask(dist: list[list[int]]) -> int:
    """Solve TSP using bitmask DP.

    Parameters
    ----------
    dist : list[list[int]]
        Distance matrix where dist[i][j] is the cost from city i to j.
        Need not be symmetric.

    Returns
    -------
    int
        Minimum cost of a Hamiltonian cycle starting and ending at city 0.
    """
    n = len(dist)
    full = (1 << n) - 1
    INF = math.inf

    # dp[mask][i] = min cost to visit cities in mask, ending at i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # start at city 0

    for mask in range(1, 1 << n):
        for i in range(n):
            if dp[mask][i] == INF:
                continue
            if not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist[i][j]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost

    # Close the cycle back to city 0
    return int(min(dp[full][i] + dist[i][0] for i in range(n)))


# ===================================================================
# Assignment problem via bitmask DP
# ===================================================================
def assignment_bitmask(cost: list[list[int]]) -> int:
    """Solve the assignment problem using bitmask DP.

    Parameters
    ----------
    cost : list[list[int]]
        Cost matrix where cost[i][j] is the cost of assigning worker i
        to task j.

    Returns
    -------
    int
        Minimum total assignment cost.
    """
    n = len(cost)
    INF = math.inf
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue
        k = bin(mask).count("1")  # number of workers assigned so far
        if k >= n:
            continue
        for j in range(n):
            if mask & (1 << j):
                continue
            new_mask = mask | (1 << j)
            val = dp[mask] + cost[k][j]
            if val < dp[new_mask]:
                dp[new_mask] = val

    return int(dp[(1 << n) - 1])


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    # TSP example: 4 cities
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    print(f"TSP minimum cost: {tsp_bitmask(dist)}")

    # Assignment example: 3 workers, 3 tasks
    cost = [
        [9, 2, 7],
        [6, 4, 3],
        [5, 8, 1],
    ]
    print(f"Assignment minimum cost: {assignment_bitmask(cost)}")
```

**Output:**
```
TSP minimum cost: 80
Assignment minimum cost: 7
```

## Complexity

| Problem | Time | Space |
|---------|------|-------|
| TSP | $O(2^n \cdot n^2)$ | $O(2^n \cdot n)$ |
| Assignment | $O(2^n \cdot n)$ | $O(2^n)$ |
| General bitmask DP | $O(2^n \cdot n)$ | $O(2^n)$ |

!!! warning "Exponential growth"
    The $2^n$ factor limits bitmask DP to roughly $n \leq 20$ (about $10^6$ states). At $n = 25$ the state space exceeds $3 \times 10^7$ and memory becomes a bottleneck. For larger instances, approximation algorithms or branch-and-bound are necessary.

## Enumerating Subsets of a Mask

A common subroutine iterates over all subsets of a given mask. The bit trick `sub = (sub - 1) & mask` generates subsets in decreasing order:

```python
def enumerate_subsets(mask: int) -> list[int]:
    """Enumerate all subsets of mask (including mask and the empty set)."""
    subsets = []
    sub = mask
    while sub > 0:
        subsets.append(sub)
        sub = (sub - 1) & mask
    subsets.append(0)
    return subsets
```

This runs in $O(2^{|S|})$ time for a single mask $S$. When applied inside a DP that iterates over *all* masks, the total work across all masks of an $n$-element ground set is:

$$
\sum_{k=0}^{n} \binom{n}{k} 2^k = (1 + 2)^n = 3^n
$$

The equality follows from the binomial theorem with $x = 2$. Each element is in one of three states --- in the superset only, in both the superset and the subset, or in neither --- giving $3^n$ total (superset, subset) pairs. This $O(3^n)$ bound arises in problems like Steiner tree DP and weighted set cover.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
- Held, M. & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *Journal of SIAM*, 10(1), 196--210.
