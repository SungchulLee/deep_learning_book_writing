# Bitmask DP

Many optimization problems require exploring all subsets of a set of elements. When the number of elements $n$ is small (typically $n \leq 20$), encoding each subset as a binary integer turns subset operations into fast bitwise instructions. Bitmask DP leverages this encoding to solve problems like the Traveling Salesman Problem, task assignment, and set cover by iterating over all $2^n$ subsets systematically. This technique transforms exponential brute-force enumeration into structured dynamic programming with efficient subset transitions.

## Subset Representation

A subset $S \subseteq \{0, 1, \ldots, n-1\}$ is represented by an $n$-bit integer (called a **mask**) where bit $i$ is 1 if element $i$ belongs to $S$ and 0 otherwise.

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
| Full set | `(1 << n) - 1` | All $n$ bits set |
| Subset size | `bin(mask).count('1')` | Population count |

## General Framework

A bitmask DP problem defines a state $dp[\text{mask}]$ (or $dp[\text{mask}][i]$) where mask encodes which elements have been processed. The recurrence transitions between states by adding or removing elements from the subset.

**Generic recurrence** (minimization variant):

$$
dp[\text{mask}] = \min_{i \in \text{mask}} \bigl( dp[\text{mask} \setminus \{i\}] + \text{cost}(i, \text{mask}) \bigr)
$$

**Base case**: $dp[0] = 0$ (empty subset, no cost).

**Time complexity**: $O(2^n \cdot n)$ for iterating over all masks and all elements in each mask.

**Space complexity**: $O(2^n)$ or $O(2^n \cdot n)$ depending on whether an additional dimension is needed.

## Example: Traveling Salesman Problem

The TSP asks for the minimum-cost Hamiltonian cycle visiting all $n$ cities exactly once. Define $dp[\text{mask}][i]$ as the minimum cost to visit exactly the cities in mask, ending at city $i$.

**Recurrence**: for each city $j \in \text{mask}$ with $j \neq i$:

$$
dp[\text{mask}][i] = \min_{j \in \text{mask} \setminus \{i\}} \bigl( dp[\text{mask} \setminus \{i\}][j] + \text{dist}(j, i) \bigr)
$$

**Base case**: $dp[\{0\}][0] = 0$ (start at city 0 with only city 0 visited).

**Answer**: $\min_{i} \bigl( dp[(1 \ll n) - 1][i] + \text{dist}(i, 0) \bigr)$ -- complete the cycle back to city 0.

## Example: Assignment Problem

Given $n$ workers and $n$ tasks with cost matrix $C$, assign each worker to exactly one task to minimize total cost. Define $dp[\text{mask}]$ as the minimum cost to assign tasks in mask to the first $|\text{mask}|$ workers.

**Recurrence**: let $k = |\text{mask}|$ (the number of assigned tasks so far):

$$
dp[\text{mask}] = \min_{j \in \text{mask}} \bigl( dp[\text{mask} \setminus \{j\}] + C[k-1][j] \bigr)
$$

**Base case**: $dp[0] = 0$.

## Implementation

```python
"""
Bitmask DP: Traveling Salesman Problem and Assignment Problem.
"""

import math


# ===================================================================
# TSP via bitmask DP
# ===================================================================
def tsp_bitmask(dist: list[list[int]]) -> int:
    """Solve TSP using bitmask DP.

    Parameters
    ----------
    dist : list[list[int]]
        Distance matrix where dist[i][j] is the cost from city i to j.

    Returns
    -------
    int
        Minimum cost of a Hamiltonian cycle.
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
    The $2^n$ factor makes bitmask DP practical only for $n \leq 20$ (about $10^6$ states). For $n = 25$, the state space exceeds $3 \times 10^7$, and memory becomes a bottleneck. For larger instances, approximation algorithms or branch-and-bound are necessary.

## Enumerating Subsets of a Mask

A common subroutine iterates over all subsets of a given mask. The bit trick `sub = (sub - 1) & mask` generates subsets in decreasing order:

```python
# ===================================================================
# Subset enumeration
# ===================================================================
def enumerate_subsets(mask: int) -> list[int]:
    """Enumerate all subsets of mask (including mask and 0)."""
    subsets = []
    sub = mask
    while sub > 0:
        subsets.append(sub)
        sub = (sub - 1) & mask
    subsets.append(0)
    return subsets
```

This runs in $O(2^{|\text{mask}|})$ time across all subsets. When applied to all masks, the total work over all masks is:

$$
\sum_{k=0}^{n} \binom{n}{k} 2^k = 3^n
$$

This $O(3^n)$ bound arises in problems like the Steiner tree DP and weighted set cover.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
- Held, M. & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *Journal of SIAM*, 10(1), 196--210.
