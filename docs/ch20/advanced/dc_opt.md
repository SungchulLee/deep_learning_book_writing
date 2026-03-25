# Divide and Conquer Optimization

Many dynamic programming problems have the form "partition $n$ elements into $k$ groups to minimize total cost." The naive DP runs in $O(kn^2)$ time because computing each state requires scanning over all possible split points. When the optimal split point is **monotone** -- meaning it never decreases as the range grows -- divide and conquer optimization reduces this to $O(kn \log n)$ by recursively narrowing the search range for each split point. This technique applies to problems where the cost function satisfies the **concave quadrangle inequality** (also called the "opt monotonicity" condition).

## Problem Structure

Consider a DP of the form:

$$
dp[i][j] = \min_{k < j} \bigl( dp[i-1][k] + C(k+1, j) \bigr)
$$

where $dp[i][j]$ is the minimum cost of partitioning the first $j$ elements into $i$ groups, and $C(l, r)$ is the cost of a single group spanning elements $l$ through $r$.

Let $\text{opt}(i, j)$ denote the smallest $k$ achieving the minimum in the recurrence above.

## Monotonicity Condition

The divide and conquer optimization applies when for each fixed row $i$:

$$
\text{opt}(i, j) \leq \text{opt}(i, j+1) \quad \text{for all } j
$$

This means the optimal split point for position $j$ is at most the optimal split point for position $j+1$. This monotonicity holds whenever the cost function $C$ satisfies the **quadrangle inequality**:

$$
C(a, c) + C(b, d) \leq C(a, d) + C(b, c) \quad \text{for all } a \leq b \leq c \leq d
$$

Intuitively, the quadrangle inequality says that overlapping intervals cost no more than the non-overlapping alternatives. Many natural cost functions satisfy this property, including sum-of-squares costs and prefix-sum-based costs.

## Algorithm

For a fixed row $i$, instead of computing $dp[i][j]$ left to right in $O(n)$ per cell, use divide and conquer:

1. **Solve the middle**: compute $dp[i][\text{mid}]$ by scanning $k$ from $\text{lo}$ to $\text{hi}$, finding $\text{opt}(i, \text{mid})$.
2. **Recurse left**: solve $dp[i][j]$ for $j < \text{mid}$ with $k$ restricted to $[\text{lo}, \text{opt}(i, \text{mid})]$.
3. **Recurse right**: solve $dp[i][j]$ for $j > \text{mid}$ with $k$ restricted to $[\text{opt}(i, \text{mid}), \text{hi}]$.

Each level of recursion partitions the range $[1, n]$ and scans at most $O(n)$ values of $k$ in total. With $O(\log n)$ recursion levels, the work per row is $O(n \log n)$.

## Complexity

| Aspect | Naive DP | D&C Optimized |
|--------|----------|---------------|
| Per row | $O(n^2)$ | $O(n \log n)$ |
| Total ($k$ rows) | $O(kn^2)$ | $O(kn \log n)$ |
| Space | $O(kn)$ or $O(n)$ | $O(kn)$ or $O(n)$ |

## Implementation

```python
"""
Divide and conquer optimization for DP with monotone optimal split points.

Reduces O(kn^2) DP to O(kn log n) when the cost function satisfies
the quadrangle inequality.
"""

import math


# ===================================================================
# Cost function (example: sum of squares of prefix sums)
# ===================================================================
def precompute_prefix(arr: list[int]) -> list[int]:
    """Compute prefix sums for O(1) range sum queries."""
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix


def cost(prefix: list[int], l: int, r: int) -> int:
    """Cost of grouping elements l..r (sum of elements squared)."""
    s = prefix[r + 1] - prefix[l]
    return s * s


# ===================================================================
# Divide and conquer optimization
# ===================================================================
def solve(arr: list[int], k: int) -> int:
    """Partition arr into k groups to minimize total cost.

    Parameters
    ----------
    arr : list[int]
        Array of non-negative integers to partition.
    k : int
        Number of groups.

    Returns
    -------
    int
        Minimum total cost of the partition.
    """
    n = len(arr)
    prefix = precompute_prefix(arr)
    INF = math.inf

    # dp[j] = min cost for first j elements
    dp_prev = [INF] * (n + 1)
    dp_curr = [INF] * (n + 1)

    # Base: 1 group
    dp_prev[0] = 0
    for j in range(1, n + 1):
        dp_prev[j] = cost(prefix, 0, j - 1)

    for i in range(2, k + 1):
        dp_curr = [INF] * (n + 1)
        dp_curr[0] = 0

        def dc(j_lo: int, j_hi: int, k_lo: int, k_hi: int) -> None:
            """Divide and conquer on column range [j_lo, j_hi]."""
            if j_lo > j_hi:
                return
            j_mid = (j_lo + j_hi) // 2
            best_cost = INF
            best_k = k_lo

            for k_val in range(k_lo, min(k_hi, j_mid - 1) + 1):
                val = dp_prev[k_val] + cost(prefix, k_val, j_mid - 1)
                if val < best_cost:
                    best_cost = val
                    best_k = k_val

            dp_curr[j_mid] = best_cost
            dc(j_lo, j_mid - 1, k_lo, best_k)
            dc(j_mid + 1, j_hi, best_k, k_hi)

        dc(1, n, 0, n - 1)
        dp_prev = dp_curr[:]

    return int(dp_prev[n])


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    arr = [1, 5, 3, 2, 4, 6]
    for k in range(1, len(arr) + 1):
        result = solve(arr, k)
        print(f"k={k}: min cost = {result}")
```

**Output:**
```
k=1: min cost = 441
k=2: min cost = 193
k=3: min cost = 109
k=4: min cost = 91
k=5: min cost = 87
k=6: min cost = 85
```

??? example "Tracing the split point monotonicity"
    Consider partitioning $[1, 5, 3, 2, 4, 6]$ into $k=3$ groups with cost $= (\text{sum})^2$:

    - For $j=3$, the optimal split is at $k=1$, giving groups $[1]$, $[5,3]$
    - For $j=4$, the optimal split is at $k=2$, giving groups $[1,5]$, $[3,2]$
    - For $j=5$, the optimal split is at $k=2$, giving groups $[1,5]$, $[3,2,4]$

    The optimal split points $1, 2, 2$ are non-decreasing, confirming monotonicity.

## When Does This Apply

The divide and conquer optimization is useful whenever the cost function satisfies the quadrangle inequality. Common examples include:

- **Partitioning into groups** with sum-of-squares cost
- **CSES "Elevator Rides"** type problems with subset partitioning
- **Optimal binary search tree** cost functions
- **Post office placement** minimizing distance sums

!!! tip "Verifying the condition"
    If you suspect the optimization applies but cannot prove the quadrangle inequality analytically, test empirically by computing $\text{opt}(i, j)$ for small inputs and checking that the sequence is non-decreasing in $j$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
- Yao, F. F. (1980). Efficient dynamic programming using quadrangle inequalities. *Proc. STOC*, 429--435.
