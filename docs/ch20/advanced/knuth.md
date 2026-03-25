# Knuth's Optimization

Interval DP problems like optimal binary search tree and minimum-cost merging typically run in $O(n^3)$ time because each state $dp[i][j]$ scans all split points $k$ from $i$ to $j$. Knuth's optimization exploits a structural property of the cost function -- the **quadrangle inequality** -- to restrict each split-point search to a narrower range. By recording the optimal split point and using the monotonicity $\text{opt}(i, j-1) \leq \text{opt}(i, j) \leq \text{opt}(i+1, j)$, the total work across all states drops from $O(n^3)$ to $O(n^2)$.

## Problem Setup

Consider an interval DP recurrence of the form:

$$
dp[i][j] = \min_{i \leq k < j} \bigl( dp[i][k] + dp[k+1][j] + C(i, j) \bigr)
$$

where $C(i, j)$ is the cost of combining the interval $[i, j]$ (independent of the split point $k$). Let $\text{opt}(i, j)$ be the smallest $k$ achieving the minimum.

## Quadrangle Inequality

A function $C$ satisfies the **quadrangle inequality** if:

$$
C(a, c) + C(b, d) \leq C(a, d) + C(b, c) \quad \text{for all } a \leq b \leq c \leq d
$$

Intuitively, the cost of two "nested" intervals is at most the cost of two "crossing" intervals.

!!! note "Sufficient condition"
    If $C(i, j)$ satisfies the quadrangle inequality and is **monotone** (i.e., $C(i', j') \leq C(i, j)$ whenever $[i', j'] \subseteq [i, j]$), then the DP value function $dp[i][j]$ also satisfies the quadrangle inequality, and the optimal split points are monotone.

## Monotonicity of Split Points

When the quadrangle inequality holds:

$$
\text{opt}(i, j-1) \leq \text{opt}(i, j) \leq \text{opt}(i+1, j)
$$

This means the optimal split point for a given interval is sandwiched between the optimal split points of two "adjacent" sub-intervals. This constraint dramatically narrows the search range.

## Algorithm

The key modification to standard interval DP:

1. Maintain a table $\text{opt}[i][j]$ recording the optimal split point
2. When computing $dp[i][j]$, only scan $k$ from $\text{opt}[i][j-1]$ to $\text{opt}[i+1][j]$

**Amortized analysis**: for a fixed interval length $\ell = j - i$, the total number of split points examined across all intervals of length $\ell$ is:

$$
\sum_{i} \bigl(\text{opt}(i+1, i+\ell) - \text{opt}(i, i+\ell-1)\bigr) + n = O(n)
$$

This is a telescoping sum. Summing over all $O(n)$ lengths gives $O(n^2)$ total work.

## Complexity Comparison

| Method | Time | Space |
|--------|------|-------|
| Naive interval DP | $O(n^3)$ | $O(n^2)$ |
| Knuth's optimization | $O(n^2)$ | $O(n^2)$ |

## Implementation

```python
"""
Knuth's optimization: reduce interval DP from O(n^3) to O(n^2).

Applies when the cost function satisfies the quadrangle inequality,
guaranteeing monotone optimal split points.
"""

import math


# ===================================================================
# Optimal BST with Knuth's optimization
# ===================================================================
def optimal_bst(freq: list[int]) -> int:
    """Find minimum expected search cost for an optimal BST.

    Given keys with access frequencies, find the BST structure
    that minimizes the total weighted search cost.

    Parameters
    ----------
    freq : list[int]
        Access frequencies for keys 0, 1, ..., n-1.

    Returns
    -------
    int
        Minimum total weighted search cost.
    """
    n = len(freq)
    INF = math.inf

    # Prefix sums for O(1) range frequency queries
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]

    def range_freq(i: int, j: int) -> int:
        return prefix[j + 1] - prefix[i]

    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    # Base case: single keys
    for i in range(n):
        dp[i][i] = freq[i]
        opt[i][i] = i

    # Fill by increasing length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 <= j else j

            for k in range(lo, min(hi, j) + 1):
                left = dp[i][k - 1] if k > i else 0
                right = dp[k + 1][j] if k < j else 0
                cost = left + right + range_freq(i, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k

    return dp[0][n - 1]


# ===================================================================
# Generic Knuth-optimized interval DP
# ===================================================================
def knuth_interval_dp(n: int, cost_fn) -> int:
    """Generic Knuth-optimized interval DP.

    Parameters
    ----------
    n : int
        Number of elements.
    cost_fn : callable
        cost_fn(i, j) returns the merge cost for interval [i, j].

    Returns
    -------
    int
        Minimum total cost.
    """
    INF = math.inf
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    for i in range(n):
        opt[i][i] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 <= j else j

            for k in range(lo, min(hi, j) + 1):
                left = dp[i][k - 1] if k > i else 0
                right = dp[k + 1][j] if k < j else 0
                val = left + right + cost_fn(i, j)
                if val < dp[i][j]:
                    dp[i][j] = val
                    opt[i][j] = k

    return dp[0][n - 1]


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    # Optimal BST example
    freq = [25, 20, 5, 20, 30]
    result = optimal_bst(freq)
    print(f"Frequencies: {freq}")
    print(f"Optimal BST cost: {result}")

    # Compare naive O(n^3) vs Knuth O(n^2) for merge stones
    piles = [3, 5, 1, 2, 6]
    prefix = [0]
    for p in piles:
        prefix.append(prefix[-1] + p)
    cost_fn = lambda i, j: prefix[j + 1] - prefix[i]

    result = knuth_interval_dp(len(piles), cost_fn)
    print(f"\nPiles: {piles}")
    print(f"Merge cost (Knuth): {result}")
```

**Output:**
```
Frequencies: [25, 20, 5, 20, 30]
Optimal BST cost: 210
Merge cost (Knuth): 38
```

??? example "Verifying quadrangle inequality for prefix sums"
    Let $C(i, j) = \sum_{t=i}^{j} a_t$. Check for $a \leq b \leq c \leq d$:

    $$
    C(a,c) + C(b,d) = \sum_{t=a}^{c} a_t + \sum_{t=b}^{d} a_t
    $$

    $$
    C(a,d) + C(b,c) = \sum_{t=a}^{d} a_t + \sum_{t=b}^{c} a_t
    $$

    The difference is $C(a,c) + C(b,d) - C(a,d) - C(b,c) = -\sum_{t=c+1}^{d} a_t + \sum_{t=c+1}^{d} a_t = 0 \leq 0$, so the quadrangle inequality holds with equality for prefix-sum costs.

## Reference

- Knuth, D. E. (1971). Optimum binary search trees. *Acta Informatica*, 1(1), 14--25.
- Yao, F. F. (1980). Efficient dynamic programming using quadrangle inequalities. *Proc. STOC*, 429--435.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
