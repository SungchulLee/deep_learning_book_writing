# Rod Cutting

The rod cutting problem is one of the classic introductory examples of dynamic programming, used in textbooks like CLRS to illustrate optimal substructure, overlapping subproblems, and the progression from naive recursion to efficient tabulation.  Given a rod of integer length $n$ and a price table specifying the revenue for each possible piece length, the goal is to determine the cuts that maximize total revenue.

## Problem Statement

Given a rod of length $n$ and a price table $p_1, p_2, \ldots, p_n$ where $p_i$ is the price for a piece of length $i$, determine the maximum revenue $r(n)$ obtainable by cutting the rod into pieces and selling them.

**Example:** With $n = 4$ and prices $p_1 = 1, p_2 = 5, p_3 = 8, p_4 = 9$:

| Cut | Revenue |
|-----|---------|
| No cut: piece of length 4 | 9 |
| Pieces of length 2 + 2 | 5 + 5 = 10 |
| Pieces of length 1 + 3 | 1 + 8 = 9 |
| Pieces of length 1 + 1 + 2 | 1 + 1 + 5 = 7 |

The optimal solution is two pieces of length 2, yielding revenue 10.

## Recurrence

The key insight is to consider the first cut.  If the first piece has length $i$ (where $1 \le i \le n$), the remaining rod has length $n - i$ and must be cut optimally.  This gives

$$
r(n) = \max_{1 \le i \le n} \bigl(p_i + r(n - i)\bigr)
$$

with base case $r(0) = 0$ (a rod of length 0 has zero revenue).

The problem has optimal substructure because the remaining rod of length $n - i$ must be cut optimally — otherwise, replacing its cutting plan with a better one would increase total revenue.

## Naive Recursion

```python
"""
Rod cutting: maximize revenue from cutting a rod of length n.
"""


# ===================================================================
# Approach 1: Naive recursion
# ===================================================================
def rod_cut_recursive(prices: list[int], n: int) -> int:
    """Maximum revenue via naive recursion. Time: O(2^n), Space: O(n)."""
    if n == 0:
        return 0
    best = -1
    for i in range(1, n + 1):
        best = max(best, prices[i - 1] + rod_cut_recursive(prices, n - i))
    return best
```

The recursion tree has $2^{n-1}$ leaves (corresponding to the $2^{n-1}$ ways to cut a rod of length $n$), giving exponential time.

## Memoization (Top-Down)

```python
# ===================================================================
# Approach 2: Memoization (top-down)
# ===================================================================
def rod_cut_memo(prices: list[int], n: int, memo: dict[int, int] | None = None) -> int:
    """Maximum revenue with memoization. Time: O(n^2), Space: O(n)."""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n == 0:
        return 0
    best = -1
    for i in range(1, n + 1):
        best = max(best, prices[i - 1] + rod_cut_memo(prices, n - i, memo))
    memo[n] = best
    return best
```

There are $n + 1$ distinct subproblems, and solving subproblem $r(j)$ requires iterating over at most $j$ choices.  Total time is $\sum_{j=0}^{n} j = O(n^2)$.

## Tabulation (Bottom-Up)

```python
# ===================================================================
# Approach 3: Tabulation (bottom-up)
# ===================================================================
def rod_cut_tabulation(prices: list[int], n: int) -> int:
    """Maximum revenue with tabulation. Time: O(n^2), Space: O(n)."""
    r = [0] * (n + 1)
    for j in range(1, n + 1):
        best = -1
        for i in range(1, j + 1):
            best = max(best, prices[i - 1] + r[j - i])
        r[j] = best
    return r[n]
```

## Reconstructing the Cuts

To find the actual pieces, maintain an auxiliary array recording the first cut at each length:

```python
# ===================================================================
# Approach 4: With reconstruction
# ===================================================================
def rod_cut_with_cuts(prices: list[int], n: int) -> tuple[int, list[int]]:
    """Return maximum revenue and the list of piece lengths."""
    r = [0] * (n + 1)
    s = [0] * (n + 1)

    for j in range(1, n + 1):
        best = -1
        for i in range(1, j + 1):
            if prices[i - 1] + r[j - i] > best:
                best = prices[i - 1] + r[j - i]
                s[j] = i
        r[j] = best

    cuts = []
    remaining = n
    while remaining > 0:
        cuts.append(s[remaining])
        remaining -= s[remaining]

    return r[n], cuts
```

## Complexity

| Approach | Time | Space |
|----------|------|-------|
| Naive recursion | $O(2^n)$ | $O(n)$ |
| Memoization | $O(n^2)$ | $O(n)$ |
| Tabulation | $O(n^2)$ | $O(n)$ |

The $O(n^2)$ time comes from solving $n$ subproblems, each requiring a loop over at most $n$ cut positions.

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    prices = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
    for n in [4, 7, 10]:
        revenue, cuts = rod_cut_with_cuts(prices, n)
        print(f"n={n}  revenue={revenue}  cuts={cuts}")
```

**Output:**
```
n=4  revenue=10  cuts=[2, 2]
n=7  revenue=18  cuts=[1, 6]
n=10  revenue=30  cuts=[10]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
