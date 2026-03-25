# Climbing Stairs

The climbing stairs problem is one of the most natural introductions to one-dimensional dynamic programming.  Given a staircase of $n$ steps where you can climb 1 or 2 steps at a time, the task is to count the number of distinct ways to reach the top.  The problem has a direct Fibonacci-like recurrence, making it an ideal bridge between the Fibonacci sequence and more complex DP formulations.

## Problem Statement

You are climbing a staircase with $n$ steps.  At each step, you can climb either 1 step or 2 steps.  How many distinct ways can you reach the top?

For example, with $n = 3$ steps, there are 3 ways: $(1,1,1)$, $(1,2)$, and $(2,1)$.

## Recurrence Derivation

Let $dp[i]$ denote the number of distinct ways to reach step $i$.  To arrive at step $i$, you must have come from either step $i-1$ (taking 1 step) or step $i-2$ (taking 2 steps).  Since these two cases are mutually exclusive, the total count is their sum:

$$
dp[i] = dp[i-1] + dp[i-2] \quad \text{for } i \ge 2
$$

The base cases capture the starting conditions:

$$
dp[0] = 1, \quad dp[1] = 1
$$

Here $dp[0] = 1$ represents the single way to stay at the ground level (doing nothing), and $dp[1] = 1$ represents the single way to reach step 1 (one step of size 1).

This recurrence is identical to the Fibonacci sequence shifted by one index: $dp[n] = F(n+1)$.

## Naive Recursion

```python
"""
Climbing stairs: count distinct ways to climb n steps,
taking 1 or 2 steps at a time.
"""


# ===================================================================
# Approach 1: Naive recursion
# ===================================================================
def climb_recursive(n: int) -> int:
    """Count ways to climb n stairs. Time: O(2^n), Space: O(n)."""
    if n <= 1:
        return 1
    return climb_recursive(n - 1) + climb_recursive(n - 2)
```

This has the same exponential $O(2^n)$ time complexity as naive Fibonacci due to the overlapping subproblem structure.

## Memoization (Top-Down)

```python
# ===================================================================
# Approach 2: Memoization (top-down)
# ===================================================================
def climb_memo(n: int, memo: dict[int, int] | None = None) -> int:
    """Count ways with memoization. Time: O(n), Space: O(n)."""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return 1
    memo[n] = climb_memo(n - 1, memo) + climb_memo(n - 2, memo)
    return memo[n]
```

Each of the $n + 1$ subproblems is solved exactly once, reducing the time to $O(n)$.

## Tabulation (Bottom-Up)

```python
# ===================================================================
# Approach 3: Tabulation (bottom-up)
# ===================================================================
def climb_tabulation(n: int) -> int:
    """Count ways with tabulation. Time: O(n), Space: O(n)."""
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

## Space Optimization

Since each state depends only on the two preceding states, space reduces to $O(1)$:

```python
# ===================================================================
# Approach 4: Space-optimized
# ===================================================================
def climb_optimized(n: int) -> int:
    """Count ways with O(1) space. Time: O(n), Space: O(1)."""
    if n <= 1:
        return 1
    prev2, prev1 = 1, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1
```

## Generalization: k Steps at a Time

A natural extension allows taking 1 through $k$ steps at a time.  The recurrence becomes

$$
dp[i] = \sum_{j=1}^{\min(i, k)} dp[i - j]
$$

with $dp[0] = 1$.  The tabulation approach extends naturally, with each cell summing the previous $k$ entries.

## Complexity Summary

| Approach | Time | Space |
|----------|------|-------|
| Naive recursion | $O(2^n)$ | $O(n)$ |
| Memoization | $O(n)$ | $O(n)$ |
| Tabulation | $O(n)$ | $O(n)$ |
| Space-optimized | $O(n)$ | $O(1)$ |

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    for n in range(1, 8):
        result = climb_optimized(n)
        print(f"climb({n}) = {result}")
```

**Output:**
```
climb(1) = 1
climb(2) = 2
climb(3) = 3
climb(4) = 5
climb(5) = 8
climb(6) = 13
climb(7) = 21
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
