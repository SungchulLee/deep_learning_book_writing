# House Robber

The house robber problem is a classic one-dimensional dynamic programming exercise that introduces the idea of **exclusion constraints**.  A robber wants to maximize the total value stolen from a row of houses, but cannot rob two adjacent houses without triggering an alarm.  This constraint creates an interesting decision at each house: rob it and skip the neighbor, or skip it and keep the option of robbing the neighbor.

## Problem Statement

Given an array $\text{nums}[0..n-1]$ where $\text{nums}[i]$ represents the value in house $i$, find the maximum total value that can be robbed without selecting two adjacent houses.

**Example:** For $\text{nums} = [2, 7, 9, 3, 1]$, the optimal choice is houses 0, 2, and 4 with total $2 + 9 + 1 = 12$.

## Recurrence Derivation

Let $dp[i]$ denote the maximum value obtainable from houses $0$ through $i$.  At house $i$, there are two choices:

1. **Rob house $i$**: gain $\text{nums}[i]$ but cannot use house $i-1$, so the best from the remaining is $dp[i-2]$.
2. **Skip house $i$**: the best remains $dp[i-1]$.

Taking the better option gives the recurrence

$$
dp[i] = \max\bigl(dp[i-1],\; dp[i-2] + \text{nums}[i]\bigr) \quad \text{for } i \ge 2
$$

with base cases

$$
dp[0] = \text{nums}[0], \quad dp[1] = \max(\text{nums}[0],\; \text{nums}[1])
$$

## Optimal Substructure

The problem exhibits optimal substructure because the optimal solution for houses $0..i$ is built from the optimal solution for either houses $0..i-1$ (skip) or houses $0..i-2$ (rob).  A cut-and-paste argument confirms this: if the sub-solution were not optimal, replacing it with a better one would improve the overall solution, contradicting optimality.

## Tabulation

```python
"""
House robber: maximize total value from non-adjacent houses.
"""


# ===================================================================
# Approach 1: Tabulation (bottom-up)
# ===================================================================
def rob_tabulation(nums: list[int]) -> int:
    """Maximum robbery value with tabulation. Time: O(n), Space: O(n)."""
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]

    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[n - 1]
```

## Space Optimization

Each state depends only on the two previous values, so space reduces to $O(1)$:

```python
# ===================================================================
# Approach 2: Space-optimized
# ===================================================================
def rob_optimized(nums: list[int]) -> int:
    """Maximum robbery value with O(1) space. Time: O(n), Space: O(1)."""
    if not nums:
        return 0
    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)
    return prev1
```

In this formulation, `prev1` tracks $dp[i-1]$ and `prev2` tracks $dp[i-2]$.  At each step, the new `prev1` is $\max(dp[i-1], dp[i-2] + \text{nums}[i])$.

## Reconstructing the Solution

To find which houses are actually robbed, trace back through the DP table:

```python
# ===================================================================
# Reconstruction
# ===================================================================
def rob_with_reconstruction(nums: list[int]) -> tuple[int, list[int]]:
    """Return maximum value and list of robbed house indices."""
    n = len(nums)
    if n == 0:
        return 0, []
    if n == 1:
        return nums[0], [0]

    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    # Backtrack to find chosen houses
    chosen = []
    i = n - 1
    while i >= 0:
        if i == 0 or dp[i] != dp[i - 1]:
            chosen.append(i)
            i -= 2
        else:
            i -= 1

    return dp[n - 1], list(reversed(chosen))
```

## Complexity

| Approach | Time | Space |
|----------|------|-------|
| Tabulation | $O(n)$ | $O(n)$ |
| Space-optimized | $O(n)$ | $O(1)$ |
| With reconstruction | $O(n)$ | $O(n)$ |

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_cases = [
        [1, 2, 3, 1],
        [2, 7, 9, 3, 1],
        [2, 1, 1, 2],
    ]
    for nums in test_cases:
        value, houses = rob_with_reconstruction(nums)
        print(f"nums={nums}  max={value}  houses={houses}")
```

**Output:**
```
nums=[1, 2, 3, 1]  max=4  houses=[0, 2]
nums=[2, 7, 9, 3, 1]  max=12  houses=[0, 2, 4]
nums=[2, 1, 1, 2]  max=4  houses=[0, 3]
```

!!! note "Circular variant"
    In the **House Robber II** variant, houses are arranged in a circle, so house 0 and house $n-1$ are adjacent.  This is solved by running the linear algorithm twice: once on houses $0..n-2$ and once on houses $1..n-1$, then taking the maximum.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
