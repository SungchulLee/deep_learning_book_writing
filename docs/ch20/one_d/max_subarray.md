# Maximum Subarray

The maximum subarray problem asks for the contiguous subarray within a one-dimensional array of numbers that has the largest sum.  This problem is a foundational exercise in dynamic programming and is solved optimally by Kadane's algorithm in $O(n)$ time.  The key insight is that the decision at each position reduces to a simple choice: extend the current subarray or start a new one.

## Problem Statement

Given an array $a[0..n-1]$ of integers (possibly negative), find

$$
\max_{0 \le l \le r < n} \sum_{k=l}^{r} a[k]
$$

That is, find the contiguous subarray $a[l..r]$ whose elements have the largest sum.

**Example:** For $a = [-2, 1, -3, 4, -1, 2, 1, -5, 4]$, the maximum subarray is $[4, -1, 2, 1]$ with sum 6.

## Recurrence (Kadane's Algorithm)

Let $dp[i]$ denote the maximum sum of a subarray **ending at** index $i$.  At position $i$, there are two choices:

1. **Extend** the subarray ending at $i-1$ by including $a[i]$: the sum is $dp[i-1] + a[i]$.
2. **Start fresh** at $i$: the sum is just $a[i]$.

Taking the better option:

$$
dp[i] = \max\bigl(a[i],\; dp[i-1] + a[i]\bigr) \quad \text{for } i \ge 1
$$

with base case $dp[0] = a[0]$.  The answer is $\max_{0 \le i < n} dp[i]$.

The recurrence works because if the maximum subarray ending at $i-1$ has negative sum, it is better to discard it and start anew at $i$.

## Tabulation

```python
"""
Maximum subarray sum using Kadane's algorithm.
"""


# ===================================================================
# Approach 1: Tabulation
# ===================================================================
def max_subarray_tabulation(nums: list[int]) -> int:
    """Maximum subarray sum with explicit DP table. Time: O(n), Space: O(n)."""
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]

    for i in range(1, n):
        dp[i] = max(nums[i], dp[i - 1] + nums[i])

    return max(dp)
```

## Space-Optimized (Kadane's Algorithm)

Since $dp[i]$ depends only on $dp[i-1]$, a single variable suffices:

```python
# ===================================================================
# Approach 2: Kadane's algorithm (space-optimized)
# ===================================================================
def kadane(nums: list[int]) -> int:
    """Maximum subarray sum. Time: O(n), Space: O(1)."""
    current_sum = nums[0]
    best_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        best_sum = max(best_sum, current_sum)

    return best_sum
```

## Reconstructing the Subarray

To find the actual subarray (not just the sum), track the start and end indices:

```python
# ===================================================================
# Approach 3: With reconstruction
# ===================================================================
def kadane_with_indices(nums: list[int]) -> tuple[int, int, int]:
    """Return (max_sum, start_index, end_index)."""
    current_sum = nums[0]
    best_sum = nums[0]
    start = 0
    temp_start = 0
    end = 0

    for i in range(1, len(nums)):
        if nums[i] > current_sum + nums[i]:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum = current_sum + nums[i]

        if current_sum > best_sum:
            best_sum = current_sum
            start = temp_start
            end = i

    return best_sum, start, end
```

## Correctness Argument

Kadane's algorithm considers every possible ending position $i$ and computes the best subarray ending there.  By taking the maximum over all ending positions, it considers every possible subarray (since every subarray ends somewhere).  The recurrence correctly computes the best subarray ending at $i$ because extending a negative-sum prefix is always worse than starting fresh.

## Complexity

| Approach | Time | Space |
|----------|------|-------|
| Brute force (all pairs) | $O(n^2)$ | $O(1)$ |
| Divide and conquer | $O(n \log n)$ | $O(\log n)$ |
| Kadane's algorithm | $O(n)$ | $O(1)$ |

Kadane's algorithm is optimal because any algorithm must read every element at least once, giving an $\Omega(n)$ lower bound.

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_cases = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [1],
        [5, 4, -1, 7, 8],
        [-1, -2, -3],
    ]
    for nums in test_cases:
        total, start, end = kadane_with_indices(nums)
        print(f"nums={nums}")
        print(f"  max sum = {total}, subarray = {nums[start:end+1]}")
```

**Output:**
```
nums=[-2, 1, -3, 4, -1, 2, 1, -5, 4]
  max sum = 6, subarray = [4, -1, 2, 1]
nums=[1]
  max sum = 1, subarray = [1]
nums=[5, 4, -1, 7, 8]
  max sum = 23, subarray = [5, 4, -1, 7, 8]
nums=[-1, -2, -3]
  max sum = -1, subarray = [-1]
```

!!! note "All-negative arrays"
    When all elements are negative, the maximum subarray consists of the single least-negative element.  Kadane's algorithm handles this correctly because `current_sum` always starts fresh when extending would make the sum worse.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Kadane, J. B. (1984). Maximum sum subarray problem.
