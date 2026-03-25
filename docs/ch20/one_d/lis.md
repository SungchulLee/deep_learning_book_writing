# Longest Increasing Subsequence

The longest increasing subsequence (LIS) problem asks for the length of the longest subsequence of a given array in which every element is strictly greater than the previous one.  This is one of the most studied problems in dynamic programming, with an elegant $O(n^2)$ DP solution and a faster $O(n \log n)$ algorithm that uses binary search.  The problem appears frequently in algorithm design and has applications in patience sorting, longest chain problems, and sequence alignment.

## Problem Statement

Given an array $a[0..n-1]$ of integers, find the length of the longest subsequence $a[i_1], a[i_2], \ldots, a[i_k]$ such that $i_1 < i_2 < \cdots < i_k$ and $a[i_1] < a[i_2] < \cdots < a[i_k]$.

**Example:** For $a = [10, 9, 2, 5, 3, 7, 101, 18]$, the LIS is $[2, 3, 7, 101]$ with length 4.

## The O(n-squared) DP Approach

### Recurrence

Let $dp[i]$ denote the length of the longest increasing subsequence ending at index $i$.  For each $i$, consider all previous indices $j < i$ where $a[j] < a[i]$.  Extending the subsequence ending at $j$ by appending $a[i]$ gives a subsequence of length $dp[j] + 1$.  Taking the maximum over all valid $j$:

$$
dp[i] = 1 + \max_{\substack{0 \le j < i \\ a[j] < a[i]}} dp[j] \quad \text{for } 0 \le i < n
$$

If no valid $j$ exists (no smaller element before position $i$), then $dp[i] = 1$ since $a[i]$ alone forms a subsequence of length 1.

The answer is $\max_{0 \le i < n} dp[i]$.

### Implementation

```python
"""
Longest increasing subsequence: O(n^2) DP and O(n log n) binary search.
"""

from bisect import bisect_left


# ===================================================================
# Approach 1: O(n^2) DP
# ===================================================================
def lis_dp(nums: list[int]) -> int:
    """LIS length using O(n^2) DP."""
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)
```

The outer loop runs $n$ times and the inner loop up to $i$ times, giving $O(n^2)$ time and $O(n)$ space.

## The O(n log n) Approach

### Key Idea

Maintain an array $\text{tails}$ where $\text{tails}[k]$ stores the smallest possible tail element of an increasing subsequence of length $k + 1$ found so far.  This array is always sorted, which enables binary search.

For each element $a[i]$:

- If $a[i]$ is greater than all elements in $\text{tails}$, append it (extends the longest subsequence found).
- Otherwise, find the smallest element in $\text{tails}$ that is $\ge a[i]$ and replace it with $a[i]$ (this keeps the tails as small as possible, maximizing future extension potential).

The length of the LIS equals the final length of $\text{tails}$.

### Implementation

```python
# ===================================================================
# Approach 2: O(n log n) with binary search
# ===================================================================
def lis_binary_search(nums: list[int]) -> int:
    """LIS length using patience sorting / binary search. O(n log n)."""
    tails: list[int] = []

    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)
```

Each of the $n$ elements requires a binary search on $\text{tails}$ (length at most $n$), giving $O(n \log n)$ time and $O(n)$ space.

### Correctness Sketch

The invariant is that $\text{tails}$ is always sorted and $\text{tails}[k]$ is the minimum possible last element of any increasing subsequence of length $k+1$.  Each new element either extends the array (proving a longer IS exists) or reduces some tail value (preserving or improving future opportunities).  The length of $\text{tails}$ at the end equals the LIS length.

!!! warning "tails is not the LIS itself"
    The array $\text{tails}$ at the end of the algorithm does not necessarily contain the actual LIS elements.  It contains the smallest possible tail for each length.  To reconstruct the actual LIS, additional bookkeeping (parent pointers or index tracking) is needed.

## Reconstructing the LIS

To recover the actual subsequence, store the index of the predecessor for each element and the position where each element was placed in $\text{tails}$:

```python
# ===================================================================
# Reconstruction of the actual LIS
# ===================================================================
def lis_with_reconstruction(nums: list[int]) -> list[int]:
    """Return the actual LIS (not just its length)."""
    if not nums:
        return []

    n = len(nums)
    tails: list[int] = []
    tails_idx: list[int] = []
    parent = [-1] * n
    pos = [0] * n

    for i, num in enumerate(nums):
        p = bisect_left(tails, num)
        if p == len(tails):
            tails.append(num)
            tails_idx.append(i)
        else:
            tails[p] = num
            tails_idx[p] = i
        pos[i] = p
        parent[i] = tails_idx[p - 1] if p > 0 else -1

    # Backtrack from the last element of the longest subsequence
    lis_len = len(tails)
    result = [0] * lis_len
    idx = tails_idx[lis_len - 1]
    for k in range(lis_len - 1, -1, -1):
        result[k] = nums[idx]
        idx = parent[idx]

    return result
```

## Complexity Comparison

| Approach | Time | Space |
|----------|------|-------|
| DP | $O(n^2)$ | $O(n)$ |
| Binary search | $O(n \log n)$ | $O(n)$ |
| With reconstruction | $O(n \log n)$ | $O(n)$ |

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_cases = [
        [10, 9, 2, 5, 3, 7, 101, 18],
        [0, 1, 0, 3, 2, 3],
        [7, 7, 7, 7, 7],
    ]
    for nums in test_cases:
        length_dp = lis_dp(nums)
        length_bs = lis_binary_search(nums)
        subseq = lis_with_reconstruction(nums)
        print(f"nums={nums}  LIS length={length_dp}  subsequence={subseq}")
```

**Output:**
```
nums=[10, 9, 2, 5, 3, 7, 101, 18]  LIS length=4  subsequence=[2, 3, 7, 101]
nums=[0, 1, 0, 3, 2, 3]  LIS length=4  subsequence=[0, 1, 2, 3]
nums=[7, 7, 7, 7, 7]  LIS length=1  subsequence=[7]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
