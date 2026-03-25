# Sliding Window Maximum

Given an array of $n$ numbers and a window size $k$, many applications need the maximum within every contiguous subarray of length $k$.  A brute-force scan of each window costs $O(k)$ per position, yielding $O(nk)$ overall.  By maintaining a **monotonic deque** — a [deque](adt.md) whose elements are kept in decreasing order — we can solve the problem in $O(n)$ total time, because each element enters and leaves the deque at most once.  This page states the problem formally, develops the algorithm, proves its correctness, and provides a Python implementation.

## Problem Statement

**Input.** An array $A[0 \ldots n-1]$ of real numbers and an integer $1 \le k \le n$.

**Output.** An array $M[0 \ldots n-k]$ where

$$
M[i] = \max(A[i], A[i+1], \dots, A[i+k-1])
$$

for each $0 \le i \le n - k$.

## Monotonic Deque Algorithm

The key insight is to maintain a deque $D$ that stores **indices** into $A$, with the invariant that the values at those indices are in strictly decreasing order from front to back.

### Invariant

At every step, the deque $D = \langle d_0, d_1, \dots, d_{m-1} \rangle$ satisfies:

1. **Monotonicity**: $A[d_0] > A[d_1] > \cdots > A[d_{m-1}]$.
2. **Window membership**: all indices in $D$ lie within the current window $[i - k + 1, \, i]$.
3. **Maximum at front**: $A[d_0]$ is the maximum of the current window.

### Algorithm

For each index $i$ from $0$ to $n - 1$:

1. **Remove expired**: if the front index $d_0$ has fallen out of the window (i.e., $d_0 \le i - k$), remove it with `pop_front()`.
2. **Maintain monotonicity**: while the deque is non-empty and $A[\text{back}] \le A[i]$, remove the back element with `pop_back()`.  These elements can never be the maximum of any future window because $A[i]$ is at least as large and will remain in the window at least as long.
3. **Insert**: push $i$ onto the back with `push_back(i)`.
4. **Record result**: if $i \ge k - 1$ (i.e., the first full window has been reached), then $M[i - k + 1] = A[d_0]$.

### Correctness Argument

After step 2, every remaining index $d_j$ in the deque satisfies $A[d_j] > A[i]$, so inserting $i$ at the back preserves monotonicity.  Step 1 ensures all indices are within the current window.  Because the deque is sorted in decreasing order by value, the front element is always the window maximum.

### Complexity

Each of the $n$ elements is pushed onto the deque exactly once and popped at most once (either from the front when it expires or from the back when a larger element arrives).  Therefore the total number of deque operations is at most $2n$, and each operation costs $O(1)$.

$$
T(n) = O(n)
$$

The algorithm uses $O(k)$ extra space for the deque, since the deque never holds more than $k$ indices.

## Worked Example

??? example "Trace with A = [1, 3, -1, -3, 5, 3, 6, 7] and k = 3"

    | $i$ | $A[i]$ | Deque (indices) | Deque (values) | $M$ |
    |-----|--------|-----------------|----------------|-----|
    | 0 | 1 | [0] | [1] | — |
    | 1 | 3 | [1] | [3] | — |
    | 2 | -1 | [1, 2] | [3, -1] | 3 |
    | 3 | -3 | [1, 2, 3] | [3, -1, -3] | 3 |
    | 4 | 5 | [4] | [5] | 5 |
    | 5 | 3 | [4, 5] | [5, 3] | 5 |
    | 6 | 6 | [6] | [6] | 6 |
    | 7 | 7 | [7] | [7] | 7 |

    At $i = 1$: index 0 is popped from the back because $A[0] = 1 \le 3 = A[1]$.

    At $i = 3$: index 1 is still valid ($1 > 3 - 3 = 0$), so it remains at the front.

    At $i = 4$: indices 1, 2, 3 are all popped from the back because their values are $\le 5$.  Index 1 would also be expired ($1 \le 4 - 3 = 1$).

    The final result is $M = [3, 3, 5, 5, 6, 7]$.

## Python Implementation

```python
"""Sliding window maximum using a monotonic deque."""

from collections import deque


# === Monotonic Deque Algorithm ===

def sliding_window_max(nums: list[int | float], k: int) -> list[int | float]:
    """Return the maximum of every contiguous subarray of length k.

    Args:
        nums: Input array of numbers.
        k: Window size (1 <= k <= len(nums)).

    Returns:
        List of length len(nums) - k + 1 with the window maximums.

    Time:  O(n) where n = len(nums).
    Space: O(k) for the deque.
    """
    n = len(nums)
    dq = deque()  # stores indices, values in decreasing order
    result = []

    for i in range(n):
        # Remove indices that have left the window
        if dq and dq[0] <= i - k:
            dq.popleft()

        # Remove back elements smaller than or equal to nums[i]
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        # Record the maximum once the first full window is reached
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# === Demo ===

if __name__ == "__main__":
    A = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"Array: {A}")
    print(f"Window size: {k}")
    print(f"Sliding window max: {sliding_window_max(A, k)}")
    # Output: [3, 3, 5, 5, 6, 7]
```

## Sliding Window Minimum

The same technique works for the sliding window minimum by reversing the comparison: pop from the back while $A[\text{back}] \ge A[i]$.  The front of the deque then holds the index of the current minimum.

## Applications

- **Max-pooling in CNNs**: computing the maximum over spatial patches can be viewed as a sliding window maximum along each dimension.
- **Stock price analysis**: finding rolling highs and lows over a fixed-size time window.
- **Monotone queue optimization in dynamic programming**: certain DP recurrences of the form $f(i) = \max_{j \in [i-k, i-1]} g(j) + h(i)$ can be accelerated from $O(nk)$ to $O(n)$ using this technique.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
