# Sliding Window

The **sliding window** technique maintains a window (contiguous subarray/substring) that expands or contracts as it slides across the data. It reduces many $O(n^2)$ or $O(nk)$ problems to $O(n)$.

## Two Types

| Type | Window Size | Use Case |
|---|---|---|
| Fixed-size | Always $k$ | Max/min/average of all windows of size $k$ |
| Variable-size | Grows/shrinks | Longest/shortest subarray satisfying a condition |

## Fixed-Size Window

**Problem:** Find the maximum sum of any subarray of size $k$.

```python
def max_sum_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

print(max_sum_window([1, 4, 2, 10, 2, 3, 1, 0, 20], 4))  # Output: 24
```

## Variable-Size Window

**Problem:** Find the length of the longest substring without repeating characters.

```python
def longest_unique_substring(s):
    seen = {}
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        max_len = max(max_len, right - left + 1)
    return max_len

print(longest_unique_substring("abcabcbb"))  # Output: 3
```

## Variable Window: Shortest Subarray with Sum >= Target

```python
def min_subarray_sum(arr, target):
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(len(arr)):
        current_sum += arr[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1
    return min_len if min_len != float('inf') else 0

print(min_subarray_sum([2, 3, 1, 2, 4, 3], 7))  # Output: 2
```

## When Sliding Window Applies

The key requirement is that expanding the window can only move the condition in one direction (monotonicity). If adding an element can both improve and worsen the condition unpredictably, sliding window may not work directly.

All sliding window algorithms achieve $O(n)$ time because each element is added at most once and removed at most once.

# Reference

- LeetCode Sliding Window tag: [https://leetcode.com/tag/sliding-window/](https://leetcode.com/tag/sliding-window/)
- Halim, S. & Halim, F. *Competitive Programming 4*, 2020.
