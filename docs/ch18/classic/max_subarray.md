# Maximum Subarray

Given an array of numbers that may include negative values, the **maximum subarray problem** asks for the contiguous subarray with the largest sum. This problem arises naturally in financial analysis (finding the most profitable trading period), signal processing (locating the strongest signal segment), and genomics (identifying regions of biological significance).

While Kadane's algorithm solves the problem in $O(n)$ time using dynamic programming, the divide-and-conquer approach provides an elegant $O(n \log n)$ solution that illustrates the paradigm's mechanics clearly, especially the combine step.

## Problem Statement

Given an array $A[0 \,..\, n-1]$ of real numbers, find indices $i$ and $j$ with $0 \le i \le j \le n-1$ that maximize

$$
\sum_{k=i}^{j} A[k]
$$

If all elements are negative, the maximum subarray is the single element with the largest value.

## Divide-and-Conquer Approach

The key insight is that a maximum subarray of $A[\text{lo} \,..\, \text{hi}]$ must lie in exactly one of three positions relative to the midpoint $\text{mid} = \lfloor (\text{lo} + \text{hi}) / 2 \rfloor$:

1. **Entirely in the left half**: $A[\text{lo} \,..\, \text{mid}]$
2. **Entirely in the right half**: $A[\text{mid}+1 \,..\, \text{hi}]$
3. **Crossing the midpoint**: some $A[i \,..\, j]$ with $i \le \text{mid} < j$

Cases 1 and 2 are subproblems of the same form (solved recursively). Case 3 requires a dedicated **combine** step.

### Finding the Maximum Crossing Subarray

A crossing subarray includes $A[\text{mid}]$ and extends left to some index $i$ and right to some index $j$. We find the best leftward extension and the best rightward extension independently, then combine them.

**Left extension.** Starting from $\text{mid}$, scan leftward, tracking the maximum suffix sum:

$$
\text{left\_sum} = \max_{i \le \text{mid}} \sum_{k=i}^{\text{mid}} A[k]
$$

**Right extension.** Starting from $\text{mid}+1$, scan rightward, tracking the maximum prefix sum:

$$
\text{right\_sum} = \max_{j \ge \text{mid}+1} \sum_{k=\text{mid}+1}^{j} A[k]
$$

The maximum crossing sum is $\text{left\_sum} + \text{right\_sum}$, computed in $O(n)$ time with a single pass in each direction.

### Algorithm

```
MAX-CROSSING-SUBARRAY(A, lo, mid, hi):
    left_sum = -infinity
    sum = 0
    for i = mid downto lo:
        sum = sum + A[i]
        if sum > left_sum:
            left_sum = sum
            max_left = i

    right_sum = -infinity
    sum = 0
    for j = mid + 1 to hi:
        sum = sum + A[j]
        if sum > right_sum:
            right_sum = sum
            max_right = j

    return (max_left, max_right, left_sum + right_sum)
```

```
MAX-SUBARRAY(A, lo, hi):
    if lo == hi:
        return (lo, hi, A[lo])

    mid = floor((lo + hi) / 2)
    (l1, r1, s1) = MAX-SUBARRAY(A, lo, mid)
    (l2, r2, s2) = MAX-SUBARRAY(A, mid + 1, hi)
    (l3, r3, s3) = MAX-CROSSING-SUBARRAY(A, lo, mid, hi)

    return the triple (li, ri, si) with the largest si
```

### Python Implementation

```python
def max_crossing_subarray(arr, lo, mid, hi):
    """
    Find the maximum subarray that crosses the midpoint.

    Parameters
    ----------
    arr : list
        The input array.
    lo, mid, hi : int
        The subarray bounds with lo <= mid < hi.

    Returns
    -------
    tuple
        (left_index, right_index, max_sum)
    """
    # Extend left from mid
    left_sum = float('-inf')
    total = 0
    max_left = mid
    for i in range(mid, lo - 1, -1):
        total += arr[i]
        if total > left_sum:
            left_sum = total
            max_left = i

    # Extend right from mid + 1
    right_sum = float('-inf')
    total = 0
    max_right = mid + 1
    for j in range(mid + 1, hi + 1):
        total += arr[j]
        if total > right_sum:
            right_sum = total
            max_right = j

    return max_left, max_right, left_sum + right_sum


def max_subarray_dc(arr, lo=None, hi=None):
    """
    Find the maximum subarray using divide and conquer.

    Parameters
    ----------
    arr : list
        The input array of numbers.
    lo : int, optional
        Left bound (default: 0).
    hi : int, optional
        Right bound (default: len(arr) - 1).

    Returns
    -------
    tuple
        (left_index, right_index, max_sum)
    """
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(arr) - 1

    # Base case: single element
    if lo == hi:
        return lo, hi, arr[lo]

    mid = (lo + hi) // 2

    # Conquer: solve left and right subproblems
    l1, r1, s1 = max_subarray_dc(arr, lo, mid)
    l2, r2, s2 = max_subarray_dc(arr, mid + 1, hi)

    # Combine: find maximum crossing subarray
    l3, r3, s3 = max_crossing_subarray(arr, lo, mid, hi)

    # Return the best of the three
    if s1 >= s2 and s1 >= s3:
        return l1, r1, s1
    elif s2 >= s1 and s2 >= s3:
        return l2, r2, s2
    else:
        return l3, r3, s3
```

## Correctness

The algorithm is correct because it exhausts all possibilities for where the maximum subarray can lie. Every contiguous subarray of $A[\text{lo} \,..\, \text{hi}]$ either lies entirely in the left half, entirely in the right half, or crosses the midpoint. The recursive calls correctly handle the first two cases (by induction), and `MAX-CROSSING-SUBARRAY` correctly handles the third by independently optimizing the left and right extensions.

## Complexity Analysis

### Recurrence

Let $T(n)$ denote the running time on an array of size $n$. The algorithm:

- Divides in $O(1)$ time (computing the midpoint).
- Conquers by solving two subproblems of size $n/2$.
- Combines in $O(n)$ time (the crossing subarray computation).

The recurrence is

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n)
$$

### Solving the Recurrence

By the Master Theorem with $a = 2$, $b = 2$, and $f(n) = \Theta(n)$:

$$
\log_b a = \log_2 2 = 1
$$

Since $f(n) = \Theta(n^1)$, this is case 2:

$$
T(n) = \Theta(n \log n)
$$

### Space Complexity

The recursion depth is $O(\log n)$, and each level uses $O(1)$ auxiliary space (besides the recursive calls), giving $O(\log n)$ total space.

## Worked Example

Consider $A = [2, -4, 3, -1, 2, -5, 4]$ with $n = 7$.

**Top-level call** on $A[0..6]$, $\text{mid} = 3$:

- **Left** $A[0..3] = [2, -4, 3, -1]$: recursive call returns subarray $[3]$ with sum $3$.
- **Right** $A[4..6] = [2, -5, 4]$: recursive call returns subarray $[4]$ with sum $4$.
- **Crossing**: best left extension from index 3 is $A[2..3]$ with sum $2$; best right extension from index 4 is $A[4]$ with sum $2$. Crossing sum = $4$.

Maximum of $\{3, 4, 4\} = 4$, achieved by either the right subarray $[4]$ or the crossing subarray $A[2..4] = [3, -1, 2]$.

## Comparison with Kadane's Algorithm

| Property | Divide and Conquer | Kadane's Algorithm |
|---|---|---|
| Time complexity | $O(n \log n)$ | $O(n)$ |
| Space complexity | $O(\log n)$ | $O(1)$ |
| Paradigm | Divide and conquer | Dynamic programming |
| Parallelizable | Yes (left and right halves) | No (sequential scan) |
| Educational value | Illustrates D&C combine step | Illustrates DP and greedy |

Kadane's algorithm is strictly faster for the serial case, but the divide-and-conquer approach is more naturally parallelizable and serves as an excellent pedagogical example of the paradigm.

## Summary

The divide-and-conquer solution to the maximum subarray problem splits the array at the midpoint, recursively finds the maximum subarrays in each half, and combines by finding the maximum crossing subarray in $O(n)$ time. The resulting $O(n \log n)$ algorithm is slower than Kadane's $O(n)$ solution but provides a clean illustration of all three divide-and-conquer steps, especially the combine step that handles the crossing case.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
