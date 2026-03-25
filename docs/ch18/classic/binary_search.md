# Binary Search

Searching for a specific value in an unsorted array requires examining every element, taking $O(n)$ time in the worst case. When the array is **sorted**, however, we can exploit the ordering to eliminate half of the remaining elements at each step. This idea -- binary search -- is one of the simplest and most powerful applications of the divide-and-conquer paradigm, reducing the search time from $O(n)$ to $O(\log n)$.

## Problem Statement

Given a sorted array $A[0 \,..\, n-1]$ in non-decreasing order and a target value $x$, determine whether $x$ is present in $A$. If so, return an index $i$ such that $A[i] = x$; otherwise, indicate that $x$ is absent.

## The Algorithm

Binary search maintains two pointers, $l$ (left) and $r$ (right), that bound the subarray where $x$ could reside. At each step, it computes the midpoint $m = \lfloor (l + r) / 2 \rfloor$ and compares $A[m]$ with $x$:

- If $A[m] = x$, the search succeeds.
- If $A[m] < x$, then $x$ can only be in $A[m+1 \,..\, r]$, so set $l = m + 1$.
- If $A[m] > x$, then $x$ can only be in $A[l \,..\, m-1]$, so set $r = m - 1$.

The search terminates when $l > r$, meaning the search space is empty and $x$ is not in the array.

### Pseudocode

```
BINARY-SEARCH(A, x):
    l = 0
    r = n - 1
    while l <= r:
        m = floor((l + r) / 2)
        if A[m] == x:
            return m
        else if A[m] < x:
            l = m + 1
        else:
            r = m - 1
    return NOT-FOUND
```

### Python Implementation

```python
def binary_search(arr, target):
    """
    Search for target in a sorted array.

    Parameters
    ----------
    arr : list
        A sorted list of comparable elements.
    target : comparable
        The value to search for.

    Returns
    -------
    int or None
        The index of target if found, otherwise None.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # avoids integer overflow
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return None
```

!!! tip "Avoiding Integer Overflow"
    Computing the midpoint as `left + (right - left) // 2` instead of `(left + right) // 2` prevents integer overflow in languages with fixed-width integers (e.g., C, Java). In Python, integers have arbitrary precision, so both formulas are correct, but the safer form is a good habit.

## Correctness

We prove correctness using a **loop invariant**.

**Loop invariant.** At the start of each iteration of the `while` loop, if $x$ is in $A$, then $x \in A[l \,..\, r]$.

**Initialization.** Before the first iteration, $l = 0$ and $r = n - 1$, so the invariant holds trivially: if $x$ is in $A$, it is in $A[0 \,..\, n-1]$.

**Maintenance.** Suppose the invariant holds at the start of an iteration. We compute $m = \lfloor (l + r) / 2 \rfloor$.

- If $A[m] = x$, we return $m$ -- correct.
- If $A[m] < x$, then because $A$ is sorted, $x \notin A[l \,..\, m]$. Setting $l = m + 1$ preserves the invariant.
- If $A[m] > x$, then $x \notin A[m \,..\, r]$. Setting $r = m - 1$ preserves the invariant.

**Termination.** The loop terminates when $l > r$. By the invariant, if $x$ were in $A$, it would be in $A[l \,..\, r]$. But $l > r$ means this subarray is empty, so $x \notin A$. Returning `NOT-FOUND` is correct. $\square$

## Complexity Analysis

### Time Complexity

Each iteration halves the search space. After $k$ iterations, the remaining search space has size at most $\lfloor n / 2^k \rfloor$. The loop terminates when this size drops to zero, which happens when $2^k > n$, i.e., after $k = \lfloor \log_2 n \rfloor + 1$ iterations.

Each iteration performs $O(1)$ work (one comparison, one midpoint computation, one pointer update), so the total time is

$$
T(n) = O(\log n)
$$

Alternatively, as a recurrence: binary search solves one subproblem of size $n/2$ with $O(1)$ overhead:

$$
T(n) = T\!\left(\frac{n}{2}\right) + O(1)
$$

By the Master Theorem (case 2, with $a = 1$, $b = 2$, $\log_b a = 0$, and $f(n) = O(1) = O(n^0)$):

$$
T(n) = O(\log n)
$$

### Space Complexity

The iterative implementation uses $O(1)$ auxiliary space. The recursive version (covered on the [Binary Search - Recursive](binary_search_recursive.md) page) uses $O(\log n)$ stack space.

### Lower Bound

Any comparison-based search algorithm on a sorted array must make at least $\lceil \log_2(n + 1) \rceil$ comparisons in the worst case, because each comparison yields at most one bit of information, and there are $n + 1$ possible outcomes (found at one of $n$ positions, or not found). Binary search matches this lower bound and is therefore **optimal** among comparison-based search algorithms.

## Worked Example

Consider the sorted array $A = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]$ with $n = 10$, and suppose we search for $x = 23$.

| Iteration | $l$ | $r$ | $m$ | $A[m]$ | Action |
|---|---|---|---|---|---|
| 1 | 0 | 9 | 4 | 16 | $16 < 23$, set $l = 5$ |
| 2 | 5 | 9 | 7 | 56 | $56 > 23$, set $r = 6$ |
| 3 | 5 | 6 | 5 | 23 | $23 = 23$, return $5$ |

The search finds $x = 23$ at index 5 in 3 iterations, consistent with $\lfloor \log_2 10 \rfloor + 1 = 4$ maximum iterations.

## Divide-and-Conquer Perspective

Binary search is a divide-and-conquer algorithm with a degenerate structure:

- **Divide**: compute the midpoint in $O(1)$.
- **Conquer**: recurse on exactly one subproblem (either the left or right half).
- **Combine**: no work needed -- the answer from the subproblem is the answer to the original problem.

Because only one subproblem is solved at each level ($a = 1$), the total work is proportional to the depth of the recursion, which is $O(\log n)$.

## Summary

Binary search exploits the sorted order of an array to halve the search space at each step, achieving $O(\log n)$ time complexity. Its correctness follows from a simple loop invariant, and its efficiency matches the information-theoretic lower bound for comparison-based search.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 2. MIT Press.
