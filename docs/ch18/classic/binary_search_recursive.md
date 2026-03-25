# Binary Search - Recursive

The [iterative binary search](binary_search.md) uses a `while` loop to narrow the search space. An equivalent **recursive** formulation expresses the same logic as a function that calls itself on a smaller subarray. The recursive version makes the divide-and-conquer structure of binary search explicit: at each call, the algorithm divides the array in half, conquers by recursing on one half, and combines by returning the result directly.

This page presents the recursive formulation, proves its correctness by structural induction, analyzes its time and space complexity, and compares it with the iterative version.

## Recursive Formulation

The recursive binary search takes the array $A$, the target $x$, and the current bounds $l$ and $r$ as parameters.

### Pseudocode

```
RECURSIVE-BINARY-SEARCH(A, x, l, r):
    if l > r:
        return NOT-FOUND
    m = floor((l + r) / 2)
    if A[m] == x:
        return m
    else if A[m] < x:
        return RECURSIVE-BINARY-SEARCH(A, x, m + 1, r)
    else:
        return RECURSIVE-BINARY-SEARCH(A, x, l, m - 1)
```

The initial call is `RECURSIVE-BINARY-SEARCH(A, x, 0, n - 1)`.

### Python Implementation

```python
def binary_search_recursive(arr, target, left=None, right=None):
    """
    Recursively search for target in a sorted array.

    Parameters
    ----------
    arr : list
        A sorted list of comparable elements.
    target : comparable
        The value to search for.
    left : int, optional
        Left boundary of the search range (default: 0).
    right : int, optional
        Right boundary of the search range (default: len(arr) - 1).

    Returns
    -------
    int or None
        The index of target if found, otherwise None.
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1

    if left > right:
        return None

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

## Correctness by Structural Induction

We prove correctness by strong induction on the size of the search space $s = r - l + 1$.

**Base case** ($s \le 0$). When $l > r$, the search space is empty. If $x$ were in $A[l \,..\, r]$, this subarray would be non-empty, so returning `NOT-FOUND` is correct.

**Inductive step.** Assume the algorithm is correct for all search spaces of size less than $s$. Consider a call with search space of size $s = r - l + 1 > 0$. Compute $m = \lfloor (l + r) / 2 \rfloor$.

- If $A[m] = x$: returning $m$ is correct.
- If $A[m] < x$: because $A$ is sorted, $x \notin A[l \,..\, m]$. The recursive call on $A[m+1 \,..\, r]$ has search space of size $r - m \le s - 1 < s$. By the inductive hypothesis, this call returns the correct answer.
- If $A[m] > x$: symmetrically, the recursive call on $A[l \,..\, m-1]$ has search space of size $m - l \le s - 1 < s$, and is correct by the inductive hypothesis.

In all cases, the algorithm returns the correct result. $\square$

## Complexity Analysis

### Time Complexity

The recursive binary search satisfies the recurrence

$$
T(n) = T\!\left(\frac{n}{2}\right) + O(1)
$$

with base case $T(0) = O(1)$. By the Master Theorem ($a = 1$, $b = 2$, $f(n) = O(1) = O(n^0)$, case 2):

$$
T(n) = O(\log n)
$$

This matches the iterative version exactly.

### Space Complexity

Each recursive call adds a frame to the call stack. Because the recursion depth is $O(\log n)$ (the search space halves at each call), the space complexity is

$$
S(n) = O(\log n)
$$

This is the key difference from the iterative version, which uses $O(1)$ auxiliary space. In practice, the $O(\log n)$ stack depth is small (e.g., $\log_2 10^9 \approx 30$ frames), so the overhead is rarely a concern.

!!! note "Tail Call Optimization"
    The recursive call is in **tail position** -- it is the last operation before the function returns. Languages that support tail call optimization (TCO), such as Scheme or certain C compilers with optimization flags, can transform the recursion into a loop, eliminating the stack overhead entirely. Python does not support TCO, so the $O(\log n)$ stack usage applies.

## Comparison: Iterative vs. Recursive

| Property | Iterative | Recursive |
|---|---|---|
| Time complexity | $O(\log n)$ | $O(\log n)$ |
| Space complexity | $O(1)$ | $O(\log n)$ |
| D&C structure | Implicit | Explicit |
| Tail call eligible | N/A | Yes |
| Stack overflow risk | None | Theoretical (depth $\approx 30$ for $n = 10^9$) |

Both versions are correct and have the same time complexity. The iterative version is generally preferred in production code for its $O(1)$ space usage. The recursive version is valuable for understanding the divide-and-conquer structure and serves as a template for more complex recursive algorithms.

## Worked Example

Search for $x = 12$ in $A = [3, 7, 12, 19, 25, 31, 42]$ ($n = 7$):

| Call | $l$ | $r$ | $m$ | $A[m]$ | Action |
|---|---|---|---|---|---|
| 1 | 0 | 6 | 3 | 19 | $19 > 12$, recurse on $[0, 2]$ |
| 2 | 0 | 2 | 1 | 7 | $7 < 12$, recurse on $[2, 2]$ |
| 3 | 2 | 2 | 2 | 12 | $12 = 12$, return $2$ |

The target is found at index 2 in 3 calls. The recursion unwinds, passing the result $2$ back through each frame.

## Summary

Recursive binary search makes the divide-and-conquer structure explicit: each call divides the search space in half, recurses on one half, and returns the result directly. It has the same $O(\log n)$ time complexity as the iterative version but uses $O(\log n)$ stack space. The correctness proof proceeds by strong induction on the search space size.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 2. MIT Press.
