# Quickselect

Sorting an array to find the $k$-th smallest element costs $O(n \log n)$ -- far more than necessary. **Quickselect**, invented by Tony Hoare in 1961, adapts the quicksort partition to solve the selection problem in $O(n)$ expected time. The key insight is that after partitioning, we know which side of the pivot contains the $k$-th element, so we only need to recurse on **one** side instead of both. This halving of work at each step (on average) produces a geometric series that sums to $O(n)$.

## Algorithm

Given array $A[lo..hi]$ and target rank $k$ (0-indexed within this subarray):

1. If $lo = hi$, return $A[lo]$.
2. Choose a pivot (randomly for best expected performance).
3. Partition $A[lo..hi]$ around the pivot. Let the pivot land at position $p$.
4. If $k = p$, return $A[p]$ (the pivot is the answer).
5. If $k < p$, recurse on $A[lo..p-1]$.
6. If $k > p$, recurse on $A[p+1..hi]$.

Unlike quicksort, which recurses on **both** sides, quickselect recurses on only one. This is what reduces the expected total work from $O(n \log n)$ to $O(n)$.

## Expected Time Analysis

With a random pivot, the expected partition splits the array roughly in half. The total expected work is:

$$
E[T(n)] = n + \frac{1}{n} \sum_{q=0}^{n-1} E\!\left[T\!\left(\max(q, n - 1 - q)\right)\right]
$$

The worst-case recursive call covers the larger side. An upper bound uses the fact that a random pivot lands in the middle half with probability $1/2$, giving:

$$
E[T(n)] \leq n + \frac{3n}{4} + \frac{9n}{16} + \cdots = n \sum_{i=0}^{\infty} \left(\frac{3}{4}\right)^i = 4n
$$

Therefore $E[T(n)] = O(n)$. A tighter analysis gives $E[T(n)] \leq 3.39\, n + o(n)$.

## Worst Case

The worst case occurs when every pivot is the minimum or maximum element:

$$
T(n) = n + (n-1) + (n-2) + \cdots + 1 = \frac{n(n+1)}{2} = O(n^2)
$$

This happens with probability at most $O(1/n!)$ for random pivots, so it is practically negligible.

## Implementation

```python
"""
Quickselect: partition-based selection in O(n) expected time.

Finds the k-th smallest element by partitioning around a random
pivot and recursing on only the side that contains the target rank.
"""

import random


# === Partition ===

def partition(arr: list, lo: int, hi: int) -> int:
    """Lomuto partition with random pivot. Returns pivot index."""
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]

    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i


# === Quickselect ===

def quickselect(arr: list, k: int):
    """Find the k-th smallest element (1-indexed).

    Returns the element that would be at index k-1 in a sorted array.
    Operates on a copy to preserve the original.
    """
    if k < 1 or k > len(arr):
        raise ValueError(f"k={k} out of range for array of size {len(arr)}")

    data = arr.copy()
    return _quickselect(data, 0, len(data) - 1, k - 1)


def _quickselect(arr: list, lo: int, hi: int, k: int):
    """Recursive quickselect on arr[lo..hi] for rank k."""
    if lo == hi:
        return arr[lo]

    pivot_pos = partition(arr, lo, hi)

    if k == pivot_pos:
        return arr[k]
    elif k < pivot_pos:
        return _quickselect(arr, lo, pivot_pos - 1, k)
    else:
        return _quickselect(arr, pivot_pos + 1, hi, k)


# === Iterative Variant ===

def quickselect_iterative(arr: list, k: int):
    """Iterative quickselect using tail-call elimination."""
    data = arr.copy()
    lo, hi = 0, len(data) - 1
    k -= 1  # convert to 0-indexed

    while lo < hi:
        pivot_pos = partition(data, lo, hi)
        if k == pivot_pos:
            return data[k]
        elif k < pivot_pos:
            hi = pivot_pos - 1
        else:
            lo = pivot_pos + 1

    return data[lo]


# === Demonstration ===

if __name__ == "__main__":
    random.seed(42)

    data = [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
    print(f"Array:  {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in [1, 3, 5, 8, 10]:
        result_rec = quickselect(data, k)
        result_iter = quickselect_iterative(data, k)
        print(f"k={k:2d}: recursive={result_rec:3d}, "
              f"iterative={result_iter:3d}")

    print()
    print("Finding median:")
    n = len(data)
    median = quickselect(data, (n + 1) // 2)
    print(f"  Array size: {n}, median (k={( n + 1) // 2}): {median}")
```

**Output:**
```
Array:  [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
Sorted: [1, 3, 4, 5, 7, 8, 10, 12, 15, 20]

k= 1: recursive=  1, iterative=  1
k= 3: recursive=  4, iterative=  4
k= 5: recursive=  7, iterative=  7
k= 8: recursive= 12, iterative= 12
k=10: recursive= 20, iterative= 20

Finding median:
  Array size: 10, median (k=5): 7
```

## Complexity

| Case | Time | Space |
|------|------|-------|
| Best | $O(n)$ | $O(1)$ iterative / $O(\log n)$ recursive |
| Expected | $O(n)$ | $O(1)$ iterative / $O(\log n)$ recursive |
| Worst | $O(n^2)$ | $O(n)$ recursive |

!!! warning "Adversarial Inputs"
    Deterministic pivot choices (e.g., always first or last element) allow an adversary to force $O(n^2)$. Always use randomized pivot selection, or fall back to median-of-medians when the recursion depth exceeds a threshold.

## Reference

- Hoare, C. A. R. (1961). Algorithm 65: Find. *Communications of the ACM*, 4(7), 321-322.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 9. MIT Press.
