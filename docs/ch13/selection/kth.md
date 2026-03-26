# K-th Smallest Element

Finding the $k$-th smallest element in an unsorted array is one of the fundamental problems in computer science. The naive approach sorts the array first, then reads off position $k$, costing $O(n \log n)$. But sorting does far more work than necessary -- we only need a single element, not the entire sorted order. The **selection problem** asks: can we find the $k$-th smallest in $O(n)$ time? The answer is yes, using partition-based algorithms that narrow the search to one side of a pivot at each step.

## Problem Statement

Given an unsorted array $A$ of $n$ elements and an integer $k$ with $1 \leq k \leq n$, find the element that would appear at position $k$ if the array were sorted in non-decreasing order.

Special cases with well-known names:

- $k = 1$: **minimum** (trivially $O(n)$ with a linear scan)
- $k = n$: **maximum** (also $O(n)$)
- $k = \lfloor (n+1)/2 \rfloor$: **median** (the hardest special case)

## Approaches

### Sorting-Based

Sort the array and return $A[k-1]$. Time: $O(n \log n)$. Simple but suboptimal for a single query.

### Heap-Based

Build a min-heap in $O(n)$, then extract the minimum $k$ times. Time: $O(n + k \log n)$. This is efficient when $k$ is small (e.g., the 5th smallest) but degrades to $O(n \log n)$ when $k = n/2$.

### Partition-Based (Quickselect)

Partition the array around a pivot. If the pivot lands at position $k$, return it. Otherwise, recurse on the side that contains position $k$. Expected time: $O(n)$, worst case $O(n^2)$.

### Deterministic Linear (Median of Medians)

Choose the pivot using the median-of-medians algorithm to guarantee $O(n)$ worst case. This is covered in detail on the dedicated pages for quickselect and median-of-medians.

## Lower Bound

Any comparison-based algorithm for finding the $k$-th smallest element requires at least:

$$
n - 1 \text{ comparisons for } k = 1 \text{ or } k = n
$$

For the general case, the information-theoretic lower bound is $\Omega(n)$, since every element must be examined at least once (an unseen element might be the answer). The partition-based algorithms achieve this bound.

## Implementation

```python
"""
K-th smallest element selection using partition-based approach.

Demonstrates both naive (sort-based) and quickselect approaches.
The quickselect variant achieves O(n) expected time by recursing
on only one side of the partition.
"""

import random


# === Naive Selection (Sort-Based) ===

def kth_smallest_sort(arr: list, k: int):
    """Find k-th smallest by sorting. O(n log n)."""
    return sorted(arr)[k - 1]


# === Quickselect ===

def kth_smallest(arr: list, k: int):
    """Find k-th smallest using randomized quickselect.

    Operates on a copy to avoid modifying the original.
    Expected O(n) time.
    """
    if k < 1 or k > len(arr):
        raise ValueError(f"k={k} out of range for array of size {len(arr)}")

    data = arr.copy()
    return _quickselect(data, 0, len(data) - 1, k - 1)


def _quickselect(arr: list, lo: int, hi: int, k: int):
    """Return element at index k in arr[lo..hi] if it were sorted."""
    if lo == hi:
        return arr[lo]

    pivot_idx = _partition(arr, lo, hi)

    if k == pivot_idx:
        return arr[k]
    elif k < pivot_idx:
        return _quickselect(arr, lo, pivot_idx - 1, k)
    else:
        return _quickselect(arr, pivot_idx + 1, hi, k)


def _partition(arr: list, lo: int, hi: int) -> int:
    """Lomuto partition with random pivot."""
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


# === Demonstration ===

if __name__ == "__main__":
    random.seed(42)

    data = [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
    print(f"Array: {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in [1, 3, 5, 7, 10]:
        result = kth_smallest(data, k)
        print(f"k={k:2d}: {result}")

    print()
    print(f"Minimum (k=1):  {kth_smallest(data, 1)}")
    print(f"Maximum (k=10): {kth_smallest(data, 10)}")
    print(f"Median (k=5):   {kth_smallest(data, 5)}")
```

**Output:**
```
Array: [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
Sorted: [1, 3, 4, 5, 7, 8, 10, 12, 15, 20]

k= 1: 1
k= 3: 4
k= 5: 7
k= 7: 10
k=10: 20

Minimum (k=1):  1
Maximum (k=10): 20
Median (k=5):   7
```

## Complexity Comparison

| Method | Time (expected) | Time (worst) | Space |
|--------|----------------|--------------|-------|
| Sort + index | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ |
| Min-heap + extract | $O(n + k \log n)$ | $O(n + k \log n)$ | $O(n)$ |
| Quickselect | $O(n)$ | $O(n^2)$ | $O(\log n)$ expected |
| Median-of-medians | $O(n)$ | $O(n)$ | $O(\log n)$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 9. MIT Press.
- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
