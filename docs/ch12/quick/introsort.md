# Introsort

Quicksort is fast on average but degrades to $O(n^2)$ on adversarial inputs. Heapsort guarantees $O(n \log n)$ worst-case but has poor cache locality. Insertion sort is optimal for tiny arrays. **Introsort** (introspective sort), introduced by David Musser in 1997, combines all three algorithms: it begins with quicksort, monitors the recursion depth, and switches to heapsort when the depth exceeds $2 \lfloor \log_2 n \rfloor$. For small subproblems, it falls back to insertion sort. This hybrid strategy achieves $O(n \log n)$ worst-case time while retaining quicksort's practical speed. Introsort is the algorithm behind C++'s `std::sort`.

## Strategy

The key insight is that quicksort's worst case is triggered by deep recursion caused by unbalanced partitions. By counting the recursion depth and aborting quicksort when it exceeds a threshold, introsort avoids the $O(n^2)$ trap.

The algorithm proceeds in three phases:

1. **Quicksort phase**: Partition the array using a standard pivot selection (typically median-of-three). Recurse on both halves, decrementing a depth counter.
2. **Heapsort fallback**: When the depth counter reaches zero, stop recursing and sort the current subarray with heapsort. This guarantees $O(n \log n)$ even if every partition is maximally unbalanced.
3. **Insertion sort finish**: When a subarray has fewer than 16 elements, sort it with insertion sort. Insertion sort's $O(n^2)$ cost is negligible on tiny arrays, and its low overhead makes it faster than quicksort for small $n$.

## Depth Limit

The depth limit is typically set to:

$$
d_{\max} = 2 \lfloor \log_2 n \rfloor
$$

This value is chosen because balanced quicksort reaches depth $\log_2 n$. A factor of 2 provides enough slack for mildly unbalanced partitions while still catching pathological cases early.

## Complexity

| Case | Time | Space |
|------|------|-------|
| Best | $O(n \log n)$ | $O(\log n)$ |
| Average | $O(n \log n)$ | $O(\log n)$ |
| Worst | $O(n \log n)$ | $O(\log n)$ |

The worst-case guarantee comes from the heapsort fallback. The average case matches standard quicksort because the depth limit is rarely reached on random inputs.

## Implementation

```python
"""
Introsort: hybrid of quicksort, heapsort, and insertion sort.

Achieves O(n log n) worst-case by monitoring recursion depth and
falling back to heapsort when the depth exceeds 2 * floor(log2(n)).
Small subarrays are finished with insertion sort.
"""

import math


# === Insertion Sort (for small subarrays) ===

def insertion_sort(arr: list, lo: int, hi: int) -> None:
    """Sort arr[lo..hi] in place using insertion sort."""
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# === Heapsort (fallback for deep recursion) ===

def heapsort(arr: list, lo: int, hi: int) -> None:
    """Sort arr[lo..hi] in place using heapsort."""
    n = hi - lo + 1

    def sift_down(start: int, end: int) -> None:
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            if child + 1 <= end and arr[lo + child] < arr[lo + child + 1]:
                child += 1
            if arr[lo + root] < arr[lo + child]:
                arr[lo + root], arr[lo + child] = arr[lo + child], arr[lo + root]
                root = child
            else:
                break

    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        sift_down(i, n - 1)

    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[lo], arr[lo + i] = arr[lo + i], arr[lo]
        sift_down(0, i - 1)


# === Median-of-Three Pivot ===

def median_of_three(arr: list, lo: int, hi: int) -> int:
    """Return index of median of arr[lo], arr[mid], arr[hi]."""
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]:
        arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]
    if arr[mid] > arr[hi]:
        arr[mid], arr[hi] = arr[hi], arr[mid]
    return mid


# === Introsort ===

def introsort(arr: list) -> None:
    """Sort arr in place using introsort."""
    if len(arr) <= 1:
        return
    max_depth = 2 * math.floor(math.log2(len(arr)))
    _introsort_impl(arr, 0, len(arr) - 1, max_depth)


SIZE_THRESHOLD = 16


def _introsort_impl(arr: list, lo: int, hi: int, depth_limit: int) -> None:
    """Recursive introsort with depth tracking."""
    while hi - lo + 1 > SIZE_THRESHOLD:
        if depth_limit == 0:
            heapsort(arr, lo, hi)
            return

        depth_limit -= 1
        pivot_idx = median_of_three(arr, lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]

        # Lomuto-style partition
        pivot = arr[hi]
        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]

        # Recurse on the smaller side; loop on the larger
        if i - lo < hi - i:
            _introsort_impl(arr, lo, i - 1, depth_limit)
            lo = i + 1
        else:
            _introsort_impl(arr, i + 1, hi, depth_limit)
            hi = i - 1

    insertion_sort(arr, lo, hi)


# === Demonstration ===

if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10, 55, 1, 72, 64, 29]
    print(f"Before: {data}")
    introsort(data)
    print(f"After:  {data}")
    print()

    # Worst case for naive quicksort — already sorted
    worst = list(range(20, 0, -1))
    print(f"Reverse-sorted input: {worst}")
    introsort(worst)
    print(f"After introsort:      {worst}")
```

**Output:**
```
Before: [38, 27, 43, 3, 9, 82, 10, 55, 1, 72, 64, 29]
After:  [1, 3, 9, 10, 27, 29, 38, 43, 55, 64, 72, 82]

Reverse-sorted input: [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
After introsort:      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

## Why Not Just Use Heapsort?

If heapsort already guarantees $O(n \log n)$, why not use it directly? The answer is **cache efficiency**. Quicksort accesses memory sequentially and benefits from spatial locality, while heapsort's parent-child jumps in the heap array cause frequent cache misses. On random data, introsort spends almost all of its time in the quicksort phase, falling back to heapsort only on adversarial inputs.

!!! tip "Practical Threshold"
    The insertion sort threshold of 16 is empirically chosen. Values between 8 and 32 work well. Below 8, the overhead of function calls dominates; above 32, insertion sort's quadratic cost becomes noticeable.

## Reference

- Musser, D. R. (1997). Introspective sorting and selection algorithms. *Software: Practice and Experience*, 27(8), 983-993.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
