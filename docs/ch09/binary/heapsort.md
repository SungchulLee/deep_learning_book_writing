# Heapsort

Heapsort combines the heap data structure with a clever in-place strategy to sort an array in $O(n \log n)$ time using only $O(1)$ extra space. The algorithm works in two phases: first build a max-heap from the input array, then repeatedly extract the maximum and place it at the end of the array. Unlike mergesort, heapsort requires no auxiliary array; unlike quicksort, it guarantees $O(n \log n)$ worst-case time.

## Algorithm Overview

Heapsort consists of two phases:

**Phase 1 -- Build Heap**: transform the unordered input array into a max-heap using the bottom-up Build-Heap procedure. This takes $O(n)$ time.

**Phase 2 -- Repeated Extraction**: repeatedly swap the root (maximum element) with the last element of the unsorted region, shrink the heap by one, and sift-down to restore the heap property. Each extraction costs $O(\log n)$, and there are $n - 1$ extractions.

### Pseudocode

```
HEAPSORT(A):
    BUILD-MAX-HEAP(A)
    for i = n-1 down to 1:
        swap A[0] and A[i]
        MAX-HEAPIFY(A, 0, i)
```

After each iteration, `A[i..n-1]` contains the $n - i$ largest elements in sorted order, and `A[0..i-1]` is a valid max-heap.

## Complexity Analysis

| Phase | Operations | Cost |
|-------|-----------|------|
| Build Heap | One call to Build-Heap | $O(n)$ |
| Extraction | $n-1$ sift-down operations | $O(n \log n)$ |
| **Total** | | $O(n \log n)$ |

$$
T(n) = O(n) + (n-1) \cdot O(\log n) = O(n \log n)
$$

This bound is tight: heapsort performs $\Theta(n \log n)$ comparisons in the worst case.

**Space complexity**: $O(1)$ auxiliary space -- heapsort sorts in place.

## Step-by-Step Example

Sort the array `[4, 1, 3, 2, 16, 9, 10, 14, 8, 7]`:

```
Phase 1: Build max-heap
  [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]  →  [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]

Phase 2: Extract and place
  Swap 16 with 1, heapify [0..8]:  [14, 8, 10, 4, 7, 9, 3, 2, 1 | 16]
  Swap 14 with 1, heapify [0..7]:  [10, 8, 9, 4, 7, 1, 3, 2 | 14, 16]
  Swap 10 with 2, heapify [0..6]:  [9, 8, 3, 4, 7, 1, 2 | 10, 14, 16]
  Swap 9 with 2, heapify [0..5]:   [8, 7, 3, 4, 2, 1 | 9, 10, 14, 16]
  Swap 8 with 1, heapify [0..4]:   [7, 4, 3, 1, 2 | 8, 9, 10, 14, 16]
  Swap 7 with 2, heapify [0..3]:   [4, 2, 3, 1 | 7, 8, 9, 10, 14, 16]
  Swap 4 with 1, heapify [0..2]:   [3, 2, 1 | 4, 7, 8, 9, 10, 14, 16]
  Swap 3 with 1, heapify [0..1]:   [2, 1 | 3, 4, 7, 8, 9, 10, 14, 16]
  Swap 2 with 1, heapify [0..0]:   [1 | 2, 3, 4, 7, 8, 9, 10, 14, 16]

Result: [1, 2, 3, 4, 7, 8, 9, 10, 14, 16]
```

The `|` separator shows the boundary between the unsorted heap region (left) and the sorted region (right).

## Properties

| Property | Heapsort |
|----------|----------|
| Time (worst case) | $O(n \log n)$ |
| Time (average case) | $O(n \log n)$ |
| Time (best case) | $O(n \log n)$ |
| Space | $O(1)$ |
| Stable | No |
| In-place | Yes |
| Comparison-based | Yes |
| Adaptive | No |

!!! warning "Heapsort is Not Stable"
    Heapsort does not preserve the relative order of equal elements. The repeated swap-to-end operation can move equal elements past each other. If stability is required, mergesort or Timsort (Python's built-in sort) are better choices.

## Comparison with Other O(n log n) Sorts

| Algorithm | Worst Case | Space | Stable | Notes |
|-----------|-----------|-------|--------|-------|
| Heapsort | $O(n \log n)$ | $O(1)$ | No | Guaranteed worst case, poor cache behavior |
| Mergesort | $O(n \log n)$ | $O(n)$ | Yes | Stable, good for linked lists |
| Quicksort | $O(n^2)$ | $O(\log n)$ | No | Fastest in practice (average case), poor worst case |

Heapsort's main advantage is the combination of guaranteed $O(n \log n)$ worst-case time with $O(1)$ space. Its main disadvantage is poor cache locality: sift-down accesses array positions that double at each level ($i, 2i+1, 4i+3, \ldots$), causing frequent cache misses on large arrays.

## Implementation

```python
"""
Heapsort implementation.

Sorts an array in ascending order using a max-heap.
Phase 1: Build-Heap in O(n).
Phase 2: Repeated extract-max in O(n log n).
Total: O(n log n) time, O(1) space.
"""


# === Sift Down ===

def sift_down(arr, i, n):
    """Restore max-heap property at index i, considering only arr[0:n]."""
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest


# === Heapsort ===

def heapsort(arr):
    """Sort arr in ascending order using heapsort. In-place, O(n log n)."""
    n = len(arr)

    # Phase 1: Build max-heap in O(n)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, i, n)

    # Phase 2: Extract max and place at end
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        sift_down(arr, 0, i)


# === Demonstration ===

if __name__ == "__main__":
    # Sort example from CLRS
    data = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    print(f"Before: {data}")
    heapsort(data)
    print(f"After:  {data}")

    # Already sorted input (worst case for quicksort, fine for heapsort)
    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"\nAlready sorted: {data2}")
    heapsort(data2)
    print(f"After heapsort: {data2}")

    # Reverse sorted input
    data3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    print(f"\nReverse sorted: {data3}")
    heapsort(data3)
    print(f"After heapsort: {data3}")

    # All equal elements
    data4 = [5, 5, 5, 5, 5]
    print(f"\nAll equal: {data4}")
    heapsort(data4)
    print(f"After:     {data4}")
```

**Output:**
```
Before: [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
After:  [1, 2, 3, 4, 7, 8, 9, 10, 14, 16]

Already sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
After heapsort: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Reverse sorted: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
After heapsort: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

All equal: [5, 5, 5, 5, 5]
After:     [5, 5, 5, 5, 5]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.4: The heapsort algorithm. MIT Press.
