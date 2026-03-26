# Median-of-Three Pivot Selection

Quicksort's performance depends heavily on pivot quality. Choosing the first or last element as the pivot leads to $O(n^2)$ behavior on sorted or nearly sorted input — precisely the inputs that appear most often in practice. **Median-of-three** pivot selection examines three elements (typically the first, middle, and last) and uses their median as the pivot. This simple heuristic avoids the worst case on sorted data and produces more balanced partitions on average.

## Motivation

Consider an array that is already sorted in ascending order. If we always pick $A[lo]$ as the pivot, every partition produces one empty subarray and one subarray of size $n - 1$, giving $T(n) = T(n-1) + \Theta(n) = \Theta(n^2)$. The median-of-three heuristic selects the median of $\{A[lo],\, A[\lfloor(lo+hi)/2\rfloor],\, A[hi]\}$. On a sorted array, this median is always the middle element, which produces perfectly balanced partitions with depth $O(\log n)$.

## Algorithm

Given indices $lo$ and $hi$, let $mid = \lfloor(lo + hi) / 2\rfloor$. The median-of-three procedure:

1. Compare $A[lo]$, $A[mid]$, and $A[hi]$.
2. Identify the element with the median value among these three.
3. Swap the median element into the pivot position (e.g., $A[hi]$ for Lomuto, $A[lo]$ for Hoare).
4. Proceed with the standard partition.

The three comparisons also partially sort the three elements as a side effect, which places sentinels at both ends of the subarray and eliminates the need for bounds checking in the inner loop of Hoare's partition.

## Analysis

The median of three random elements from a uniformly random permutation has expected rank $n/2$, which produces balanced partitions. More precisely, the probability that the pivot lands in the middle third of the array is:

$$
P\!\left(\frac{n}{3} \leq \text{rank} \leq \frac{2n}{3}\right) = \frac{11}{27} \approx 0.407
$$

compared to $1/3 \approx 0.333$ for a single random element. The expected number of comparisons drops from $\approx 1.386\, n \ln n$ (random pivot) to $\approx 1.188\, n \ln n$ (median-of-three).

!!! tip "Ninther (Median of Medians of Three)"
    For very large arrays, some implementations use the **ninther**: take three groups of three elements, find each group's median, then take the median of those three medians. This gives a better pivot estimate at the cost of more comparisons per partition, and is used in Bentley-McIlroy's engineering of quicksort.

## Implementation

```python
"""
Median-of-three pivot selection for quicksort.

Demonstrates how selecting the median of the first, middle,
and last elements avoids worst-case behavior on sorted inputs
and produces more balanced partitions on average.
"""


# === Median-of-Three Selection ===

def median_of_three(arr: list, lo: int, hi: int) -> int:
    """Return index of the median of arr[lo], arr[mid], arr[hi].

    As a side effect, partially sorts these three elements so that
    arr[lo] <= arr[mid] <= arr[hi].
    """
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]:
        arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]
    if arr[mid] > arr[hi]:
        arr[mid], arr[hi] = arr[hi], arr[mid]
    return mid


# === Quicksort with Median-of-Three ===

def quicksort_mot(arr: list, lo: int, hi: int) -> None:
    """Quicksort using median-of-three pivot selection."""
    if lo >= hi:
        return

    # Select pivot and move it to arr[hi - 1]
    pivot_idx = median_of_three(arr, lo, hi)
    arr[pivot_idx], arr[hi - 1] = arr[hi - 1], arr[pivot_idx]
    pivot = arr[hi - 1]

    # Hoare-like partition (arr[lo] and arr[hi] are sentinels)
    i = lo
    j = hi - 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            break
        arr[i], arr[j] = arr[j], arr[i]

    # Place pivot in final position
    arr[i], arr[hi - 1] = arr[hi - 1], arr[i]

    quicksort_mot(arr, lo, i - 1)
    quicksort_mot(arr, i + 1, hi)


# === Demonstration ===

if __name__ == "__main__":
    # Random input
    data = [38, 27, 43, 3, 9, 82, 10, 55, 1, 72]
    print(f"Before: {data}")
    quicksort_mot(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # Sorted input (worst case for naive pivot)
    sorted_data = list(range(1, 11))
    print(f"Sorted input: {sorted_data}")
    quicksort_mot(sorted_data, 0, len(sorted_data) - 1)
    print(f"After:        {sorted_data}")
    print()

    # Show pivot selection
    example = [50, 10, 30, 90, 70]
    print(f"Array: {example}")
    idx = median_of_three(example, 0, len(example) - 1)
    print(f"Median-of-three index: {idx}, value: {example[idx]}")
    print(f"After partial sort: {example}")
```

**Output:**
```
Before: [38, 27, 43, 3, 9, 82, 10, 55, 1, 72]
After:  [1, 3, 9, 10, 27, 38, 43, 55, 72, 82]

Sorted input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
After:        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Array: [50, 10, 30, 90, 70]
Median-of-three index: 2, value: 50
After partial sort: [30, 10, 50, 90, 70]
```

## Complexity

| Pivot Strategy | Avg Comparisons | Worst Case | Sorted Input |
|----------------|-----------------|------------|--------------|
| First element | $\approx 1.386\, n \ln n$ | $O(n^2)$ | $O(n^2)$ |
| Random element | $\approx 1.386\, n \ln n$ | $O(n^2)$ expected $O(n \log n)$ | $O(n \log n)$ expected |
| Median-of-three | $\approx 1.188\, n \ln n$ | $O(n^2)$ | $O(n \log n)$ |

Median-of-three does not eliminate the $O(n^2)$ worst case entirely — an adversary can still craft inputs that defeat it. However, such inputs do not arise naturally, and combining median-of-three with introsort's depth limit yields $O(n \log n)$ worst-case.

## Reference

- Sedgewick, R. (1978). Implementing Quicksort programs. *Communications of the ACM*, 21(10), 847-857.
- Bentley, J. L., & McIlroy, M. D. (1993). Engineering a sort function. *Software: Practice and Experience*, 23(11), 1249-1265.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
