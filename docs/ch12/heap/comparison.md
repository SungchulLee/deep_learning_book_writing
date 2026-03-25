# Heapsort Comparison with Other Sorts

Choosing a sorting algorithm involves balancing worst-case guarantees, average-case speed, memory usage, stability, and cache behavior.  Heapsort occupies a unique niche: it offers $O(n \log n)$ worst-case time with $O(1)$ auxiliary space, but its constant factors and cache performance make it slower in practice than quicksort for most inputs.  This page compares heapsort with the other major $O(n \log n)$ algorithms.

## Asymptotic Comparison

$$
\begin{array}{lcccc}
\textbf{Algorithm} & \textbf{Best} & \textbf{Average} & \textbf{Worst} & \textbf{Space} \\
\hline
\text{Heapsort}    & O(n \log n) & O(n \log n) & O(n \log n) & O(1) \\
\text{Merge sort}  & O(n \log n) & O(n \log n) & O(n \log n) & O(n) \\
\text{Quicksort}   & O(n \log n) & O(n \log n) & O(n^2)      & O(\log n) \\
\text{Timsort}     & O(n)        & O(n \log n) & O(n \log n) & O(n) \\
\text{Insertion sort} & O(n)     & O(n^2)      & O(n^2)      & O(1)
\end{array}
$$

All three $O(n \log n)$ algorithms match the comparison-based lower bound of $\Omega(n \log n)$, but they differ in which cases achieve the bound and how much extra memory they require.

## Heapsort vs Quicksort

Quicksort is typically 2-3 times faster than heapsort on random data despite sharing the same asymptotic average-case complexity.  The reasons are architectural rather than algorithmic:

**Cache locality.**  Quicksort's partition scans the array sequentially, producing excellent spatial locality.  Heapsort's `sift_down` jumps between parent and child indices ($i \to 2i+1$), causing frequent cache misses on large arrays.

**Branch prediction.**  During partitioning, quicksort compares every element against the same pivot value, and the branch outcome is roughly random -- modern branch predictors handle this well.  Heapsort alternates between comparing children and comparing child with parent, producing a less predictable pattern.

**Swap count.**  Quicksort performs fewer swaps on average.  Heapsort moves every element at least once during the build-heap phase and again during extraction.

!!! tip "When to prefer heapsort over quicksort"
    Use heapsort when you need a guaranteed $O(n \log n)$ worst case without the $O(n)$ extra memory that merge sort requires.  Examples include real-time systems and embedded environments where both time and space budgets are strict.

## Heapsort vs Merge Sort

Both algorithms guarantee $O(n \log n)$ worst-case time, but they make different trade-offs.

| Property        | Heapsort       | Merge sort     |
|-----------------|----------------|----------------|
| Auxiliary space | $O(1)$         | $O(n)$         |
| Stability       | Not stable     | Stable         |
| Cache behavior  | Poor           | Good           |
| Comparison count| ~$2n \log n$   | ~$n \log n$    |
| Parallelism     | Hard to parallelize | Naturally parallel |

Merge sort performs roughly half as many comparisons as heapsort because each merge comparison places one element, whereas `sift_down` uses two comparisons per level.  When comparison cost is high (e.g., comparing long strings), merge sort has a clear advantage.

## Heapsort vs Timsort

Timsort is the default sorting algorithm in Python (`sorted()` and `list.sort()`) and Java (`Arrays.sort()` for objects).  It exploits existing order in the input:

- **Already sorted data**: Timsort runs in $O(n)$; heapsort still takes $O(n \log n)$.
- **Nearly sorted data**: Timsort detects natural runs and merges them efficiently.
- **Random data**: Both are $O(n \log n)$, but Timsort's merge-based approach has better cache behavior.

Heapsort's advantage is its $O(1)$ space, whereas Timsort requires $O(n)$ auxiliary space for merging.

## Stability

A sorting algorithm is **stable** if it preserves the relative order of elements with equal keys.

- **Heapsort**: Not stable.  During extraction, elements are swapped to the end of the array, disrupting the original order of equal elements.
- **Merge sort**: Stable (when ties are broken by taking from the left subarray first).
- **Quicksort**: Not stable in its standard form (Lomuto or Hoare partition both may reorder equal elements).
- **Timsort**: Stable.

When stability is a hard requirement, merge sort or Timsort should be preferred over heapsort.

## Practical Hybrid Approaches

Modern sorting implementations rarely use a single algorithm.  Instead, they combine strengths:

- **Introsort** (used in C++ `std::sort`): starts with quicksort, switches to heapsort if the recursion depth exceeds $2 \lfloor \log_2 n \rfloor$, and uses insertion sort for small partitions.  This guarantees $O(n \log n)$ worst case while retaining quicksort's average-case speed.
- **Timsort**: combines merge sort with insertion sort for small runs.
- **Pattern-defeating quicksort (pdqsort)**: detects adversarial patterns and falls back to heapsort.

In all three hybrids, heapsort serves as the **safety net** that prevents worst-case degradation.

## Python Demonstration

```python
"""
Comparison of sorting algorithm performance.

Times heapsort, merge sort, and Python's built-in Timsort on random
arrays to illustrate the practical constant-factor differences.
"""

import random
import time


# === Heapsort =================================================================

def heapsort(arr: list) -> list:
    """In-place heapsort returning the sorted list."""
    a = arr[:]
    n = len(a)

    def sift_down(i: int, size: int) -> None:
        largest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < size and a[left] > a[largest]:
            largest = left
        if right < size and a[right] > a[largest]:
            largest = right
        if largest != i:
            a[i], a[largest] = a[largest], a[i]
            sift_down(largest, size)

    for i in range(n // 2 - 1, -1, -1):
        sift_down(i, n)
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        sift_down(0, i)
    return a


# === Merge sort ===============================================================

def merge_sort(arr: list) -> list:
    """Top-down merge sort returning a new sorted list."""
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# === Benchmark ================================================================

def benchmark(sort_fn, arr: list) -> float:
    """Return the wall-clock time in milliseconds for sorting arr."""
    start = time.perf_counter()
    sort_fn(arr)
    return (time.perf_counter() - start) * 1000


# === Main =====================================================================

if __name__ == "__main__":
    n = 10000
    data = list(range(n))
    random.shuffle(data)

    t_heap = benchmark(heapsort, data)
    t_merge = benchmark(merge_sort, data)
    t_tim = benchmark(sorted, data)

    print(f"n = {n}")
    print(f"Heapsort:   {t_heap:8.2f} ms")
    print(f"Merge sort: {t_merge:8.2f} ms")
    print(f"Timsort:    {t_tim:8.2f} ms")
```

**Output (typical, hardware-dependent):**
```
n = 10000
Heapsort:     18.42 ms
Merge sort:   12.65 ms
Timsort:       1.23 ms
```

Timsort's large advantage comes from being implemented in C within CPython, while heapsort and merge sort here are pure Python.  Even with equal implementation effort, however, heapsort's cache-unfriendly access pattern makes it consistently slower than the alternatives on modern hardware.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Chapter 6.
- Musser, D. R. (1997). Introspective sorting and selection algorithms. *Software: Practice and Experience*, 27(8), 983-993.
- Peters, T. (2002). Timsort description. CPython source: `Objects/listsort.txt`.
