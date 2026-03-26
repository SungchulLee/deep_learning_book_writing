# Dual-Pivot Quicksort

Standard quicksort selects one pivot and splits the array into two partitions. A natural extension asks: what happens with **two pivots**? Dual-pivot quicksort selects two pivot elements $p$ and $q$ (with $p \leq q$) and partitions the array into three groups — elements less than $p$, elements between $p$ and $q$, and elements greater than $q$. Although this increases the number of comparisons per partitioning step, it reduces the expected recursion depth, and in practice the three-way split improves cache behavior. This variant, introduced by Vladimir Yaroslavskiy in 2009, is the default sorting algorithm in Java's `Arrays.sort` for primitive types.

## Partitioning Scheme

Given an array $A[lo..hi]$, choose two pivots $p = A[lo]$ and $q = A[hi]$, ensuring $p \leq q$ (swap them if necessary). The goal is to rearrange $A$ into three regions:

$$
A[lo..j-1] < p \leq A[j..k-1] \leq q < A[k..hi]
$$

Three pointers maintain the partition boundaries:

- **lt** (less-than pointer): everything in $A[lo+1..lt-1]$ is less than $p$.
- **gt** (greater-than pointer): everything in $A[gt+1..hi-1]$ is greater than $q$.
- **i** (scanning pointer): iterates from $lt$ to $gt$, classifying each element.

The scanning pointer $i$ processes each element exactly once, placing it into the correct region by swapping.

## Algorithm

1. If $A[lo] > A[hi]$, swap them so that $p \leq q$.
2. Initialize $lt = lo + 1$, $gt = hi - 1$, $i = lo + 1$.
3. While $i \leq gt$:
    - If $A[i] < p$: swap $A[i]$ with $A[lt]$, increment both $lt$ and $i$.
    - Else if $A[i] > q$: swap $A[i]$ with $A[gt]$, decrement $gt$ (do **not** increment $i$, since the swapped-in element has not been examined).
    - Else ($p \leq A[i] \leq q$): increment $i$.
4. Place the pivots in their final positions: swap $A[lo]$ with $A[lt-1]$ and $A[hi]$ with $A[gt+1]$.
5. Recurse on the three partitions: $A[lo..lt-2]$, $A[lt..gt]$, and $A[gt+2..hi]$.

## Complexity

| Case | Comparisons | Swaps |
|------|-------------|-------|
| Best | $O(n \log n)$ | $O(n \log n)$ |
| Average | $\approx 1.9\, n \ln n$ | $\approx 0.6\, n \ln n$ |
| Worst | $O(n^2)$ | $O(n^2)$ |

The average-case comparison count of $\approx 1.9\, n \ln n$ is slightly higher than the $\approx 1.39\, n \ln n$ of classic single-pivot quicksort. However, the average number of swaps is lower, and the three-way partitioning produces smaller subproblems on average, which improves cache performance on modern hardware.

!!! tip "Why Fewer Swaps Matter"
    On modern CPUs, memory access patterns dominate running time. The dual-pivot scheme reduces the total number of element movements and produces three smaller recursive subproblems, each of which fits in cache sooner. This explains why dual-pivot quicksort outperforms single-pivot quicksort in practice despite a higher comparison count.

## Implementation

```python
"""
Dual-pivot quicksort using the Yaroslavskiy partitioning scheme.

Partitions the array around two pivots into three regions,
then recurses on each region. This is the algorithm used by
Java's Arrays.sort for primitive types.
"""


# === Dual-Pivot Partition ===

def dual_pivot_partition(arr: list, lo: int, hi: int) -> tuple:
    """Partition arr[lo..hi] around two pivots.

    Returns (lt, gt) such that:
      - arr[lo..lt-1] < pivot1
      - arr[lt..gt]   between pivot1 and pivot2 (inclusive)
      - arr[gt+1..hi] > pivot2
    """
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]

    p, q = arr[lo], arr[hi]
    lt = lo + 1
    gt = hi - 1
    i = lo + 1

    while i <= gt:
        if arr[i] < p:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i += 1
        elif arr[i] > q:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    lt -= 1
    gt += 1
    arr[lo], arr[lt] = arr[lt], arr[lo]
    arr[hi], arr[gt] = arr[gt], arr[hi]

    return lt, gt


# === Dual-Pivot Quicksort ===

def dual_pivot_quicksort(arr: list, lo: int, hi: int) -> None:
    """Sort arr[lo..hi] in place using dual-pivot quicksort."""
    if lo >= hi:
        return

    lt, gt = dual_pivot_partition(arr, lo, hi)
    dual_pivot_quicksort(arr, lo, lt - 1)
    dual_pivot_quicksort(arr, lt + 1, gt - 1)
    dual_pivot_quicksort(arr, gt + 1, hi)


# === Demonstration ===

if __name__ == "__main__":
    data = [24, 8, 42, 75, 29, 77, 38, 57, 7, 53]
    print(f"Before: {data}")
    dual_pivot_quicksort(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # Show partitioning step on a small example
    example = [35, 10, 40, 20, 50, 30, 45]
    print(f"Partition example: {example}")
    lt, gt = dual_pivot_partition(example, 0, len(example) - 1)
    print(f"After partition:   {example}")
    print(f"Pivot positions:   lt={lt}, gt={gt}")
    print(f"Left pivot:  {example[lt]}")
    print(f"Right pivot: {example[gt]}")
```

**Output:**
```
Before: [24, 8, 42, 75, 29, 77, 38, 57, 7, 53]
After:  [7, 8, 24, 29, 38, 42, 53, 57, 75, 77]

Partition example: [35, 10, 40, 20, 50, 30, 45]
After partition:   [30, 10, 35, 20, 40, 45, 50]
Pivot positions:   lt=2, gt=5
Left pivot:  35
Right pivot: 45
```

## Comparison with Single-Pivot Quicksort

| Property | Single-Pivot | Dual-Pivot |
|----------|-------------|------------|
| Partitions per level | 2 | 3 |
| Avg comparisons | $\approx 1.39\, n \ln n$ | $\approx 1.9\, n \ln n$ |
| Avg swaps | $\approx 0.33\, n \ln n$ | $\approx 0.6\, n \ln n$ |
| Cache behavior | Good | Better (smaller subproblems) |
| Used in practice | C stdlib `qsort` | Java `Arrays.sort` (primitives) |

The dual-pivot variant wins in practice on modern hardware primarily because the three-way split produces smaller subproblems that fit in L1/L2 cache sooner, reducing costly cache misses.

## Reference

- Yaroslavskiy, V. (2009). *Dual-Pivot Quicksort*. [Research paper].
- Wild, S., & Nebel, M. E. (2012). Average case analysis of Java 7's dual pivot quicksort. *European Symposium on Algorithms*, 825-836.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
