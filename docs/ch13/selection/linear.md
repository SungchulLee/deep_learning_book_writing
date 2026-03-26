# Linear-Time Selection

Quickselect finds the $k$-th smallest element in $O(n)$ expected time, but its worst case is $O(n^2)$ when the pivot consistently produces unbalanced partitions. **Linear-time selection** (the BFPRT algorithm, after Blum, Floyd, Pratt, Rivest, and Tarjan, 1973) guarantees $O(n)$ worst-case time by choosing a pivot that is guaranteed to eliminate a constant fraction of elements at each step. The key technique is the **median-of-medians**: divide the array into groups of five, find each group's median, and recursively select the median of those medians as the pivot.

## Algorithm Overview

The SELECT algorithm finds the $k$-th smallest element of $A[1..n]$:

1. **Divide** the $n$ elements into $\lceil n/5 \rceil$ groups of 5 (the last group may have fewer).
2. **Find the median** of each group by sorting the group (at most 6 comparisons per group).
3. **Recursively select** the median $m$ of the $\lceil n/5 \rceil$ group medians.
4. **Partition** the array around $m$. Let $m$ end up at position $q$.
5. If $k = q$, return $m$. If $k < q$, recurse on the left side. If $k > q$, recurse on the right side.

## Why Groups of Five

The choice of 5 is the smallest odd group size that makes the recurrence work out to $O(n)$. The median-of-medians $m$ is greater than roughly $3n/10$ elements and less than roughly $3n/10$ elements. This means the worst-case recursive call operates on at most $7n/10$ elements.

The recurrence is:

$$
T(n) \leq T(\lceil n/5 \rceil) + T(7n/10) + O(n)
$$

Since $n/5 + 7n/10 = 9n/10 < n$, this solves to $T(n) = O(n)$.

!!! tip "Why Not Groups of Three?"
    With groups of 3, the median-of-medians guarantees eliminating only $n/4$ elements per step, giving the recurrence $T(n) \leq T(n/3) + T(3n/4) + O(n)$. Since $1/3 + 3/4 = 13/12 > 1$, this does not solve to $O(n)$.

## Guarantee Analysis

Among the $\lceil n/5 \rceil$ group medians, at least half are $\leq m$ and at least half are $\geq m$. Each such group median has 2 elements below it in its group (since it is a median of 5). Therefore, the number of elements guaranteed to be $\leq m$ is at least:

$$
3 \cdot \left\lfloor \frac{1}{2} \cdot \lceil n/5 \rceil \right\rfloor \geq \frac{3n}{10} - 6
$$

Symmetrically, at least $3n/10 - 6$ elements are $\geq m$. The recursive call on the larger partition has at most $7n/10 + 6$ elements.

## Implementation

```python
"""
Worst-case linear-time selection (BFPRT / median-of-medians).

Guarantees O(n) time for finding the k-th smallest element by
choosing a pivot via the median-of-medians technique, which
ensures at least 30% of elements are eliminated at each step.
"""


# === Insertion Sort for Small Groups ===

def insertion_sort(arr: list, lo: int, hi: int) -> None:
    """Sort arr[lo..hi] in place using insertion sort."""
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# === Median of Medians ===

def median_of_medians(arr: list, lo: int, hi: int) -> int:
    """Find a good pivot using the median-of-medians technique.

    Returns the value (not index) of the pivot.
    """
    n = hi - lo + 1
    if n <= 5:
        insertion_sort(arr, lo, hi)
        return arr[lo + n // 2]

    # Divide into groups of 5 and find each group's median
    medians = []
    for i in range(lo, hi + 1, 5):
        group_end = min(i + 4, hi)
        insertion_sort(arr, i, group_end)
        medians.append(arr[i + (group_end - i) // 2])

    # Recursively find the median of the group medians
    return select(medians, len(medians) // 2)


# === Linear-Time Selection ===

def select(arr: list, k: int):
    """Find the k-th smallest element (0-indexed) in O(n) worst case."""
    if len(arr) <= 5:
        return sorted(arr)[k]

    pivot = median_of_medians(arr, 0, len(arr) - 1)

    # Three-way partition around pivot
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    if k < len(less):
        return select(less, k)
    elif k < len(less) + len(equal):
        return pivot
    else:
        return select(greater, k - len(less) - len(equal))


# === Demonstration ===

if __name__ == "__main__":
    data = [12, 3, 5, 7, 4, 19, 26, 1, 8, 15, 20, 11, 9, 2, 6]
    print(f"Array:  {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in range(len(data)):
        result = select(data.copy(), k)
        print(f"k={k:2d} (rank {k+1:2d}): {result}")
```

**Output:**
```
Array:  [12, 3, 5, 7, 4, 19, 26, 1, 8, 15, 20, 11, 9, 2, 6]
Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 19, 20, 26]

k= 0 (rank  1): 1
k= 1 (rank  2): 2
k= 2 (rank  3): 3
k= 3 (rank  4): 4
k= 4 (rank  5): 5
k= 5 (rank  6): 6
k= 6 (rank  7): 7
k= 7 (rank  8): 8
k= 8 (rank  9): 9
k= 9 (rank 10): 11
k=10 (rank 11): 12
k=11 (rank 12): 15
k=12 (rank 13): 19
k=13 (rank 14): 20
k=14 (rank 15): 26
```

## Complexity

| Property | Value |
|----------|-------|
| Time (worst case) | $O(n)$ |
| Time (average) | $O(n)$ |
| Space | $O(\log n)$ (recursion stack) |
| Comparisons | $\leq 5.43\, n + o(n)$ |

!!! warning "Practical Performance"
    Although $O(n)$ worst case, the constant factor in median-of-medians is large (roughly 5x compared to quickselect). In practice, randomized quickselect is faster on average and is preferred unless a worst-case guarantee is required. Many practical implementations use quickselect with a fallback to median-of-medians only when the recursion depth exceeds a threshold.

## Reference

- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 9. MIT Press.
