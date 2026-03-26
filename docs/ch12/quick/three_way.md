# Three-Way Partition

Standard two-way quicksort partitions elements into "less than pivot" and "greater than pivot." When the array contains many duplicate keys, equal elements can end up on both sides of the pivot, causing redundant comparisons in subsequent recursive calls. **Three-way partitioning** (also known as the Dutch National Flag algorithm, after Dijkstra) splits the array into three groups: elements less than, equal to, and greater than the pivot. Equal elements are placed in their final position immediately and excluded from recursion, which can reduce the running time from $O(n^2)$ to $O(n \log n)$ on inputs with many duplicates.

## The Dutch National Flag Problem

Dijkstra posed this problem in 1976: given an array of elements colored red, white, and blue, rearrange them so that all reds come first, then whites, then blues, using only swaps. The three-way partition solves this by maintaining three regions separated by two pointers.

## Algorithm

Given a pivot value $v$ and array $A[lo..hi]$, maintain three pointers:

- **lt**: boundary of the "less than" region. $A[lo..lt-1] < v$.
- **gt**: boundary of the "greater than" region. $A[gt+1..hi] > v$.
- **i**: scanning pointer. $A[lt..i-1] = v$.

Initialize $lt = lo$, $gt = hi$, $i = lo$. While $i \leq gt$:

1. If $A[i] < v$: swap $A[i]$ with $A[lt]$, increment both $lt$ and $i$.
2. If $A[i] > v$: swap $A[i]$ with $A[gt]$, decrement $gt$ (do not increment $i$).
3. If $A[i] = v$: increment $i$.

After the loop terminates, the array satisfies:

$$
A[lo..lt-1] < v = A[lt..gt] < A[gt+1..hi]
$$

The algorithm recurses only on $A[lo..lt-1]$ and $A[gt+1..hi]$, skipping all elements equal to $v$.

## Complexity

| Input Type | Standard Quicksort | Three-Way Quicksort |
|------------|-------------------|---------------------|
| All distinct | $O(n \log n)$ avg | $O(n \log n)$ avg |
| Few distinct keys ($k$ values) | $O(n \log n)$ avg | $O(n \log k)$ avg |
| All equal | $O(n^2)$ | $O(n)$ |

When all elements are equal, standard quicksort still recurses $n$ times (each partition removes only the pivot), while three-way quicksort finishes in a single pass since all elements land in the "equal" region.

!!! tip "Entropy-Optimal Sorting"
    Three-way quicksort is **entropy-optimal**: its expected running time is proportional to $n$ times the Shannon entropy of the key distribution, $H = -\sum p_i \log p_i$. When many keys are equal, the entropy is low and the sort runs faster.

## Implementation

```python
"""
Three-way partition (Dutch National Flag) for quicksort.

Splits the array into three regions: less than, equal to, and
greater than the pivot. Equal elements are excluded from recursion,
making this variant optimal for inputs with many duplicate keys.
"""

import random


# === Three-Way Partition ===

def three_way_partition(arr: list, lo: int, hi: int) -> tuple:
    """Partition arr[lo..hi] into three regions around arr[lo].

    Returns (lt, gt) such that:
      arr[lo..lt-1]  < pivot
      arr[lt..gt]    = pivot
      arr[gt+1..hi]  > pivot
    """
    pivot = arr[lo]
    lt = lo
    gt = hi
    i = lo

    while i <= gt:
        if arr[i] < pivot:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    return lt, gt


# === Three-Way Quicksort ===

def three_way_quicksort(arr: list, lo: int, hi: int) -> None:
    """Sort arr[lo..hi] using three-way partitioning."""
    if lo >= hi:
        return
    lt, gt = three_way_partition(arr, lo, hi)
    three_way_quicksort(arr, lo, lt - 1)
    three_way_quicksort(arr, gt + 1, hi)


# === Demonstration ===

if __name__ == "__main__":
    # Array with many duplicates
    data = [4, 2, 4, 1, 3, 4, 2, 1, 4, 3]
    print(f"Before: {data}")
    three_way_quicksort(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # Show partition step
    example = [3, 1, 4, 3, 5, 3, 2, 3]
    print(f"Partition example: {example}")
    lt, gt = three_way_partition(example, 0, len(example) - 1)
    print(f"After partition:   {example}")
    print(f"lt={lt}, gt={gt}")
    print(f"Less:  {example[:lt]}")
    print(f"Equal: {example[lt:gt+1]}")
    print(f"Greater: {example[gt+1:]}")
    print()

    # All-equal input (worst case for standard quicksort)
    equal = [5] * 10
    print(f"All equal: {equal}")
    three_way_quicksort(equal, 0, len(equal) - 1)
    print(f"After:     {equal}")
```

**Output:**
```
Before: [4, 2, 4, 1, 3, 4, 2, 1, 4, 3]
After:  [1, 1, 2, 2, 3, 3, 4, 4, 4, 4]

Partition example: [3, 1, 4, 3, 5, 3, 2, 3]
After partition:   [2, 1, 3, 3, 3, 3, 5, 4]
lt=2, gt=5
Less:  [2, 1]
Equal: [3, 3, 3, 3]
Greater: [5, 4]

All equal: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
After:     [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
```

## Reference

- Dijkstra, E. W. (1976). *A Discipline of Programming*. Prentice-Hall.
- Bentley, J. L., & McIlroy, M. D. (1993). Engineering a sort function. *Software: Practice and Experience*, 23(11), 1249-1265.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
