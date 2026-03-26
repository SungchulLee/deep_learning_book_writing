# Randomized Quicksort

Deterministic quicksort with a fixed pivot rule (e.g., always pick the first element) can be forced into $O(n^2)$ by an adversary who constructs the input to produce maximally unbalanced partitions. **Randomized quicksort** eliminates this vulnerability by choosing the pivot uniformly at random from the subarray. Because the pivot is random, no fixed input can consistently trigger worst-case behavior. The expected running time becomes $O(n \log n)$ for every input, not just "on average over random inputs."

## Key Idea

The only change from deterministic quicksort is the pivot selection step. Before partitioning $A[lo..hi]$, the algorithm picks a uniformly random index $r \in [lo, hi]$ and swaps $A[r]$ with $A[hi]$ (or $A[lo]$, depending on the partition scheme). The partition then proceeds exactly as in Lomuto or Hoare.

Because the pivot is equally likely to be any element, the expected depth of the recursion tree is $O(\log n)$, and the expected total number of comparisons is:

$$
E[C(n)] = 2n \ln n + O(n) \approx 1.386\, n \log_2 n
$$

This expectation is taken over the algorithm's random choices, not over the input distribution. In other words, the guarantee holds for every input.

## Expected Comparisons Analysis

Let $z_1 < z_2 < \cdots < z_n$ be the elements in sorted order. Define the indicator variable $X_{ij} = 1$ if $z_i$ and $z_j$ are ever compared during the sort. Since each comparison happens at most once (when one of $z_i, z_j$ is chosen as the pivot), the total number of comparisons is:

$$
C(n) = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} X_{ij}
$$

The elements $z_i$ and $z_j$ are compared if and only if one of them is chosen as the pivot before any element in $\{z_{i+1}, \ldots, z_{j-1}\}$. Since each of the $j - i + 1$ elements in $\{z_i, z_{i+1}, \ldots, z_j\}$ is equally likely to be the first pivot chosen from this set:

$$
P(X_{ij} = 1) = \frac{2}{j - i + 1}
$$

Taking expectations and summing:

$$
E[C(n)] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j - i + 1} = 2n H_n - O(n) = 2n \ln n + O(n)
$$

where $H_n = \sum_{k=1}^{n} 1/k$ is the $n$-th harmonic number.

## Complexity

| Case | Time | Space |
|------|------|-------|
| Best | $O(n \log n)$ | $O(\log n)$ |
| Expected (any input) | $O(n \log n)$ | $O(\log n)$ expected |
| Worst | $O(n^2)$ | $O(n)$ |

The worst case is $O(n^2)$ but occurs with probability at most $O(1/n!)$, making it negligible in practice.

## Implementation

```python
"""
Randomized quicksort with in-place Lomuto partition.

By choosing the pivot uniformly at random, the expected running time
is O(n log n) for every input, eliminating adversarial worst cases.
"""

import random


# === Randomized Partition ===

def randomized_partition(arr: list, lo: int, hi: int) -> int:
    """Partition arr[lo..hi] around a randomly chosen pivot.

    Returns the final index of the pivot element.
    Uses Lomuto's partitioning scheme.
    """
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


# === Randomized Quicksort ===

def randomized_quicksort(arr: list, lo: int, hi: int) -> None:
    """Sort arr[lo..hi] in place using randomized quicksort."""
    if lo < hi:
        pivot_pos = randomized_partition(arr, lo, hi)
        randomized_quicksort(arr, lo, pivot_pos - 1)
        randomized_quicksort(arr, pivot_pos + 1, hi)


# === Demonstration ===

if __name__ == "__main__":
    random.seed(42)

    data = [3, 6, 8, 10, 1, 2, 1]
    print(f"Before: {data}")
    randomized_quicksort(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # Sorted input — no longer adversarial
    sorted_input = list(range(1, 16))
    print(f"Sorted input: {sorted_input}")
    randomized_quicksort(sorted_input, 0, len(sorted_input) - 1)
    print(f"After:        {sorted_input}")
    print()

    # Reverse-sorted input
    reverse_input = list(range(15, 0, -1))
    print(f"Reverse input: {reverse_input}")
    randomized_quicksort(reverse_input, 0, len(reverse_input) - 1)
    print(f"After:         {reverse_input}")
```

**Output:**
```
Before: [3, 6, 8, 10, 1, 2, 1]
After:  [1, 1, 2, 3, 6, 8, 10]

Sorted input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
After:        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Reverse input: [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
After:         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

!!! warning "Randomized vs Deterministic Guarantees"
    Randomized quicksort guarantees $O(n \log n)$ **expected** time for every input, but the worst case is still $O(n^2)$. If a strict $O(n \log n)$ worst-case bound is required, use introsort, which falls back to heapsort when the recursion depth exceeds a threshold.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 7. MIT Press.
