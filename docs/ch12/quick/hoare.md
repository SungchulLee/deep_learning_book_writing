# Hoare Partition Scheme

C. A. R. Hoare invented quicksort in 1960 and proposed the original partition scheme that bears his name.  Unlike Lomuto's scheme, which uses a single pointer scanning left-to-right, Hoare's partition uses **two pointers** starting at opposite ends of the array and moving inward until they cross.  This produces roughly three times fewer swaps on average, making it faster in practice despite being slightly harder to implement correctly.

## Algorithm

Given an array $A[\ell..r]$ with pivot $p = A[\ell]$ (the first element):

1. Initialize $i = \ell - 1$ and $j = r + 1$.
2. Repeat:
    - Increment $i$ until $A[i] \geq p$.
    - Decrement $j$ until $A[j] \leq p$.
    - If $i < j$, swap $A[i]$ and $A[j]$.
    - If $i \geq j$, return $j$.

The returned index $j$ is not necessarily the pivot's final position.  Instead, it guarantees that $A[\ell..j] \leq p$ and $A[j+1..r] \geq p$.  This is a weaker postcondition than Lomuto's, which matters for the recursive calls.

!!! warning "Recursive call structure"
    Because the pivot may not be at index $j$, quicksort with Hoare partition must recurse on $A[\ell..j]$ and $A[j+1..r]$ (not $A[\ell..j-1]$ and $A[j+1..r]$).  Using Lomuto-style recursion with Hoare partition causes infinite loops.

## Pseudocode

```
HOARE-PARTITION(A, left, right):
    pivot = A[left]
    i = left - 1
    j = right + 1
    while true:
        repeat i = i + 1 until A[i] >= pivot
        repeat j = j - 1 until A[j] <= pivot
        if i >= j:
            return j
        swap A[i] and A[j]
```

## Step-by-Step Example

Partition $A = [7, 2, 1, 6, 8, 5, 3, 4]$ with pivot $p = A[0] = 7$:

| Step | $i$ scan | $j$ scan | Action | Array |
|------|----------|----------|--------|-------|
| 1    | $i=0$: $A[0]=7 \geq 7$, stop | $j=7$: $A[7]=4 \leq 7$, stop | swap $A[0], A[7]$ | $[4, 2, 1, 6, 8, 5, 3, 7]$ |
| 2    | $i=1$: $2 < 7$; $i=2$: $1 < 7$; $i=3$: $6 < 7$; $i=4$: $8 \geq 7$, stop | $j=6$: $3 \leq 7$, stop | swap $A[4], A[6]$ | $[4, 2, 1, 6, 3, 5, 8, 7]$ |
| 3    | $i=5$: $5 < 7$; $i=6$: $8 \geq 7$, stop | $j=5$: $5 \leq 7$, stop | $i \geq j$, return $j=5$ | $[4, 2, 1, 6, 3, 5, 8, 7]$ |

Result: $j = 5$.  Elements $A[0..5] = [4, 2, 1, 6, 3, 5]$ are all $\leq 7$, and $A[6..7] = [8, 7]$ are all $\geq 7$.

## Why Fewer Swaps

In Lomuto's scheme, every element $\leq$ pivot triggers a swap, even when the element is already on the correct side.  Hoare's scheme only swaps when both pointers have found an out-of-place element -- an element $\geq$ pivot on the left and an element $\leq$ pivot on the right.

On random data with distinct elements, the expected number of swaps per partition is:

$$
\mathbb{E}[\text{swaps}]_{\text{Hoare}} \approx \frac{n}{6}, \quad \mathbb{E}[\text{swaps}]_{\text{Lomuto}} \approx \frac{n}{2}
$$

This 3:1 ratio explains Hoare's consistent practical advantage.

## Complexity Analysis

**Comparisons per partition call:** at most $n + 1$ (each pointer scans at most $n$ positions total, plus one comparison to check crossing).

**Swaps per partition call:** at most $n/2$ (each swap fixes two elements).

**Overall quicksort complexity** is the same as with Lomuto partition: $O(n \log n)$ average, $O(n^2)$ worst case.  The improvement is in the constant factor.

## Correctness Argument

The key invariant maintained by Hoare partition:

- When $i$ stops, $A[i] \geq p$.
- When $j$ stops, $A[j] \leq p$.
- After swapping, $A[i] \leq p$ and $A[j] \geq p$.
- When $i \geq j$, every element in $A[\ell..j]$ has been "approved" by the $j$ pointer (i.e., $\leq p$), and every element in $A[j+1..r]$ has been "approved" by the $i$ pointer (i.e., $\geq p$).

This ensures a valid partition even though the pivot element may end up anywhere in $A[\ell..j]$.

## Python Implementation

```python
"""
Hoare partition scheme for quicksort.

Uses two pointers scanning inward from opposite ends of the array,
producing roughly 3x fewer swaps than Lomuto on random data.
"""


# === Hoare partition ==========================================================

def hoare_partition(arr: list, left: int, right: int) -> int:
    """Partition arr[left..right] using the first element as pivot.

    Returns index j such that:
    - arr[left..j] <= pivot
    - arr[j+1..right] >= pivot
    """
    pivot = arr[left]
    i = left - 1
    j = right + 1

    while True:
        i += 1
        while arr[i] < pivot:
            i += 1

        j -= 1
        while arr[j] > pivot:
            j -= 1

        if i >= j:
            return j

        arr[i], arr[j] = arr[j], arr[i]


# === Quicksort with Hoare partition ===========================================

def quicksort_hoare(arr: list, left: int = 0, right: int = None) -> None:
    """Sort arr[left..right] in place using Hoare partition.

    Note: recurse on [left..j] and [j+1..right], NOT [left..j-1].
    """
    if right is None:
        right = len(arr) - 1
    if left < right:
        j = hoare_partition(arr, left, right)
        quicksort_hoare(arr, left, j)        # includes j
        quicksort_hoare(arr, j + 1, right)


# === Main =====================================================================

if __name__ == "__main__":
    # Partition demonstration
    data = [7, 2, 1, 6, 8, 5, 3, 4]
    print(f"Before: {data}")
    j = hoare_partition(data, 0, len(data) - 1)
    print(f"After:  {data}  (partition index j={j})")
    print(f"Left:   {data[:j+1]}")
    print(f"Right:  {data[j+1:]}")
    print()

    # Full sort
    data2 = [10, 80, 30, 90, 40, 50, 70]
    print(f"Before: {data2}")
    quicksort_hoare(data2)
    print(f"After:  {data2}")
```

**Output:**
```
Before: [7, 2, 1, 6, 8, 5, 3, 4]
After:  [4, 2, 1, 6, 3, 5, 8, 7]  (partition index j=5)
Left:   [4, 2, 1, 6, 3, 5]
Right:  [8, 7]

Before: [10, 80, 30, 90, 40, 50, 70]
After:  [10, 30, 40, 50, 70, 80, 90]
```

## References

- Hoare, C. A. R. (1962). Quicksort. *The Computer Journal*, 5(1), 10-16.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Problem 7-1.
- Sedgewick, R. (1978). Implementing Quicksort programs. *Communications of the ACM*, 21(10), 847-857.
