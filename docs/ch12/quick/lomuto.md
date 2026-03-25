# Lomuto Partition Scheme

The Lomuto partition scheme is the version of quicksort partitioning presented in most introductory textbooks, including CLRS.  It selects the **last element** as the pivot and maintains a boundary index $i$ that separates elements already known to be $\leq$ pivot from those $>$ pivot.  A single pointer $j$ scans left-to-right, swapping small elements behind the boundary.  While simpler to understand and implement than Hoare's scheme, Lomuto performs roughly three times as many swaps on random data.

## Algorithm

Given an array $A[\ell..r]$ with pivot $p = A[r]$:

1. Initialize boundary $i = \ell$.
2. For each $j$ from $\ell$ to $r - 1$:
    - If $A[j] \leq p$, swap $A[i]$ and $A[j]$, then increment $i$.
3. Swap $A[i]$ and $A[r]$ to place the pivot in its final position.
4. Return $i$.

### Loop Invariant

At the start of each iteration of the `for` loop:

- $A[k] \leq p$ for all $k \in [\ell, i-1]$ (elements already partitioned to the left).
- $A[k] > p$ for all $k \in [i, j-1]$ (elements already partitioned to the right).
- $A[r] = p$ (pivot, untouched until the end).

This invariant ensures correctness: when $j$ reaches $r$, swapping $A[i]$ with $A[r]$ places the pivot at index $i$, with all smaller elements to its left and all larger elements to its right.

## Pseudocode

```
LOMUTO-PARTITION(A, left, right):
    pivot = A[right]
    i = left
    for j = left to right - 1:
        if A[j] <= pivot:
            swap A[i] and A[j]
            i = i + 1
    swap A[i] and A[right]
    return i
```

## Step-by-Step Example

Partition $A = [7, 2, 1, 6, 8, 5, 3, 4]$ with pivot $p = A[7] = 4$:

| $j$ | $A[j]$ | $A[j] \leq 4$? | Action | $i$ | Array |
|-----|---------|-----------------|--------|-----|-------|
| 0   | 7       | No              | skip   | 0   | $[7, 2, 1, 6, 8, 5, 3, 4]$ |
| 1   | 2       | Yes             | swap $A[0], A[1]$ | 1 | $[2, 7, 1, 6, 8, 5, 3, 4]$ |
| 2   | 1       | Yes             | swap $A[1], A[2]$ | 2 | $[2, 1, 7, 6, 8, 5, 3, 4]$ |
| 3   | 6       | No              | skip   | 2   | $[2, 1, 7, 6, 8, 5, 3, 4]$ |
| 4   | 8       | No              | skip   | 2   | $[2, 1, 7, 6, 8, 5, 3, 4]$ |
| 5   | 5       | No              | skip   | 2   | $[2, 1, 7, 6, 8, 5, 3, 4]$ |
| 6   | 3       | Yes             | swap $A[2], A[6]$ | 3 | $[2, 1, 3, 6, 8, 5, 7, 4]$ |

Final swap: $A[3] \leftrightarrow A[7]$, giving $[2, 1, 3, \mathbf{4}, 8, 5, 7, 6]$.  Pivot 4 is at index 3.

## Complexity Analysis

**Comparisons.**  The `for` loop makes exactly $r - \ell$ comparisons (one per element, excluding the pivot):

$$
C = n - 1 \quad \text{per partition call}
$$

**Swaps.**  In the worst case, every element is $\leq$ pivot, causing $n - 1$ swaps (each redundant, as the element is swapped with itself).  On random data, the expected number of swaps is approximately $n/2$ because half the elements are expected to be $\leq$ pivot.

$$
\mathbb{E}[\text{swaps}] \approx \frac{n}{2}
$$

This is roughly three times more swaps than Hoare's partition, which typically performs about $n/6$ swaps on random data.

**Time.** Each partition call is $O(n)$.  The overall quicksort time depends on partition quality (see the analysis page).

!!! warning "Performance on arrays with many duplicates"
    When all elements are equal, Lomuto partition always places the pivot at one end (e.g., index $r$), producing a maximally unbalanced split.  This causes $O(n^2)$ behavior.  The three-way partition (Dutch National Flag) handles duplicates gracefully.

## Python Implementation

```python
"""
Lomuto partition scheme for quicksort.

Selects the last element as pivot and partitions the array using
a single left-to-right scan with a boundary pointer.
"""


# === Lomuto partition =========================================================

def lomuto_partition(arr: list, left: int, right: int) -> int:
    """Partition arr[left..right] using the last element as pivot.

    Returns the final index of the pivot.

    Loop invariant:
    - arr[left..i-1] contains elements <= pivot
    - arr[i..j-1] contains elements > pivot
    - arr[right] is the pivot
    """
    pivot = arr[right]
    i = left
    for j in range(left, right):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[right] = arr[right], arr[i]
    return i


# === Quicksort with Lomuto partition ==========================================

def quicksort_lomuto(arr: list, left: int = 0, right: int = None) -> None:
    """Sort arr[left..right] in place using Lomuto partition."""
    if right is None:
        right = len(arr) - 1
    if left < right:
        pivot_idx = lomuto_partition(arr, left, right)
        quicksort_lomuto(arr, left, pivot_idx - 1)
        quicksort_lomuto(arr, pivot_idx + 1, right)


# === Main =====================================================================

if __name__ == "__main__":
    # Partition demonstration
    data = [7, 2, 1, 6, 8, 5, 3, 4]
    print(f"Before: {data}")
    idx = lomuto_partition(data, 0, len(data) - 1)
    print(f"After:  {data}  (pivot at index {idx})")
    print()

    # Full sort
    data2 = [10, 80, 30, 90, 40, 50, 70]
    print(f"Before: {data2}")
    quicksort_lomuto(data2)
    print(f"After:  {data2}")
```

**Output:**
```
Before: [7, 2, 1, 6, 8, 5, 3, 4]
After:  [2, 1, 3, 4, 8, 5, 7, 6]  (pivot at index 3)

Before: [10, 80, 30, 90, 40, 50, 70]
After:  [10, 30, 40, 50, 70, 80, 90]
```

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Section 7.1.
- Lomuto, N. (attributed). Partition scheme popularized by Bentley in *Programming Pearls*.
- Bentley, J. L. (2000). *Programming Pearls* (2nd ed.). Addison-Wesley, Chapter 11.
