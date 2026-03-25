# The Partition Procedure

Quicksort's performance depends entirely on a single subroutine: **partition**.  Given a **pivot** element, partition rearranges the array so that all elements smaller than the pivot appear before it and all elements larger appear after it.  After partitioning, the pivot is in its final sorted position, and the algorithm recurses on the two sides independently.  The efficiency of quicksort -- its $O(n \log n)$ average case and $O(n^2)$ worst case -- is determined by how well the partition divides the array.

## The Partition Contract

Given an array $A[\ell..r]$ and a pivot value $p$, the partition procedure returns an index $q$ such that:

- $A[i] \leq p$ for all $i \in [\ell, q-1]$
- $A[q] = p$
- $A[j] > p$ for all $j \in [q+1, r]$

After partition, $A[q]$ is in its correct sorted position and never moves again.  The two subarrays $A[\ell..q-1]$ and $A[q+1..r]$ can then be sorted independently.

## How Quicksort Uses Partition

```
QUICKSORT(A, left, right):
    if left < right:
        q = PARTITION(A, left, right)
        QUICKSORT(A, left, q - 1)
        QUICKSORT(A, q + 1, right)
```

The recursion has no combine step -- once both sides are sorted, the entire array is sorted because the pivot is already in place.  This makes quicksort a **conquer-then-divide** algorithm rather than the traditional divide-and-conquer pattern of merge sort.

## Partition Variants

Several partition schemes exist, each with different trade-offs:

| Scheme      | Pivot position | Pointer movement | Swaps | Stable |
|-------------|---------------|------------------|-------|--------|
| Lomuto      | Last element  | One pointer left-to-right | More | No |
| Hoare       | First element | Two pointers inward | Fewer | No |
| Three-way   | Any           | Three regions | Handles duplicates | No |
| Dual-pivot  | Two pivots    | Three pointers | Three regions | No |

Each variant is covered in detail on its own page.

## Partition Quality and Quicksort Performance

The quality of a partition is measured by how evenly it splits the array.  Let $q$ be the partition index for an array of size $n$:

**Balanced partition** ($q \approx n/2$): both subarrays have roughly $n/2$ elements, giving $O(\log n)$ recursion depth and $O(n \log n)$ total time.

**Unbalanced partition** ($q = 0$ or $q = n-1$): one subarray is empty and the other has $n-1$ elements, giving $O(n)$ recursion depth and $O(n^2)$ total time.

Even a constant-fraction split (e.g., 10%/90%) still gives $O(n \log n)$ time, because the recursion tree has $O(\log n)$ levels:

$$
T(n) = T(n/10) + T(9n/10) + O(n) = O(n \log n)
$$

The worst case occurs only when **every** partition produces the maximally unbalanced split.

!!! warning "Worst-case triggers"
    A naive pivot choice (always first or last element) hits $O(n^2)$ on already-sorted or reverse-sorted input.  Randomized pivot selection or median-of-three avoids this in practice.

## Step-by-Step Partition Example

Partition $A = [7, 2, 1, 6, 8, 5, 3, 4]$ using pivot $p = A[7] = 4$ (Lomuto-style, last element):

| Step | Compare | Action                  | Array state              | $i$ |
|------|---------|-------------------------|--------------------------|-----|
| 1    | $7 > 4$ | skip                    | $[7, 2, 1, 6, 8, 5, 3, 4]$ | 0 |
| 2    | $2 \leq 4$ | swap $A[0]$ and $A[1]$ | $[2, 7, 1, 6, 8, 5, 3, 4]$ | 1 |
| 3    | $1 \leq 4$ | swap $A[1]$ and $A[2]$ | $[2, 1, 7, 6, 8, 5, 3, 4]$ | 2 |
| 4    | $6 > 4$ | skip                    | $[2, 1, 7, 6, 8, 5, 3, 4]$ | 2 |
| 5    | $8 > 4$ | skip                    | $[2, 1, 7, 6, 8, 5, 3, 4]$ | 2 |
| 6    | $5 > 4$ | skip                    | $[2, 1, 7, 6, 8, 5, 3, 4]$ | 2 |
| 7    | $3 \leq 4$ | swap $A[2]$ and $A[6]$ | $[2, 1, 3, 6, 8, 5, 7, 4]$ | 3 |
| Final| --      | swap pivot into position | $[2, 1, 3, \mathbf{4}, 8, 5, 7, 6]$ | 3 |

The pivot 4 is now at index 3, with all smaller elements to its left and all larger elements to its right.

## Python Implementation

```python
"""
The partition procedure for quicksort.

Demonstrates how partition rearranges an array around a pivot element,
placing the pivot in its final sorted position.  Includes both
Lomuto-style and a simple list-comprehension variant for clarity.
"""


# === Lomuto partition =========================================================

def lomuto_partition(arr: list, left: int, right: int) -> int:
    """Partition arr[left..right] using the last element as pivot.

    Returns the final index of the pivot element.

    All elements at indices < pivot_index are <= pivot.
    All elements at indices > pivot_index are > pivot.
    """
    pivot = arr[right]
    i = left
    for j in range(left, right):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[right] = arr[right], arr[i]
    return i


# === Quicksort using partition ================================================

def quicksort(arr: list, left: int = 0, right: int = None) -> None:
    """Sort arr[left..right] in place using quicksort with Lomuto partition."""
    if right is None:
        right = len(arr) - 1
    if left < right:
        pivot_idx = lomuto_partition(arr, left, right)
        quicksort(arr, left, pivot_idx - 1)
        quicksort(arr, pivot_idx + 1, right)


# === Main =====================================================================

if __name__ == "__main__":
    # Demonstrate partition
    data = [7, 2, 1, 6, 8, 5, 3, 4]
    print(f"Before partition: {data}")
    pivot_pos = lomuto_partition(data, 0, len(data) - 1)
    print(f"After partition:  {data}")
    print(f"Pivot index: {pivot_pos}, pivot value: {data[pivot_pos]}")
    print()

    # Full quicksort
    data2 = [38, 27, 43, 3, 9, 82, 10]
    print(f"Before sort: {data2}")
    quicksort(data2)
    print(f"After sort:  {data2}")
```

**Output:**
```
Before partition: [7, 2, 1, 6, 8, 5, 3, 4]
After partition:  [2, 1, 3, 4, 8, 5, 7, 6]
Pivot index: 3, pivot value: 4

Before sort: [38, 27, 43, 3, 9, 82, 10]
After sort:  [3, 9, 10, 27, 38, 43, 82]
```

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Section 7.1.
- Hoare, C. A. R. (1962). Quicksort. *The Computer Journal*, 5(1), 10-16.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley, Section 2.3.
