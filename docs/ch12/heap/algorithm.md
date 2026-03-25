# Heapsort Algorithm

Sorting an array by comparison requires at least $O(n \log n)$ time in the worst case.  Heapsort achieves this bound exactly by exploiting the **max-heap** property: the largest element always sits at the root, so repeatedly extracting it yields a sorted sequence.  Unlike merge sort, heapsort sorts in place with only $O(1)$ auxiliary space, making it attractive when memory is tight.

## Core Idea

Heapsort works in two phases:

1. **Build a max-heap** from the unsorted array using the bottom-up `heapify` procedure in $O(n)$ time.
2. **Repeatedly extract the maximum**: swap the root (largest element) with the last unsorted element, shrink the heap by one, and restore the heap property with `sift_down` in $O(\log n)$ time.

After $n - 1$ extractions the array is sorted in ascending order.

## Key Operations

The algorithm relies on three array-based heap operations.  For an array of length $n$ with 0-based indexing:

- **Parent**: $\text{parent}(i) = \lfloor (i - 1) / 2 \rfloor$
- **Left child**: $\text{left}(i) = 2i + 1$
- **Right child**: $\text{right}(i) = 2i + 2$

### Sift Down

`sift_down` restores the max-heap property rooted at index $i$ by comparing the node with its children and swapping with the larger child if necessary, then recursing on the affected subtree.

### Build Max-Heap

Starting from the last internal node $\lfloor n/2 \rfloor - 1$ down to index 0, call `sift_down` on each node.  Because most nodes are near the leaves and require little work, the total cost is $O(n)$ rather than $O(n \log n)$.

## Complexity Summary

$$
\begin{array}{lcl}
\textbf{Phase} & & \textbf{Time} \\
\hline
\text{Build max-heap (bottom-up heapify)} & & O(n) \\
\text{Extract max } \times\, (n-1) & & O(n \log n) \\
\hline
\text{Heapsort total} & & O(n \log n)
\end{array}
$$

Space complexity is $O(1)$ auxiliary because the heap is built inside the input array.

## Pseudocode

```
HEAPSORT(A, n):
    BUILD-MAX-HEAP(A, n)
    for i = n - 1 down to 1:
        swap A[0] and A[i]
        SIFT-DOWN(A, 0, i)     // heap size shrinks to i

SIFT-DOWN(A, i, heap_size):
    largest = i
    l = 2i + 1
    r = 2i + 2
    if l < heap_size and A[l] > A[largest]:
        largest = l
    if r < heap_size and A[r] > A[largest]:
        largest = r
    if largest != i:
        swap A[i] and A[largest]
        SIFT-DOWN(A, largest, heap_size)
```

## Step-by-Step Example

Consider the array $[4, 10, 3, 5, 1]$.

**Phase 1 -- Build max-heap:**

Starting from the last internal node (index 1), apply `sift_down`:

- Index 1: node 10 is already larger than its child 1 -- no swap.
- Index 0: node 4 is smaller than its child 10 -- swap 4 and 10, then sift 4 down past 5.

Result: $[10, 5, 3, 4, 1]$.

**Phase 2 -- Repeated extraction:**

| Step | Swap root with | Heap after sift-down | Sorted tail |
|------|----------------|----------------------|-------------|
| 1    | index 4        | $[5, 4, 3, 1]$      | $10$        |
| 2    | index 3        | $[4, 1, 3]$         | $5, 10$     |
| 3    | index 2        | $[3, 1]$            | $4, 5, 10$  |
| 4    | index 1        | $[1]$               | $3, 4, 5, 10$ |

Final sorted array: $[1, 3, 4, 5, 10]$.

## Python Implementation

```python
"""
Heapsort algorithm.

Demonstrates the two-phase heapsort: build a max-heap in O(n), then
repeatedly extract the maximum to produce a sorted array in O(n log n)
total time with O(1) auxiliary space.
"""


# === Sift-down procedure =====================================================

def sift_down(arr: list, i: int, heap_size: int) -> None:
    """Restore the max-heap property rooted at index *i*.

    Assumes both subtrees of *i* are valid max-heaps.
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < heap_size and arr[left] > arr[largest]:
        largest = left
    if right < heap_size and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        sift_down(arr, largest, heap_size)


# === Build max-heap ===========================================================

def build_max_heap(arr: list) -> None:
    """Build a max-heap in place in O(n) time."""
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, i, n)


# === Heapsort =================================================================

def heapsort(arr: list) -> None:
    """Sort *arr* in ascending order using heapsort (in place)."""
    n = len(arr)
    build_max_heap(arr)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        sift_down(arr, 0, i)


# === Main =====================================================================

if __name__ == "__main__":
    data = [4, 10, 3, 5, 1]
    print(f"Before: {data}")
    heapsort(data)
    print(f"After:  {data}")

    data2 = [38, 27, 43, 3, 9, 82, 10]
    heapsort(data2)
    print(f"Sorted: {data2}")
```

**Output:**
```
Before: [4, 10, 3, 5, 1]
After:  [1, 3, 4, 5, 10]
Sorted: [3, 9, 10, 27, 38, 43, 82]
```

## Why Heapsort Matters

Heapsort guarantees $O(n \log n)$ time in the **worst case** -- a property that quicksort lacks without randomization or introspection.  Combined with $O(1)$ extra space, heapsort is the go-to algorithm when both time and space guarantees are required.  Its main practical drawback is poor cache locality compared to quicksort and merge sort, which leads to higher constant factors on modern hardware.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Chapter 6.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley, Section 2.4.
