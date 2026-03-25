# Sorted Array

One straightforward way to implement the [priority queue ADT](adt.md) is to keep the elements in a sorted array.  Because the array is always sorted by key, the minimum (or maximum) element sits at a known position, making extraction trivial.  The price is paid at insertion time: each new element must be placed in its correct sorted position, which requires shifting existing elements.  This page describes the sorted-array priority queue, analyzes its complexity, and contrasts it with the [unsorted-array](unsorted.md) and [heap-based](heap_preview.md) alternatives.

## Representation

The priority queue stores its $n$ elements in an array $A[0 \ldots n-1]$ sorted in non-decreasing order of key:

$$
\text{key}(A[0]) \le \text{key}(A[1]) \le \cdots \le \text{key}(A[n-1])
$$

The minimum element is always at index $0$, and the maximum is at index $n - 1$.

## Operations

### Insert

To insert a new element $x$ with key $k$:

1. Find the correct position $j$ using **binary search** in $O(\log n)$ time.
2. Shift elements $A[j], A[j+1], \dots, A[n-1]$ one position to the right to make room.
3. Place $x$ at index $j$.
4. Increment $n$.

The binary search costs $O(\log n)$, but the shifting costs $O(n)$ in the worst case (when $x$ must be inserted at the beginning).  Therefore:

$$
T_{\text{insert}} = O(n)
$$

### Extract-Min

The minimum is at index $0$.  Remove it by shifting all elements one position to the left:

$$
T_{\text{extract-min}} = O(n)
$$

Alternatively, store elements in non-increasing order so the minimum is at the end.  Then extraction is $O(1)$ (just decrement $n$), but inserting at the correct position still requires shifting.

!!! tip "Optimization: reverse the sort order"
    If we store elements in **non-increasing** order — largest at index 0, smallest at index $n-1$ — then `extract_min()` simply returns and removes the last element in $O(1)$.  Insertion still costs $O(n)$ due to shifting, but extraction becomes constant time.  This is the more practical variant.

### Find-Min

With either sort order, the minimum is at a known position (index $0$ or index $n-1$):

$$
T_{\text{find-min}} = O(1)
$$

## Complexity Summary

Using the non-increasing order optimization:

| Operation | Time |
|---|---|
| `insert(x)` | $O(n)$ |
| `extract_min()` | $O(1)$ |
| `find_min()` | $O(1)$ |
| `is_empty()` | $O(1)$ |

## When to Use a Sorted Array

The sorted-array approach is best when:

- **Extractions dominate insertions**: if the workload performs many more `extract_min` calls than `insert` calls, the $O(1)$ extraction is advantageous.
- **The dataset is small**: for small $n$, the $O(n)$ insertion cost is negligible, and the simplicity of the implementation outweighs the benefit of a heap.
- **The data arrives pre-sorted**: if elements are inserted in sorted order, each insertion requires no shifting and takes $O(1)$.

For workloads with frequent insertions, a [binary heap](heap_preview.md) provides $O(\log n)$ for both insertion and extraction and is generally preferred.

??? example "Operation trace with non-increasing order"
    Elements are stored largest-first.  The minimum is always at the end.

    | Step | Operation | Array (non-increasing) | Returned |
    |------|-----------|----------------------|----------|
    | 1 | `insert(5)` | [5] | — |
    | 2 | `insert(3)` | [5, 3] | — |
    | 3 | `insert(8)` | [8, 5, 3] | — |
    | 4 | `find_min()` | [8, 5, 3] | 3 |
    | 5 | `extract_min()` | [8, 5] | 3 |
    | 6 | `insert(1)` | [8, 5, 1] | — |
    | 7 | `extract_min()` | [8, 5] | 1 |

## Python Implementation

```python
"""Sorted-array priority queue (non-increasing order for O(1) extract-min)."""

import bisect


# === Sorted Array Priority Queue ===

class SortedArrayPQ:
    """A min-priority queue backed by an array sorted in non-increasing order.

    insert:      O(n) due to shifting
    extract_min: O(1) by removing the last element
    find_min:    O(1) by peeking at the last element
    """

    def __init__(self):
        self._data = []  # sorted in non-increasing order

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def insert(self, key):
        """Insert a key in O(n) time, maintaining non-increasing order."""
        # bisect works on non-decreasing order, so we negate keys
        # to reuse it for non-increasing order.
        pos = bisect.bisect_left(self._data, key, key=lambda x: -x)
        self._data.insert(pos, key)

    def find_min(self):
        """Return the minimum key in O(1) time."""
        if self.is_empty():
            raise IndexError("find_min from empty priority queue")
        return self._data[-1]

    def extract_min(self):
        """Remove and return the minimum key in O(1) time."""
        if self.is_empty():
            raise IndexError("extract_min from empty priority queue")
        return self._data.pop()


# === Demo ===

if __name__ == "__main__":
    pq = SortedArrayPQ()
    for val in [5, 3, 8, 1, 9, 2]:
        pq.insert(val)
    print(f"Min: {pq.find_min()}")  # 1
    print(f"Extract: {pq.extract_min()}")  # 1
    print(f"Extract: {pq.extract_min()}")  # 2
    print(f"Remaining size: {len(pq)}")     # 4
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 6. MIT Press.
