# Unsorted Array

The simplest way to implement the [priority queue ADT](adt.md) is to store elements in an unsorted array.  Insertion is trivial — just append to the end — but finding and removing the minimum requires scanning the entire array.  This trade-off makes the unsorted-array implementation attractive when insertions vastly outnumber extractions, and it serves as a useful baseline against which to compare more sophisticated implementations like [sorted arrays](sorted.md) and [heaps](heap_preview.md).

## Representation

The priority queue stores its $n$ elements in an array $A[0 \ldots n-1]$ with **no ordering constraint**.  Elements are placed in whatever order they happen to be inserted.

## Operations

### Insert

To insert a new element $x$, append it to the end of the array:

$$
T_{\text{insert}} = O(1)
$$

No comparisons or shifts are needed.  With a dynamic array, occasional resizing costs $O(n)$, but the amortized cost remains $O(1)$.

### Extract-Min

To remove and return the element with the smallest key:

1. Scan the entire array to find the index $j$ of the minimum element.
2. Swap $A[j]$ with the last element $A[n-1]$.
3. Remove the last element (decrement $n$).

The scan costs $O(n)$:

$$
T_{\text{extract-min}} = O(n)
$$

Swapping with the last element avoids shifting and keeps the removal itself at $O(1)$.

### Find-Min

Without extracting, finding the minimum still requires a full scan:

$$
T_{\text{find-min}} = O(n)
$$

!!! tip "Caching the minimum"
    An optimization is to maintain a cached pointer to the current minimum.  This reduces `find_min` to $O(1)$, but `extract_min` must still scan to find the new minimum after removal, and `insert` must compare the new element against the cached minimum.  The worst-case asymptotic complexity does not improve.

## Complexity Summary

| Operation | Time |
|---|---|
| `insert(x)` | $O(1)$ amortized |
| `extract_min()` | $O(n)$ |
| `find_min()` | $O(n)$ |
| `is_empty()` | $O(1)$ |

## Comparison with Sorted Array

The unsorted and [sorted](sorted.md) array implementations offer complementary trade-offs:

| Operation | Unsorted | Sorted (non-increasing) |
|---|---|---|
| `insert` | $O(1)$ | $O(n)$ |
| `extract_min` | $O(n)$ | $O(1)$ |
| `find_min` | $O(n)$ | $O(1)$ |

Neither achieves $O(\log n)$ for both operations simultaneously — that requires a [heap](heap_preview.md).

## When to Use an Unsorted Array

- **Batch insert, then extract all**: insert $n$ elements in $O(n)$, then extract all in sorted order.  The total cost is $O(n) + O(n) + O(n-1) + \cdots + O(1) = O(n^2)$, which is equivalent to selection sort.
- **Very small datasets**: when $n$ is tiny, the simplicity of the unsorted array can outperform the constant-factor overhead of a heap.
- **Insert-heavy workloads**: if the ratio of inserts to extractions is high, the $O(1)$ insertion cost dominates the overall performance.

??? example "Operation trace"
    | Step | Operation | Array | Returned |
    |------|-----------|-------|----------|
    | 1 | `insert(5)` | [5] | — |
    | 2 | `insert(3)` | [5, 3] | — |
    | 3 | `insert(8)` | [5, 3, 8] | — |
    | 4 | `insert(1)` | [5, 3, 8, 1] | — |
    | 5 | `find_min()` | [5, 3, 8, 1] | 1 |
    | 6 | `extract_min()` | [5, 3, 8] | 1 (swap 1 with 8, remove last) |
    | 7 | `extract_min()` | [5, 8] | 3 (swap 3 with 8, remove last) |

    At step 6, the minimum (1 at index 3) is swapped with the last element (8), then the last position is removed.  The array becomes [5, 3, 8].  Note the ordering has changed — this is expected since the array is unsorted.

## Python Implementation

```python
"""Unsorted-array priority queue with O(1) insert and O(n) extract-min."""


# === Unsorted Array Priority Queue ===

class UnsortedArrayPQ:
    """A min-priority queue backed by an unsorted dynamic array.

    insert:      O(1) amortized (append)
    extract_min: O(n) (linear scan)
    find_min:    O(n) (linear scan)
    """

    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def insert(self, key):
        """Insert a key in O(1) amortized time."""
        self._data.append(key)

    def find_min(self):
        """Return the minimum key in O(n) time."""
        if self.is_empty():
            raise IndexError("find_min from empty priority queue")
        return min(self._data)

    def extract_min(self):
        """Remove and return the minimum key in O(n) time."""
        if self.is_empty():
            raise IndexError("extract_min from empty priority queue")
        min_idx = 0
        for i in range(1, len(self._data)):
            if self._data[i] < self._data[min_idx]:
                min_idx = i
        # Swap with last element and pop
        self._data[min_idx], self._data[-1] = self._data[-1], self._data[min_idx]
        return self._data.pop()


# === Demo ===

if __name__ == "__main__":
    pq = UnsortedArrayPQ()
    for val in [5, 3, 8, 1, 9, 2]:
        pq.insert(val)
    print(f"Min: {pq.find_min()}")  # 1
    print(f"Extract: {pq.extract_min()}")  # 1
    print(f"Extract: {pq.extract_min()}")  # 2
    print(f"Remaining size: {len(pq)}")     # 4
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 6. MIT Press.
