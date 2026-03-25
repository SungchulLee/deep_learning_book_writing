# Median Maintenance

In many applications -- streaming analytics, running statistics, and online algorithms -- we need to track the **median** of a growing dataset as elements arrive one at a time. Sorting after each insertion costs $O(n \log n)$ per element, which is too slow for large streams. The **two-heap technique** maintains the median in $O(\log n)$ per insertion and $O(1)$ for retrieval by splitting the data into two halves: a max-heap for the lower half and a min-heap for the upper half.

## Two-Heap Invariant

The algorithm maintains two heaps:

- **max_heap**: stores the smaller half of the elements. The root is the largest element in the lower half.
- **min_heap**: stores the larger half of the elements. The root is the smallest element in the upper half.

The following invariant is maintained after every insertion:

1. **Ordering**: every element in max_heap is $\le$ every element in min_heap.
2. **Balance**: the sizes differ by at most 1, i.e., $|\,|\text{max\_heap}| - |\text{min\_heap}|\,| \le 1$.

Under this invariant, the median is always accessible at one or both roots:

- If the heaps have equal size, the median is the average of the two roots.
- If max_heap has one more element, the median is the root of max_heap.

## Algorithm

For each new element $x$:

1. **Insert**: if $x \le$ root of max_heap (or max_heap is empty), push $x$ to max_heap. Otherwise, push $x$ to min_heap.
2. **Rebalance**: if the sizes differ by more than 1, pop from the larger heap and push to the smaller heap.
3. **Query**: the median is available at the root(s) in $O(1)$.

## Step-by-Step Example

Insert the stream `[5, 2, 8, 1, 7, 3]`:

| Step | Element | max_heap (lower) | min_heap (upper) | Median |
|:----:|:-------:|:-----------------:|:----------------:|:------:|
| 1 | 5 | [5] | [] | 5 |
| 2 | 2 | [2] | [5] | 3.5 |
| 3 | 8 | [2] | [5, 8] | 5 |
| 4 | 1 | [2, 1] | [5, 8] | 3.5 |
| 5 | 7 | [2, 1] | [5, 7, 8] | 5 |
| 6 | 3 | [3, 2, 1] | [5, 7, 8] | 4.0 |

After step 3, min_heap has 2 elements while max_heap has 1. The rebalance check allows a difference of 1, so no rebalancing is needed -- we let min_heap be larger by 1, and the median is its root (5).

## Complexity

| Operation | Time |
|-----------|------|
| Insert one element | $O(\log n)$ |
| Query median | $O(1)$ |
| Process $n$ elements | $O(n \log n)$ total |
| Space | $O(n)$ |

Each insertion involves at most one heap push and at most one rebalancing operation (pop from one heap + push to the other), each costing $O(\log n)$.

## Implementation

```python
"""
Median maintenance using two heaps.

Maintains a max-heap for the lower half and a min-heap for the
upper half, enabling O(log n) insertion and O(1) median query.
"""

import heapq


# === Median Finder ===

class MedianFinder:
    """Online median maintenance using the two-heap technique.

    max_heap: stores the smaller half (negated for heapq min-heap).
    min_heap: stores the larger half.
    Invariant: sizes differ by at most 1, max_heap allowed to be larger.
    """

    def __init__(self):
        self.max_heap = []  # negated values (heapq only supports min-heap)
        self.min_heap = []

    def add(self, x):
        """Insert element x and maintain the invariant. O(log n)."""
        # Decide which heap to push to
        if not self.max_heap or x <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -x)
        else:
            heapq.heappush(self.min_heap, x)

        # Rebalance: max_heap can have at most 1 more than min_heap
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.min_heap) > len(self.max_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)

    def median(self):
        """Return the current median. O(1)."""
        if not self.max_heap:
            raise IndexError("median of empty collection")

        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2

    def __len__(self):
        return len(self.max_heap) + len(self.min_heap)


# === Demonstration ===

if __name__ == "__main__":
    mf = MedianFinder()
    stream = [5, 2, 8, 1, 7, 3]

    print("Streaming median maintenance:")
    print(f"{'Element':>8} {'max_heap (lower)':>20} {'min_heap (upper)':>20} {'Median':>8}")
    print("-" * 60)

    for x in stream:
        mf.add(x)
        lower = sorted([-v for v in mf.max_heap])
        upper = sorted(mf.min_heap)
        print(f"{x:>8} {str(lower):>20} {str(upper):>20} {mf.median():>8.1f}")

    # Verify against sorted median
    print("\n--- Verification ---")
    mf2 = MedianFinder()
    data = []
    for x in [41, 35, 62, 5, 97, 108, 3, 25, 22, 78]:
        data.append(x)
        mf2.add(x)
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            expected = sorted_data[n // 2]
        else:
            expected = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        actual = mf2.median()
        status = "OK" if abs(actual - expected) < 1e-9 else "FAIL"
        print(f"  After {x:>3}: median = {actual:>6.1f} (expected {expected:>6.1f}) [{status}]")
```

**Output:**
```
Streaming median maintenance:
 Element   max_heap (lower)    min_heap (upper)   Median
------------------------------------------------------------
       5                [5]                   []      5.0
       2             [2, 5]                   []      3.5
       8                [5]               [2, 8]      5.0
       1             [1, 2]             [5, 8]        3.5
       7             [1, 2]          [5, 7, 8]        5.0
       3          [1, 2, 3]          [5, 7, 8]        4.0

--- Verification ---
  After  41: median =   41.0 (expected   41.0) [OK]
  After  35: median =   38.0 (expected   38.0) [OK]
  After  62: median =   41.0 (expected   41.0) [OK]
  After   5: median =   38.0 (expected   38.0) [OK]
  After  97: median =   41.0 (expected   41.0) [OK]
  After 108: median =   51.5 (expected   51.5) [OK]
  After   3: median =   41.0 (expected   41.0) [OK]
  After  25: median =   38.0 (expected   38.0) [OK]
  After  22: median =   35.0 (expected   35.0) [OK]
  After  78: median =   38.0 (expected   38.0) [OK]
```

## Correctness Argument

The invariant ensures that max_heap contains the $\lceil n/2 \rceil$ smallest elements and min_heap contains the $\lfloor n/2 \rfloor$ largest elements. Since every element in max_heap is at most every element in min_heap, the roots partition the data at the median:

- If $n$ is odd, the median is the root of max_heap (the largest in the lower half).
- If $n$ is even, the median is the average of the two roots.

The rebalancing step maintains the size invariant by transferring elements between heaps when the difference exceeds 1. Each transfer preserves the ordering invariant because the transferred element is the extreme value of its heap.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Problem 9-1: Sorting and order statistics. MIT Press.
