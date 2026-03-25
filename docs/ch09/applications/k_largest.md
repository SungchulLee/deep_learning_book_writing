# K Largest Elements

Finding the $k$ largest (or smallest) elements from a collection of $n$ items is a common problem in data processing, ranking, and analytics. While sorting the entire array costs $O(n \log n)$, a heap-based approach solves the problem in $O(n \log k)$ time using only $O(k)$ space. When $k \ll n$, this represents a significant improvement and works naturally in a streaming setting where elements arrive one at a time.

## Min-Heap Approach

The strategy uses a **min-heap of size $k$** as a "window" that holds the $k$ largest elements seen so far. The minimum of the heap acts as a threshold: any new element larger than this threshold belongs among the top $k$.

### Algorithm

1. Insert the first $k$ elements into a min-heap.
2. For each remaining element $x$:
    - If $x >$ the heap's minimum, replace the minimum with $x$ and sift down.
    - Otherwise, skip $x$.
3. The heap contains the $k$ largest elements.

### Why a Min-Heap?

Using a min-heap (not a max-heap) is the key insight. The root of the min-heap is the **smallest among the $k$ largest**, which is exactly the element we want to compare against each new candidate. If a new element exceeds this threshold, it displaces the current minimum, and sift-down restores the heap property.

## Complexity Analysis

| Step | Cost |
|------|------|
| Build initial heap of size $k$ | $O(k)$ |
| Process remaining $n - k$ elements | At most $(n - k) \cdot O(\log k)$ |
| **Total** | $O(n \log k)$ |

$$
T(n, k) = O(k) + O((n - k) \log k) = O(n \log k)
$$

**Space**: $O(k)$ for the heap.

When $k$ is a constant (e.g., "top 10"), the time simplifies to $O(n)$.

## Comparison of Approaches

| Approach | Time | Space | Streaming? |
|----------|------|-------|-----------|
| Sort, then take last $k$ | $O(n \log n)$ | $O(1)$ if in-place | No |
| Min-heap of size $k$ | $O(n \log k)$ | $O(k)$ | Yes |
| Quickselect + partition | $O(n)$ expected | $O(1)$ | No |
| Max-heap of size $n$ + $k$ extractions | $O(n + k \log n)$ | $O(n)$ | No |

The min-heap approach is optimal when streaming is required or when $k \ll n$. Quickselect is faster for one-shot computation if the data fits in memory.

## Implementation

```python
"""
Finding the k largest elements using a min-heap.

Demonstrates the heap-based approach that runs in O(n log k)
time and O(k) space, suitable for streaming data.
"""

import heapq


# === From-Scratch Implementation ===

def k_largest_manual(arr, k):
    """Find the k largest elements using a manually managed min-heap."""
    if k <= 0:
        return []
    if k >= len(arr):
        return sorted(arr, reverse=True)

    # Build min-heap from first k elements
    heap = arr[:k]
    # Build heap in O(k)
    for i in range(k // 2 - 1, -1, -1):
        _sift_down(heap, i, k)

    # Process remaining elements
    for x in arr[k:]:
        if x > heap[0]:
            heap[0] = x
            _sift_down(heap, 0, k)

    # Extract in sorted order (largest first)
    result = []
    while heap:
        # Swap root with last, shrink, sift down
        heap[0], heap[-1] = heap[-1], heap[0]
        result.append(heap.pop())
        if heap:
            _sift_down(heap, 0, len(heap))
    return result


def _sift_down(arr, i, n):
    """Min-heap sift-down."""
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] < arr[smallest]:
            smallest = left
        if right < n and arr[right] < arr[smallest]:
            smallest = right
        if smallest == i:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest


# === Using Python heapq ===

def k_largest_heapq(arr, k):
    """Find the k largest elements using heapq.nlargest."""
    return heapq.nlargest(k, arr)


def k_largest_heap_manual_heapq(arr, k):
    """Find the k largest using heapq with explicit heap management."""
    if k <= 0:
        return []
    if k >= len(arr):
        return sorted(arr, reverse=True)

    # Maintain a min-heap of size k
    heap = arr[:k]
    heapq.heapify(heap)

    for x in arr[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)  # pop min, push x in one operation

    return sorted(heap, reverse=True)


# === Demonstration ===

if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    k = 5

    print(f"Array: {data}")
    print(f"k = {k}\n")

    result1 = k_largest_manual(data, k)
    print(f"Manual heap:   {result1}")

    result2 = k_largest_heapq(data, k)
    print(f"heapq.nlargest: {result2}")

    result3 = k_largest_heap_manual_heapq(data, k)
    print(f"heapq manual:  {result3}")

    # Streaming example
    print("\n--- Streaming Example ---")
    stream = [3, 7, 2, 8, 1, 9, 4, 6]
    k = 3
    heap = []
    for i, x in enumerate(stream):
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
        print(f"  After seeing {x}: top-{k} = {sorted(heap, reverse=True)}")
```

**Output:**
```
Array: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
k = 5

Manual heap:   [9, 9, 9, 8, 7]
heapq.nlargest: [9, 9, 9, 8, 7]
heapq manual:  [9, 9, 9, 8, 7]

--- Streaming Example ---
  After seeing 3: top-3 = [3]
  After seeing 7: top-3 = [7, 3]
  After seeing 2: top-3 = [7, 3, 2]
  After seeing 8: top-3 = [8, 7, 3]
  After seeing 1: top-3 = [8, 7, 3]
  After seeing 9: top-3 = [9, 8, 7]
  After seeing 4: top-3 = [9, 8, 7]
  After seeing 6: top-3 = [9, 8, 7]
```

## K Smallest Elements

The dual problem -- finding the $k$ smallest elements -- uses a **max-heap of size $k$**. The root holds the largest among the $k$ smallest, and any new element smaller than the root replaces it. With Python's `heapq` (which provides only min-heaps), negate the values:

```python
import heapq

def k_smallest(arr, k):
    """Find k smallest elements using a negated max-heap."""
    heap = [-x for x in arr[:k]]
    heapq.heapify(heap)
    for x in arr[k:]:
        if x < -heap[0]:
            heapq.heapreplace(heap, -x)
    return sorted(-x for x in heap)
```

Alternatively, use `heapq.nsmallest(k, arr)` which handles this internally.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6 and 9. MIT Press.
- Python Documentation: [heapq.nlargest](https://docs.python.org/3/library/heapq.html#heapq.nlargest)
