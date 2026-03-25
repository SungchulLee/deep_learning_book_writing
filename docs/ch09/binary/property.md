# Heap Property

Every efficient use of a heap depends on a single structural invariant: the **heap property**. This ordering constraint, applied recursively at every node, enables the root to always hold the extreme (minimum or maximum) element. Unlike a fully sorted array where every pair of elements is ordered, a heap only enforces a parent-child relationship, trading global order for the ability to insert and delete in logarithmic time.

## Complete Binary Tree

A binary heap is built on top of a **complete binary tree** -- a binary tree in which every level is fully filled except possibly the last, which is filled from left to right. This shape constraint guarantees that a heap with $n$ nodes has height

$$
h = \lfloor \log_2 n \rfloor
$$

and ensures efficient array-based storage without any wasted space.

## Max-Heap Property

A binary tree satisfies the **max-heap property** if, for every node $i$ other than the root, the value of $i$ is at most the value of its parent:

$$
A[\text{parent}(i)] \ge A[i]
$$

This means the largest element in any subtree is always at the subtree's root. By induction, the largest element in the entire heap resides at the root.

!!! example "Max-Heap Example"
    Consider the array `[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]` stored as a max-heap:

    ```
              16
            /    \
          14      10
         /  \    /  \
        8    7  9    3
       / \  /
      2  4 1
    ```

    Every parent is greater than or equal to its children: $16 \ge 14$, $16 \ge 10$, $14 \ge 8$, $14 \ge 7$, and so on.

## Min-Heap Property

A binary tree satisfies the **min-heap property** if, for every node $i$ other than the root, the value of $i$ is at least the value of its parent:

$$
A[\text{parent}(i)] \le A[i]
$$

The smallest element always resides at the root. Min-heaps are the default in Python's `heapq` module and are the natural choice for priority queues where the highest-priority item has the smallest key.

!!! example "Min-Heap Example"
    Consider the array `[1, 2, 3, 8, 7, 9, 10, 14, 4, 16]` stored as a min-heap:

    ```
              1
            /    \
          2        3
         /  \    /   \
        8    7  9    10
       / \  /
     14  4 16
    ```

    Every parent is less than or equal to its children: $1 \le 2$, $1 \le 3$, $2 \le 8$, $2 \le 7$, and so on.

## Why Partial Order Suffices

A fully sorted array supports $O(1)$ access to the minimum or maximum, but insertion and deletion cost $O(n)$ to maintain sorted order. A heap relaxes the ordering requirement: it only enforces the parent-child relationship, not sibling relationships. This partial order is exactly enough to support efficient priority queue operations.

The following table summarizes the complexity of core heap operations, all of which rely on the heap property:

| Operation | Description | Time Complexity |
|-----------|-------------|-----------------|
| Insert (sift up) | Add element and restore heap property upward | $O(\log n)$ |
| Extract root (sift down) | Remove root and restore heap property downward | $O(\log n)$ |
| Peek | Return root element without removal | $O(1)$ |
| Build heap | Convert unordered array into a heap | $O(n)$ |
| Heapsort | Build heap, then extract all elements | $O(n \log n)$ |

## Verifying the Heap Property

A simple recursive check confirms whether an array satisfies the max-heap property. The algorithm compares each node with its children and recurses on the subtrees.

```python
"""
Heap property verification.

Provides functions to check whether an array satisfies the
max-heap or min-heap property.
"""


# === Max-Heap Property Check ===

def is_max_heap(arr, i=0):
    """Check if arr satisfies the max-heap property starting at index i."""
    n = len(arr)
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[i]:
        return False
    if right < n and arr[right] > arr[i]:
        return False

    left_ok = is_max_heap(arr, left) if left < n else True
    right_ok = is_max_heap(arr, right) if right < n else True
    return left_ok and right_ok


# === Min-Heap Property Check ===

def is_min_heap(arr, i=0):
    """Check if arr satisfies the min-heap property starting at index i."""
    n = len(arr)
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] < arr[i]:
        return False
    if right < n and arr[right] < arr[i]:
        return False

    left_ok = is_min_heap(arr, left) if left < n else True
    right_ok = is_min_heap(arr, right) if right < n else True
    return left_ok and right_ok


# === Demonstration ===

if __name__ == "__main__":
    max_heap = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
    print(f"Array: {max_heap}")
    print(f"Is max-heap: {is_max_heap(max_heap)}")
    print(f"Is min-heap: {is_min_heap(max_heap)}")

    min_heap = [1, 2, 3, 8, 7, 9, 10, 14, 4, 16]
    print(f"\nArray: {min_heap}")
    print(f"Is max-heap: {is_max_heap(min_heap)}")
    print(f"Is min-heap: {is_min_heap(min_heap)}")

    not_heap = [3, 16, 10, 8, 7, 9, 1, 2, 4, 14]
    print(f"\nArray: {not_heap}")
    print(f"Is max-heap: {is_max_heap(not_heap)}")
    print(f"Is min-heap: {is_min_heap(not_heap)}")
```

**Output:**
```
Array: [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Is max-heap: True
Is min-heap: False

Array: [1, 2, 3, 8, 7, 9, 10, 14, 4, 16]
Is max-heap: False
Is min-heap: True

Array: [3, 16, 10, 8, 7, 9, 1, 2, 4, 14]
Is max-heap: False
Is min-heap: False
```

The verification runs in $O(n)$ time since it visits each node exactly once.

## Python heapq Module

Python's standard library provides a min-heap implementation through the `heapq` module. The module operates directly on ordinary lists, maintaining the min-heap property as an invariant.

```python
"""
Python heapq module demonstration.

Shows basic heap operations using the standard library min-heap.
"""

from heapq import heapify, heappop, heappush


# === Basic Heap Operations ===

if __name__ == "__main__":
    # Start with an unordered list
    lst = [4, 5, 1, 2, 3]
    print(f"Original list: {lst}")

    # Transform into a min-heap in O(n)
    heapify(lst)
    print(f"After heapify:  {lst}")

    # Extract minimum element in O(log n)
    smallest = heappop(lst)
    print(f"Popped {smallest}, heap is now: {lst}")

    # Insert new element in O(log n)
    heappush(lst, 0)
    print(f"Pushed 0, heap is now:  {lst}")

    # Extract all elements in sorted order
    sorted_result = []
    while lst:
        sorted_result.append(heappop(lst))
    print(f"Sorted extraction: {sorted_result}")
```

**Output:**
```
Original list: [4, 5, 1, 2, 3]
After heapify:  [1, 2, 4, 5, 3]
Popped 1, heap is now: [2, 3, 4, 5]
Pushed 0, heap is now:  [0, 3, 4, 5, 2]
Sorted extraction: [0, 2, 3, 4, 5]
```

??? tip "Simulating a Max-Heap with heapq"
    Since `heapq` only provides a min-heap, a common technique is to negate all values on insertion and negate again on extraction:

    ```python
    import heapq

    max_heap = []
    for val in [4, 5, 1, 2, 3]:
        heapq.heappush(max_heap, -val)

    # Extract maximum
    largest = -heapq.heappop(max_heap)  # returns 5
    ```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6: Heapsort. MIT Press.
- Python Documentation: [heapq -- Heap queue algorithm](https://docs.python.org/3/library/heapq.html)
