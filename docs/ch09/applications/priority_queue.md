# Priority Queue

A **priority queue** is an abstract data type that stores elements with associated priorities and supports efficient retrieval of the highest-priority element. While the ADT can be implemented with sorted arrays or linked lists, a binary heap provides the best balance of operation costs -- $O(\log n)$ insertion and extraction, with $O(1)$ peek. Priority queues are used throughout computer science: in Dijkstra's shortest-path algorithm, Huffman coding, event-driven simulation, and operating system task scheduling.

## Abstract Data Type

A priority queue supports the following core operations:

| Operation | Description | Heap Cost |
|-----------|-------------|-----------|
| `insert(key, priority)` | Add an element with given priority | $O(\log n)$ |
| `extract_min` / `extract_max` | Remove and return highest-priority element | $O(\log n)$ |
| `peek` | Return highest-priority element without removal | $O(1)$ |
| `decrease_key(i, new_key)` | Reduce the key of element at position $i$ | $O(\log n)$ |
| `is_empty` | Check if the queue contains any elements | $O(1)$ |

A **min-priority queue** treats the smallest key as highest priority (used in Dijkstra's algorithm). A **max-priority queue** treats the largest key as highest priority (used in scheduling by deadline).

## Heap-Based Implementation

A binary min-heap directly implements a min-priority queue. The heap property ensures that the minimum element is always at the root, enabling $O(1)$ peek. The key operations map to heap operations:

- `insert` = append to end + sift up
- `extract_min` = save root, move last to root, sift down
- `decrease_key` = reduce key at index $i$, then sift up

### Decrease-Key

The **decrease-key** operation reduces the key of an element at a known position $i$ to a new, smaller value. Since the new key may be smaller than the parent, we sift up from position $i$:

```
DECREASE-KEY(A, i, new_key):
    if new_key > A[i]:
        error "new key is larger than current key"
    A[i] = new_key
    while i > 0 and A[parent(i)] > A[i]:
        swap A[i] and A[parent(i)]
        i = parent(i)
```

This operation is critical in graph algorithms like Dijkstra and Prim, where edge relaxation requires updating the priority of a vertex already in the queue.

## Implementation

```python
"""
Priority queue implemented with a binary min-heap.

Supports insert, extract-min, peek, decrease-key, and
handles (priority, value) pairs for practical usage.
"""


# === Min-Priority Queue ===

class MinPriorityQueue:
    """A min-priority queue backed by a binary min-heap.

    Elements are stored as (priority, value) pairs.
    The element with the smallest priority is extracted first.
    """

    def __init__(self):
        self.heap = []

    def _sift_up(self, i):
        """Move element at index i up to restore heap property."""
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i][0] < self.heap[parent][0]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        """Move element at index i down to restore heap property."""
        n = len(self.heap)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right

            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

    def insert(self, priority, value):
        """Insert an element with the given priority. O(log n)."""
        self.heap.append((priority, value))
        self._sift_up(len(self.heap) - 1)

    def peek(self):
        """Return the minimum-priority element without removal. O(1)."""
        if not self.heap:
            raise IndexError("peek from empty priority queue")
        return self.heap[0]

    def extract_min(self):
        """Remove and return the minimum-priority element. O(log n)."""
        if not self.heap:
            raise IndexError("extract from empty priority queue")
        min_elem = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        return min_elem

    def decrease_key(self, i, new_priority):
        """Decrease the priority of element at index i. O(log n)."""
        if new_priority > self.heap[i][0]:
            raise ValueError("new priority is larger than current priority")
        self.heap[i] = (new_priority, self.heap[i][1])
        self._sift_up(i)

    def is_empty(self):
        """Check if the queue is empty. O(1)."""
        return len(self.heap) == 0

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return f"MinPriorityQueue({self.heap})"


# === Demonstration ===

if __name__ == "__main__":
    pq = MinPriorityQueue()

    # Simulate task scheduling
    tasks = [
        (3, "low-priority task"),
        (1, "urgent task"),
        (2, "medium task"),
        (5, "background task"),
        (1, "another urgent task"),
    ]

    print("Inserting tasks:")
    for priority, task in tasks:
        pq.insert(priority, task)
        print(f"  Inserted ({priority}, '{task}')")

    print(f"\nPeek: {pq.peek()}")

    print("\nExtracting tasks in priority order:")
    while not pq.is_empty():
        priority, task = pq.extract_min()
        print(f"  ({priority}) {task}")
```

**Output:**
```
Inserting tasks:
  Inserted (3, 'low-priority task')
  Inserted (1, 'urgent task')
  Inserted (2, 'medium task')
  Inserted (5, 'background task')
  Inserted (1, 'another urgent task')

Peek: (1, 'urgent task')

Extracting tasks in priority order:
  (1) urgent task
  (1) another urgent task
  (2) medium task
  (3) low-priority task
  (5) background task
```

## Python heapq as a Priority Queue

Python's `heapq` module provides a min-heap that works on lists of tuples. When elements are tuples, Python compares by the first element (the priority), making it natural for priority queue usage.

```python
"""
Using Python's heapq module as a priority queue.
"""

import heapq


# === heapq-Based Priority Queue ===

if __name__ == "__main__":
    pq = []

    # Push tasks with priorities
    heapq.heappush(pq, (3, "write report"))
    heapq.heappush(pq, (1, "fix critical bug"))
    heapq.heappush(pq, (2, "review PR"))
    heapq.heappush(pq, (1, "deploy hotfix"))

    print("Processing tasks:")
    while pq:
        priority, task = heapq.heappop(pq)
        print(f"  Priority {priority}: {task}")
```

**Output:**
```
Processing tasks:
  Priority 1: deploy hotfix
  Priority 1: fix critical bug
  Priority 2: review PR
  Priority 3: write report
```

??? warning "Limitation: No Efficient Decrease-Key"
    Python's `heapq` does not support decrease-key directly because it has no way to locate an element by identity in $O(1)$ time. Common workarounds include:

    1. **Lazy deletion**: mark entries as invalid and push a new entry with the updated priority. Skip invalid entries during extraction.
    2. **Index map**: maintain a dictionary mapping values to heap indices, updating the map on every swap.

## Applications

Priority queues implemented with heaps appear in numerous algorithms:

| Application | Queue Type | Key Operation |
|------------|-----------|---------------|
| Dijkstra's shortest path | Min-PQ | decrease-key on relaxed vertices |
| Prim's MST | Min-PQ | decrease-key on frontier vertices |
| Huffman coding | Min-PQ | extract two lowest-frequency nodes |
| Event-driven simulation | Min-PQ | extract next event by timestamp |
| Job scheduling | Max-PQ | extract highest-priority job |
| Median maintenance | Two PQs | balance max-PQ and min-PQ |

## Complexity Comparison

| Implementation | Insert | Extract | Peek | Decrease-Key |
|---------------|--------|---------|------|-------------|
| Unsorted array | $O(1)$ | $O(n)$ | $O(n)$ | $O(1)$ |
| Sorted array | $O(n)$ | $O(1)$ | $O(1)$ | $O(n)$ |
| Binary heap | $O(\log n)$ | $O(\log n)$ | $O(1)$ | $O(\log n)$ |
| Fibonacci heap | $O(1)$ amortized | $O(\log n)$ amortized | $O(1)$ | $O(1)$ amortized |

The binary heap offers the best practical trade-off for most applications. Fibonacci heaps provide better theoretical bounds for decrease-key-heavy algorithms but are rarely used in practice due to large constants.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.5: Priority queues. MIT Press.
