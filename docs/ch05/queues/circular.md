# Circular Queue

A naive array-based queue uses the front of the array as the dequeue end. Each dequeue shifts all remaining elements forward, taking $O(n)$ time. A **circular queue** (also called a ring buffer) solves this by treating the underlying array as a circle: when the rear pointer reaches the end of the array, it wraps around to the beginning. Both enqueue and dequeue then take $O(1)$ time because no elements need to be shifted. This technique is widely used in operating system I/O buffers, audio processing, and network packet queues. This page explains the wraparound mechanism, handles the full-vs-empty ambiguity, and provides a complete implementation.

## The Wraparound Problem

In a linear array, after many enqueue and dequeue operations, both the front and rear pointers advance toward the end of the array. Even though positions at the beginning of the array are now free, they cannot be reused. The array appears full while most of its capacity is wasted.

A circular queue fixes this by using **modular arithmetic** to wrap pointers:

$$
\text{next}(i) = (i + 1) \bmod C
$$

where $C$ is the capacity of the underlying array. When the rear pointer reaches position $C-1$, the next enqueue places the element at position 0 (if it is free), reusing the space vacated by earlier dequeues.

## Full vs Empty Detection

Both the full and empty states have the same symptom: `front == rear`. Two common solutions disambiguate:

1. **Waste one slot**: keep one array slot always empty. The queue is full when `(rear + 1) mod C == front`. The maximum number of stored elements is $C - 1$.
2. **Use a count**: maintain a separate `size` variable. The queue is full when `size == C` and empty when `size == 0`.

This implementation uses the count approach for clarity.

## Implementation

```python
"""
Circular queue — fixed-size queue using a circular array.

Uses modular arithmetic for O(1) enqueue and dequeue by wrapping
the front and rear pointers around the end of the array.
"""


# === Circular Queue ===========================================================

class CircularQueue:
    """Fixed-capacity queue implemented as a circular array.

    Uses a size counter to distinguish full from empty states.
    All operations run in O(1) worst-case time.
    """

    def __init__(self, capacity):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._rear = 0
        self._size = 0

    def enqueue(self, x):
        """Add element x to the rear. Raises OverflowError if full."""
        if self.is_full():
            raise OverflowError("enqueue to full queue")
        self._data[self._rear] = x
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1

    def dequeue(self):
        """Remove and return the front element. Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        value = self._data[self._front]
        self._data[self._front] = None  # help garbage collection
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return value

    def front(self):
        """Return the front element without removing it."""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._data[self._front]

    def is_empty(self):
        """Return True if the queue contains no elements."""
        return self._size == 0

    def is_full(self):
        """Return True if the queue has reached its capacity."""
        return self._size == self._capacity

    def size(self):
        """Return the number of elements currently in the queue."""
        return self._size

    def _snapshot(self):
        """Return the internal array state for debugging."""
        return self._data.copy()

    def __repr__(self):
        # Show elements in logical order (front to rear)
        if self._size == 0:
            return "CircularQueue([])"
        elements = []
        i = self._front
        for _ in range(self._size):
            elements.append(self._data[i])
            i = (i + 1) % self._capacity
        return f"CircularQueue({elements})"


# === Demonstration ============================================================

if __name__ == "__main__":
    cq = CircularQueue(capacity=5)

    print(f"{'Operation':<20s} {'Logical':<25s} {'Internal Array':<25s} {'front':>5s} {'rear':>5s}")
    print("-" * 82)

    def show(label):
        print(f"{label:<20s} {str(cq):<25s} {str(cq._snapshot()):<25s} {cq._front:>5d} {cq._rear:>5d}")

    # Enqueue 1-4
    for x in [1, 2, 3, 4]:
        cq.enqueue(x)
        show(f"enqueue({x})")

    # Dequeue 2 elements
    for _ in range(2):
        val = cq.dequeue()
        show(f"dequeue() → {val}")

    # Enqueue 2 more (wraps around!)
    for x in [5, 6]:
        cq.enqueue(x)
        show(f"enqueue({x})")

    # Dequeue remaining
    while not cq.is_empty():
        val = cq.dequeue()
        show(f"dequeue() → {val}")
```

**Output:**
```
Operation            Logical                   Internal Array            front  rear
----------------------------------------------------------------------------------
enqueue(1)           CircularQueue([1])        [1, None, None, None, None]     0     1
enqueue(2)           CircularQueue([1, 2])     [1, 2, None, None, None]     0     2
enqueue(3)           CircularQueue([1, 2, 3])  [1, 2, 3, None, None]     0     3
enqueue(4)           CircularQueue([1, 2, 3, 4]) [1, 2, 3, 4, None]     0     4
dequeue() → 1        CircularQueue([2, 3, 4])  [None, 2, 3, 4, None]     1     4
dequeue() → 2        CircularQueue([3, 4])     [None, None, 3, 4, None]     2     4
enqueue(5)           CircularQueue([3, 4, 5])  [None, None, 3, 4, 5]     2     0
enqueue(6)           CircularQueue([3, 4, 5, 6]) [6, None, 3, 4, 5]     2     1
dequeue() → 3        CircularQueue([4, 5, 6])  [6, None, None, 4, 5]     3     1
dequeue() → 4        CircularQueue([5, 6])     [6, None, None, None, 5]     4     1
dequeue() → 5        CircularQueue([6])        [6, None, None, None, None]     0     1
dequeue() → 6        CircularQueue([])         [None, None, None, None, None]     1     1
```

The trace shows the wraparound in action. After enqueuing 1-4 and dequeuing 1-2, the rear pointer is at position 4. When we enqueue 5, it goes into position 4 and the rear wraps to position 0. Enqueuing 6 places it at position 0 --- reusing space freed by earlier dequeues. The logical order (front to rear) always reflects FIFO, even though elements are scattered across the internal array.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `enqueue(x)` | $O(1)$ | $O(1)$ |
| `dequeue()` | $O(1)$ | $O(1)$ |
| `front()` | $O(1)$ | $O(1)$ |
| `is_empty()` | $O(1)$ | $O(1)$ |
| `is_full()` | $O(1)$ | $O(1)$ |

Total space for the data structure is $O(C)$ where $C$ is the capacity. All operations are $O(1)$ worst-case --- no amortization needed, unlike dynamic arrays.

!!! tip "Fixed vs Dynamic Capacity"
    A circular queue has fixed capacity. If dynamic resizing is needed, the array must be doubled and all elements copied to the new array starting at position 0. This makes the occasional enqueue $O(n)$ but preserves $O(1)$ amortized time. For applications where the maximum queue size is known in advance (such as I/O buffers), fixed capacity is preferred.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
