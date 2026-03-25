# Circular Arrays

When implementing a queue with a static array, elements are enqueued at the back and dequeued from the front. In a naive approach, dequeuing shifts all remaining elements left in $O(n)$ time, or alternatively the front index advances forward, eventually exhausting the array even when most slots are empty. A **circular array** (also called a ring buffer) solves both problems by treating the array as if it wraps around: when an index moves past the last slot, it jumps back to the first. This achieves $O(1)$ enqueue and dequeue without wasting any space.

## Modular Index Arithmetic

The wrapping behavior is implemented with the modulo operator. Given an array of capacity $c$, any index $i$ maps to a physical position via

$$
\text{pos}(i) = i \bmod c
$$

This means index $c$ wraps to position 0, index $c + 1$ wraps to position 1, and so on. The two key pointers maintained by a circular buffer are:

- **front**: the index of the first (oldest) element.
- **rear**: the index of the next available slot for insertion.

After each enqueue, the rear advances as

$$
\text{rear} \leftarrow (\text{rear} + 1) \bmod c
$$

After each dequeue, the front advances as

$$
\text{front} \leftarrow (\text{front} + 1) \bmod c
$$

The current number of elements is

$$
\text{size} = (\text{rear} - \text{front}) \bmod c
$$

## Full vs Empty Distinction

A subtle issue arises: when `front == rear`, the buffer could be either completely empty or completely full. Two common solutions exist:

1. **Waste one slot**: keep one slot always empty, so the buffer is full when $(\text{rear} + 1) \bmod c = \text{front}$. This limits usable capacity to $c - 1$.
2. **Maintain a count**: track the number of elements separately, allowing all $c$ slots to be used.

The implementation below uses the count-based approach.

## Operations and Complexity

| Operation | Time Complexity | Description                                 |
|-----------|-----------------|---------------------------------------------|
| Enqueue   | $O(1)$          | Write at rear, advance rear pointer         |
| Dequeue   | $O(1)$          | Read at front, advance front pointer        |
| Peek      | $O(1)$          | Read at front without advancing             |
| Is Empty  | $O(1)$          | Check if count equals zero                  |
| Is Full   | $O(1)$          | Check if count equals capacity              |

All operations are worst-case $O(1)$, not merely amortized. No element shifting or reallocation occurs.

## Implementation

```python
"""Circular buffer (ring buffer) implementation using a fixed-size array."""


# === Circular Buffer Class ===
class CircularBuffer:
    """A fixed-capacity circular buffer supporting O(1) enqueue and dequeue."""

    def __init__(self, capacity: int):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._rear = 0
        self._size = 0

    def is_empty(self) -> bool:
        return self._size == 0

    def is_full(self) -> bool:
        return self._size == self._capacity

    def enqueue(self, value) -> None:
        if self.is_full():
            raise OverflowError("Buffer is full")
        self._data[self._rear] = value
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Buffer is empty")
        value = self._data[self._front]
        self._data[self._front] = None  # help garbage collection
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return value

    def peek(self):
        if self.is_empty():
            raise IndexError("Buffer is empty")
        return self._data[self._front]

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        items = []
        idx = self._front
        for _ in range(self._size):
            items.append(repr(self._data[idx]))
            idx = (idx + 1) % self._capacity
        return f"CircularBuffer([{', '.join(items)}])"


# === Demonstration ===
if __name__ == "__main__":
    buf = CircularBuffer(4)
    buf.enqueue("A")
    buf.enqueue("B")
    buf.enqueue("C")
    print(f"After 3 enqueues: {buf}")

    print(f"Dequeue: {buf.dequeue()}")
    print(f"Dequeue: {buf.dequeue()}")
    print(f"After 2 dequeues: {buf}")

    buf.enqueue("D")
    buf.enqueue("E")
    print(f"After 2 more enqueues: {buf}")
    print(f"Internal array: {buf._data}")
```

**Output:**
```
After 3 enqueues: CircularBuffer(['A', 'B', 'C'])
Dequeue: A
Dequeue: B
After 2 dequeues: CircularBuffer(['C'])
After 2 more enqueues: CircularBuffer(['C', 'D', 'E'])
Internal array: ['E', None, 'C', 'D']
```

The internal array shows the wrap-around effect: after dequeuing A and B from positions 0 and 1, new elements D and E fill position 3 and wrap back to position 0.

??? example "Tracing the Wrap-Around"

    Starting with capacity 4, `front = 0`, `rear = 0`, `size = 0`:

    | Operation   | front | rear | size | Array State              |
    |-------------|-------|------|------|--------------------------|
    | enqueue(A)  | 0     | 1    | 1    | `[A, _, _, _]`           |
    | enqueue(B)  | 0     | 2    | 2    | `[A, B, _, _]`           |
    | enqueue(C)  | 0     | 3    | 3    | `[A, B, C, _]`           |
    | dequeue→A   | 1     | 3    | 2    | `[_, B, C, _]`           |
    | dequeue→B   | 2     | 3    | 1    | `[_, _, C, _]`           |
    | enqueue(D)  | 2     | 0    | 2    | `[_, _, C, D]`           |
    | enqueue(E)  | 2     | 1    | 3    | `[E, _, C, D]`           |

    When `rear` advances past index 3, it wraps to index 0 via $(3 + 1) \bmod 4 = 0$.

## Applications

Circular arrays are the standard backing structure for several important use cases:

- **Queue implementations**: the circular array queue provides $O(1)$ operations without wasted space, used in the array-based queue discussed in Chapter 5.
- **Bounded producer-consumer buffers**: operating systems and I/O systems use ring buffers to pass data between a producer and consumer at different speeds.
- **Streaming data**: audio processing, network packet buffers, and logging systems use circular buffers to keep the most recent $c$ elements and automatically discard older ones.
- **Sliding window algorithms**: maintaining a fixed-size window over a data stream maps directly to a circular buffer.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
