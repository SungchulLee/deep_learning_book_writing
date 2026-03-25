# Array-Based Deque

The [deque ADT](adt.md) requires $O(1)$ insertion and deletion at both ends.  A naive array supports efficient operations only at the back: pushing to the front forces every element to shift right, costing $O(n)$.  A **circular array** (ring buffer) eliminates this problem by treating the underlying array as if its ends were connected, so both the front and the back can grow and shrink with simple index arithmetic.  This page describes the circular-array deque, derives its amortized complexity, and provides a complete Python implementation.

## Circular Buffer Idea

A circular array of capacity $C$ stores elements in positions $0, 1, \dots, C-1$.  Two indices track the deque's boundaries:

- **front**: the index of the first element.
- **back**: the index *one past* the last element (the next free slot at the back).

The size of the deque is

$$
n = (\texttt{back} - \texttt{front}) \bmod C
$$

All index arithmetic uses modular arithmetic so that incrementing past $C-1$ wraps around to $0$, and decrementing past $0$ wraps around to $C-1$.

## Operations

### Push Back

To insert an element $x$ at the back:

1. Store $x$ at index `back`.
2. Update `back = (back + 1) mod C`.

### Push Front

To insert an element $x$ at the front:

1. Update `front = (front - 1) mod C`.
2. Store $x$ at the new `front` index.

### Pop Back

To remove the element at the back:

1. Update `back = (back - 1) mod C`.
2. Return the element at the new `back` index.

### Pop Front

To remove the element at the front:

1. Save the element at index `front`.
2. Update `front = (front + 1) mod C`.
3. Return the saved element.

Each of these four operations performs a constant number of steps, giving $O(1)$ worst-case time when the array does not need to resize.

## Dynamic Resizing

A fixed-capacity circular array becomes full when $n = C - 1$ (one slot is kept empty to distinguish full from empty).  To support an arbitrary number of elements, the array is **doubled** when full and **halved** when the size drops below $C / 4$.

**Doubling** (when the array is full):

1. Allocate a new array of size $2C$.
2. Copy all $n$ elements to positions $0, 1, \dots, n-1$ in the new array.
3. Set `front = 0` and `back = n`.

**Halving** (when $n < C / 4$):

1. Allocate a new array of size $C / 2$.
2. Copy all $n$ elements to positions $0, 1, \dots, n-1$.
3. Set `front = 0` and `back = n`.

### Amortized Analysis

Individual push or pop operations cost $O(1)$ without resizing.  A resize copies $n$ elements and costs $O(n)$.  Using the standard doubling argument (see the [amortized analysis pages](../../ch02/amortized/aggregate.md)), any sequence of $m$ push and pop operations starting from an empty deque takes $O(m)$ total time.

$$
\text{Amortized cost per operation} = O(1)
$$

!!! tip "Accounting method intuition"
    Charge each push operation $3$ units: $1$ for the actual insertion and $2$ saved as credit.  When a doubling occurs, the $n$ elements in the array have collectively saved at least $n$ credits, which pay for the $O(n)$ copy.

## Python Implementation

```python
"""Array-based deque using a circular buffer with dynamic resizing."""


# === Circular Array Deque ===

class ArrayDeque:
    """A double-ended queue backed by a circular array.

    All four push/pop operations run in O(1) amortized time.
    The internal array doubles when full and halves when one-quarter full.
    """

    _MIN_CAPACITY = 8

    def __init__(self):
        self._capacity = self._MIN_CAPACITY
        self._data = [None] * self._capacity
        self._front = 0
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    # --- Push operations ---

    def push_back(self, x):
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        back = (self._front + self._size) % self._capacity
        self._data[back] = x
        self._size += 1

    def push_front(self, x):
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._front = (self._front - 1) % self._capacity
        self._data[self._front] = x
        self._size += 1

    # --- Pop operations ---

    def pop_front(self):
        if self.is_empty():
            raise IndexError("pop from empty deque")
        value = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(max(self._MIN_CAPACITY, self._capacity // 2))
        return value

    def pop_back(self):
        if self.is_empty():
            raise IndexError("pop from empty deque")
        back = (self._front + self._size - 1) % self._capacity
        value = self._data[back]
        self._data[back] = None
        self._size -= 1
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(max(self._MIN_CAPACITY, self._capacity // 2))
        return value

    # --- Peek operations ---

    def front(self):
        if self.is_empty():
            raise IndexError("front from empty deque")
        return self._data[self._front]

    def back(self):
        if self.is_empty():
            raise IndexError("back from empty deque")
        return self._data[(self._front + self._size - 1) % self._capacity]

    # --- Internal ---

    def _resize(self, new_capacity):
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[(self._front + i) % self._capacity]
        self._data = new_data
        self._front = 0
        self._capacity = new_capacity


# === Demo ===

if __name__ == "__main__":
    dq = ArrayDeque()
    for v in [10, 20, 30]:
        dq.push_back(v)
    dq.push_front(5)
    print(f"Front: {dq.front()}")   # 5
    print(f"Back:  {dq.back()}")    # 30
    print(f"Size:  {len(dq)}")      # 4
    print(f"Pop front: {dq.pop_front()}")  # 5
    print(f"Pop back:  {dq.pop_back()}")   # 30
    print(f"Size:  {len(dq)}")      # 2
```

## Complexity Summary

| Operation | Worst case | Amortized |
|---|---|---|
| `push_front(x)` | $O(n)$ (resize) | $O(1)$ |
| `push_back(x)` | $O(n)$ (resize) | $O(1)$ |
| `pop_front()` | $O(n)$ (resize) | $O(1)$ |
| `pop_back()` | $O(n)$ (resize) | $O(1)$ |
| `front()` / `back()` | $O(1)$ | $O(1)$ |
| `is_empty()` / `size()` | $O(1)$ | $O(1)$ |
| Space | — | $O(n)$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
