# Queue Abstract Data Type

A checkout line at a grocery store, a printer processing documents, and an operating system scheduling tasks all share a common pattern: items are served in the order they arrive. The **queue** abstract data type formalizes this "first in, first out" principle. Unlike a stack, which only exposes the most recently added element, a queue provides access to the *oldest* element --- the one that has been waiting the longest. This page defines the queue ADT, specifies its operations with their contracts, and establishes the time complexity guarantees.

## FIFO Principle

A queue maintains a sequence of elements governed by the **First-In, First-Out (FIFO)** policy. New elements are added at the **rear** (also called the *back* or *tail*) and removed from the **front** (also called the *head*). The element that has been in the queue the longest is the next one to be served.

Formally, if elements $e_1, e_2, \ldots, e_n$ are enqueued in that order, then successive dequeue operations return $e_1, e_2, \ldots, e_n$ --- the same order.

!!! info "FIFO vs LIFO"
    The queue's FIFO policy is the complement of the stack's LIFO policy. A stack reverses the input order; a queue preserves it. Choosing between them depends on whether the most recent or the oldest item should be processed next.

## Core Operations

The queue ADT specifies the following operations. Every correct implementation must provide all of them.

| Operation | Description | Precondition | Postcondition |
|-----------|-------------|--------------|---------------|
| `Enqueue(x)` | Insert element $x$ at the rear | None | Queue size increases by 1; $x$ becomes the new rear element |
| `Dequeue()` | Remove and return the front element | Queue is non-empty | Queue size decreases by 1; the second element becomes the new front |
| `Front()` / `Peek()` | Return the front element without removing it | Queue is non-empty | Queue is unchanged |
| `IsEmpty()` | Return whether the queue contains no elements | None | Queue is unchanged |
| `Size()` | Return the number of elements | None | Queue is unchanged |

!!! warning "Underflow"
    Calling `Dequeue()` or `Front()` on an empty queue is an **underflow error**. Implementations typically raise an exception. Client code should check `IsEmpty()` before accessing the front element.

## Time Complexity Contract

All five core operations must run in $O(1)$ time for the ADT to be useful. Both circular-array and linked-list implementations achieve this.

$$
T_{\text{Enqueue}} = T_{\text{Dequeue}} = T_{\text{Front}} = T_{\text{IsEmpty}} = T_{\text{Size}} = O(1)
$$

For dynamic arrays without circular indexing, `Dequeue` takes $O(n)$ time because all remaining elements must shift forward. This is why circular arrays or linked lists are preferred for queue implementations.

## Abstraction Barrier

Like the stack ADT, the queue ADT specifies the interface but not the internal representation. Client code depends only on the five operations and their guarantees, allowing the implementation to be swapped freely between array-based, circular-array, and linked-list variants.

```python
"""
Queue ADT — interface demonstration.

Shows that client code interacts only with the public operations
(enqueue, dequeue, front, is_empty, size) without knowing the
internal storage mechanism.
"""


# === Queue ADT Interface =====================================================

class Queue:
    """Queue following the FIFO (First-In, First-Out) principle."""

    def __init__(self):
        self._items = []

    def enqueue(self, x):
        """Add element x to the rear of the queue."""
        self._items.append(x)

    def dequeue(self):
        """Remove and return the front element. Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.pop(0)

    def front(self):
        """Return the front element without removing it."""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._items[0]

    def is_empty(self):
        """Return True if the queue contains no elements."""
        return len(self._items) == 0

    def size(self):
        """Return the number of elements in the queue."""
        return len(self._items)

    def __repr__(self):
        return f"Queue({self._items})"


# === Demonstration ============================================================

if __name__ == "__main__":
    q = Queue()
    print(f"Empty? {q.is_empty()}")       # True

    for x in [10, 20, 30, 40]:
        q.enqueue(x)
        print(f"Enqueued {x:>3} → {q}")

    print(f"Front: {q.front()}")          # 10
    print(f"Size:  {q.size()}")           # 4

    while not q.is_empty():
        print(f"Dequeued {q.dequeue():>3} → {q}")
```

**Output:**
```
Empty? True
Enqueued  10 → Queue([10])
Enqueued  20 → Queue([10, 20])
Enqueued  30 → Queue([10, 20, 30])
Enqueued  40 → Queue([10, 20, 30, 40])
Front: 10
Size:  4
Dequeued  10 → Queue([20, 30, 40])
Dequeued  20 → Queue([30, 40])
Dequeued  30 → Queue([40])
Dequeued  40 → Queue([])
```

The output confirms FIFO ordering: elements enqueued in order 10, 20, 30, 40 are dequeued in the same order. The `front` operation returns 10 --- the oldest element --- without modifying the queue.

!!! note "Implementation Note"
    The `pop(0)` call above takes $O(n)$ time because Python lists shift all remaining elements. This simple implementation illustrates the ADT interface. For $O(1)$ dequeue performance, use a circular array or linked list, as described on the sibling pages.

## Queue Invariant

A correct queue implementation must maintain the following invariant at all times:

!!! info "Queue Invariant"
    After any sequence of `Enqueue` and `Dequeue` operations, the element returned by `Front()` is the element earliest added by `Enqueue` that has not yet been removed by `Dequeue`. If no such element exists, the queue is empty.

This invariant is the formal statement of the FIFO property. Any implementation that violates it --- for example, by returning elements in a different order --- is not a valid queue.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
