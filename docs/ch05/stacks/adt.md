# Stack Abstract Data Type

Many everyday interactions follow a "last in, first out" pattern: the browser back button revisits pages in reverse order, the undo command in a text editor reverses the most recent action first, and a compiler matches closing braces with the nearest unmatched opener. All of these rely on the **stack** abstract data type, which enforces the constraint that only the most recently added element is accessible at any moment. This page defines the stack ADT formally, specifies its core operations with their contracts, and analyzes the time complexity that any correct implementation must achieve.

## LIFO Principle

A stack maintains a sequence of elements governed by the **Last-In, First-Out (LIFO)** policy. When a new element is added, it goes on top of the stack. When an element is removed, it is always the topmost element --- the one most recently added --- that leaves. This ordering distinguishes a stack from other container types like queues (FIFO) or random-access arrays.

Formally, if elements $e_1, e_2, \ldots, e_n$ are pushed in that order, then successive pop operations return $e_n, e_{n-1}, \ldots, e_1$.

## Core Operations

The stack ADT specifies the following operations. Every correct implementation must provide all of them.

| Operation | Description | Precondition | Postcondition |
|-----------|-------------|--------------|---------------|
| `Push(x)` | Insert element $x$ onto the top | None | Stack size increases by 1; $x$ becomes the new top |
| `Pop()` | Remove and return the top element | Stack is non-empty | Stack size decreases by 1; the previous second element becomes the new top |
| `Peek()` / `Top()` | Return the top element without removing it | Stack is non-empty | Stack is unchanged |
| `IsEmpty()` | Return whether the stack contains no elements | None | Stack is unchanged |
| `Size()` | Return the number of elements | None | Stack is unchanged |

!!! warning "Underflow"
    Calling `Pop()` or `Peek()` on an empty stack is an **underflow error**. Implementations typically raise an exception or return a sentinel value. A well-designed client always checks `IsEmpty()` before accessing the top element.

## Time Complexity Contract

A key property of the stack ADT is that all five core operations run in $O(1)$ worst-case time for both array-based and linked-list-based implementations.

$$
T_{\text{Push}} = T_{\text{Pop}} = T_{\text{Peek}} = T_{\text{IsEmpty}} = T_{\text{Size}} = O(1)
$$

For dynamic arrays, `Push` has $O(1)$ **amortized** time due to occasional resizing, but $O(n)$ worst-case for a single operation. Linked-list implementations achieve $O(1)$ worst-case for every operation.

## Abstraction Barrier

The ADT specifies **what** operations are available and **what guarantees** they provide, but says nothing about **how** data is stored internally. This separation is the abstraction barrier: client code depends only on the interface, so swapping one implementation for another (array for linked list, for instance) requires no changes to the client.

```python
"""
Stack ADT — interface demonstration.

Shows that client code interacts only with the public operations
(push, pop, peek, is_empty, size) without knowing the internal
storage mechanism.
"""


# === Stack ADT Interface =====================================================

class Stack:
    """Stack following the LIFO (Last-In, First-Out) principle."""

    def __init__(self):
        self._items = []

    def push(self, x):
        """Add element x to the top of the stack."""
        self._items.append(x)

    def pop(self):
        """Remove and return the top element. Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top element without removing it."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self):
        """Return True if the stack contains no elements."""
        return len(self._items) == 0

    def size(self):
        """Return the number of elements in the stack."""
        return len(self._items)

    def __repr__(self):
        return f"Stack({self._items})"


# === Demonstration ============================================================

if __name__ == "__main__":
    s = Stack()
    print(f"Empty? {s.is_empty()}")       # True

    for x in [10, 20, 30]:
        s.push(x)
        print(f"Pushed {x:>3} → {s}")

    print(f"Peek:  {s.peek()}")           # 30
    print(f"Size:  {s.size()}")           # 3

    while not s.is_empty():
        print(f"Popped {s.pop():>3} → {s}")
```

**Output:**
```
Empty? True
Pushed  10 → Stack([10])
Pushed  20 → Stack([10, 20])
Pushed  30 → Stack([10, 20, 30])
Peek:  30
Size:  3
Popped  30 → Stack([10, 20])
Popped  20 → Stack([10])
Popped  10 → Stack([])
```

The output confirms LIFO ordering: elements pushed in order 10, 20, 30 are popped in reverse order 30, 20, 10. The `peek` operation returns 30 --- the most recently pushed element --- without modifying the stack.

## Stack Invariant

A correct stack implementation must maintain the following invariant at all times:

!!! info "Stack Invariant"
    After any sequence of `Push` and `Pop` operations, the element returned by `Peek()` is the element most recently added by `Push` that has not yet been removed by `Pop`. If no such element exists, the stack is empty.

This invariant is the formal statement of the LIFO property. Any implementation that violates it --- for example, by returning elements in a different order --- is not a valid stack.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
