# Min Stack

A standard stack supports push, pop, and peek in $O(1)$ time, but finding the minimum element requires scanning all elements in $O(n)$ time. In many applications --- priority-aware undo systems, sliding window minimum problems, stock price monitoring --- we need to retrieve the current minimum instantly. The **min stack** augments a standard stack so that `get_min` also runs in $O(1)$ time, at the cost of $O(n)$ additional space. This page describes the auxiliary stack technique, proves its correctness, and explores a space-optimized variant.

## The Problem

Design a stack that supports four operations, each in $O(1)$ time:

| Operation | Description |
|-----------|-------------|
| `push(x)` | Push element $x$ onto the stack |
| `pop()` | Remove and return the top element |
| `peek()` | Return the top element without removing it |
| `get_min()` | Return the minimum element in the stack |

The challenge is `get_min`: in a standard stack, the minimum might be buried anywhere, and there is no $O(1)$ way to locate it without auxiliary information.

## Auxiliary Stack Approach

The key idea is to maintain a second stack --- the **min stack** --- that tracks the running minimum. The min stack has the same height as the main stack, and its top element is always the current minimum of the main stack.

**Push rule**: when pushing $x$, also push $\min(x, \text{min\_stack.top})$ onto the min stack.

**Pop rule**: when popping from the main stack, also pop from the min stack.

**Get-min rule**: return the top of the min stack.

This works because the minimum of the first $k$ elements is fully determined by the minimum of the first $k-1$ elements and the $k$-th element. Popping an element removes it from consideration, and the min stack automatically reverts to the previous minimum.

??? example "Worked Example"
    Push sequence: 5, 3, 7, 2, 8

    | Operation | Main Stack | Min Stack | `get_min()` |
    |-----------|-----------|-----------|-------------|
    | `push(5)` | `[5]` | `[5]` | 5 |
    | `push(3)` | `[5, 3]` | `[5, 3]` | 3 |
    | `push(7)` | `[5, 3, 7]` | `[5, 3, 3]` | 3 |
    | `push(2)` | `[5, 3, 7, 2]` | `[5, 3, 3, 2]` | 2 |
    | `push(8)` | `[5, 3, 7, 2, 8]` | `[5, 3, 3, 2, 2]` | 2 |
    | `pop()` → 8 | `[5, 3, 7, 2]` | `[5, 3, 3, 2]` | 2 |
    | `pop()` → 2 | `[5, 3, 7]` | `[5, 3, 3]` | 3 |

    After popping 2, the minimum correctly reverts to 3.

## Implementation

```python
"""
Min stack — a stack that supports O(1) push, pop, peek, and get_min.

Uses an auxiliary stack to track the running minimum at each level.
"""


# === Min Stack with Auxiliary Stack ===========================================

class MinStack:
    """Stack supporting O(1) minimum queries via an auxiliary min stack."""

    def __init__(self):
        self._main = []
        self._mins = []

    def push(self, x):
        """Push x and update the running minimum."""
        self._main.append(x)
        current_min = min(x, self._mins[-1]) if self._mins else x
        self._mins.append(current_min)

    def pop(self):
        """Pop and return the top element, updating the minimum."""
        if not self._main:
            raise IndexError("pop from empty stack")
        self._mins.pop()
        return self._main.pop()

    def peek(self):
        """Return the top element without removing it."""
        if not self._main:
            raise IndexError("peek from empty stack")
        return self._main[-1]

    def get_min(self):
        """Return the minimum element in O(1) time."""
        if not self._mins:
            raise IndexError("get_min from empty stack")
        return self._mins[-1]

    def is_empty(self):
        """Return True if the stack is empty."""
        return len(self._main) == 0

    def __repr__(self):
        return f"MinStack(main={self._main}, mins={self._mins})"


# === Demonstration ============================================================

if __name__ == "__main__":
    ms = MinStack()

    operations = [
        ("push", 5), ("push", 3), ("push", 7), ("push", 2), ("push", 8),
        ("pop", None), ("pop", None), ("push", 1), ("pop", None), ("pop", None),
    ]

    print(f"{'Operation':<15s} {'Main Stack':<25s} {'Min Stack':<25s} {'Min':>5s}")
    print("-" * 72)

    for op, val in operations:
        if op == "push":
            ms.push(val)
            label = f"push({val})"
        else:
            popped = ms.pop()
            label = f"pop() → {popped}"

        if not ms.is_empty():
            print(f"{label:<15s} {str(ms._main):<25s} {str(ms._mins):<25s} {ms.get_min():>5}")
        else:
            print(f"{label:<15s} {str(ms._main):<25s} {str(ms._mins):<25s} {'N/A':>5}")
```

**Output:**
```
Operation       Main Stack                Min Stack                   Min
------------------------------------------------------------------------
push(5)         [5]                       [5]                           5
push(3)         [5, 3]                    [5, 3]                        3
push(7)         [5, 3, 7]                 [5, 3, 3]                     3
push(2)         [5, 3, 7, 2]              [5, 3, 3, 2]                  2
push(8)         [5, 3, 7, 2, 8]           [5, 3, 3, 2, 2]               2
pop() → 8       [5, 3, 7, 2]              [5, 3, 3, 2]                  2
pop() → 2       [5, 3, 7]                 [5, 3, 3]                     3
push(1)         [5, 3, 7, 1]              [5, 3, 3, 1]                  1
pop() → 1       [5, 3, 7]                 [5, 3, 3]                     3
pop() → 7       [5, 3]                    [5, 3]                        3
```

The min stack correctly tracks the minimum through all push and pop operations, reverting to the previous minimum whenever the current minimum is removed.

## Correctness

!!! info "Invariant"
    After every operation, `_mins[k]` equals the minimum of `_main[0], _main[1], ..., _main[k]` for all valid indices $k$.

**Proof by induction on the number of operations.**

*Base case*: after the first `push(x)`, `_main = [x]` and `_mins = [x]`, so `_mins[0] = min(x) = x`.

*Inductive step (push)*: suppose the invariant holds for a stack of height $k$. When we push $x$, we set `_mins[k] = min(x, _mins[k-1])`. Since `_mins[k-1]` is the minimum of the first $k$ elements (by hypothesis), `_mins[k] = min(x, min of first k) = min of first k+1` elements.

*Inductive step (pop)*: removing the top element from both `_main` and `_mins` restores both to their state before the last push, where the invariant held.

## Space-Optimized Variant

The auxiliary stack doubles memory usage. A space-optimized approach pushes onto the min stack only when the new element is less than or equal to the current minimum. On pop, the min stack is popped only if the popped element equals `_mins[-1]`. This saves space when there are many elements larger than the current minimum, though worst-case space remains $O(n)$.

## Complexity Summary

| Operation | Time | Space (total) |
|-----------|------|---------------|
| `push(x)` | $O(1)$ | $O(n)$ |
| `pop()` | $O(1)$ | $O(n)$ |
| `peek()` | $O(1)$ | $O(n)$ |
| `get_min()` | $O(1)$ | $O(n)$ |

All operations are $O(1)$ time. The $O(n)$ space accounts for both the main stack and the auxiliary min stack.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
