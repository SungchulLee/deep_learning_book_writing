# Lock-Free Stack

A concurrent stack must support `push` and `pop` operations from multiple threads without corruption. The simplest correct approach wraps operations in a mutex, but this serializes all access. The **Treiber stack** (1986) is the classic lock-free stack: it uses a singly-linked list with an atomic `top` pointer, and all modifications go through compare-and-swap (CAS). The result is a non-blocking stack where at least one thread always makes progress.

## Treiber Stack Algorithm

### Structure

The stack is a singly-linked list where each node points to the node below it. A single shared pointer `top` references the current stack top.

### Push

1. Create a new node with the value to push.
2. Set `new_node.next = top`.
3. Attempt `CAS(top, old_top, new_node)`.
    - On success, the push is complete.
    - On failure (another thread modified `top`), go to step 2 and retry.

### Pop

1. Read `old_top = top`.
2. If `old_top` is null, the stack is empty.
3. Read `new_top = old_top.next`.
4. Attempt `CAS(top, old_top, new_top)`.
    - On success, return `old_top.value`.
    - On failure, go to step 1 and retry.

Both operations are $O(1)$ in the uncontended case. Under contention, the expected number of CAS retries is bounded.

## Implementation

```python
"""
Lock-free stack (Treiber stack) simulation.

Uses a singly-linked list with a simulated CAS on the top
pointer. In production, CAS would be a hardware atomic
instruction.
"""

import threading

# ===================================================================
# Treiber Stack (Simulated Lock-Free)
# ===================================================================

class StackNode:
    """Stack node in singly-linked list."""

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class TreiberStack:
    """Lock-free stack using simulated CAS.

    The algorithm follows the Treiber stack design.
    Python's GIL + explicit lock simulates atomic CAS.
    """

    def __init__(self):
        self.top = None
        self._lock = threading.Lock()  # simulates CAS
        self._size = 0

    def push(self, value):
        """Push value onto the stack.

        Args:
            value: item to push
        """
        new_node = StackNode(value)
        with self._lock:
            new_node.next = self.top
            self.top = new_node
            self._size += 1

    def pop(self):
        """Pop and return the top value.

        Returns:
            Top value, or None if empty
        """
        with self._lock:
            if self.top is None:
                return None
            value = self.top.value
            self.top = self.top.next
            self._size -= 1
            return value

    def peek(self):
        """Return the top value without removing it."""
        if self.top is None:
            return None
        return self.top.value

    def is_empty(self):
        """Check if stack is empty."""
        return self.top is None

    def size(self):
        """Return current stack size."""
        return self._size

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    stack = TreiberStack()

    # Single-threaded test
    for x in [10, 20, 30, 40]:
        stack.push(x)

    print("Single-threaded (push 10,20,30,40 then pop all):")
    while not stack.is_empty():
        print(f"  pop: {stack.pop()}")

    # Multi-threaded push/pop
    stack = TreiberStack()
    pushed = []
    popped = []
    barrier = threading.Barrier(4)

    def pusher(items):
        barrier.wait()
        for item in items:
            stack.push(item)
            pushed.append(item)

    def popper(count):
        barrier.wait()
        local = []
        attempts = 0
        while len(local) < count and attempts < count * 20:
            val = stack.pop()
            if val is not None:
                local.append(val)
            attempts += 1
        popped.extend(local)

    t1 = threading.Thread(target=pusher, args=([1, 2, 3, 4, 5],))
    t2 = threading.Thread(target=pusher, args=([6, 7, 8, 9, 10],))
    t3 = threading.Thread(target=popper, args=(5,))
    t4 = threading.Thread(target=popper, args=(5,))

    for t in [t1, t2, t3, t4]:
        t.start()
    for t in [t1, t2, t3, t4]:
        t.join()

    print(f"\nMulti-threaded test:")
    print(f"  Pushed: {sorted(pushed)}")
    print(f"  Popped: {sorted(popped)}")
    remaining = []
    while not stack.is_empty():
        remaining.append(stack.pop())
    print(f"  Remaining in stack: {sorted(remaining)}")
    all_items = sorted(popped + remaining)
    print(f"  All accounted for: {all_items == list(range(1, 11))}")
```

**Output:**
```
Single-threaded (push 10,20,30,40 then pop all):
  pop: 40
  pop: 30
  pop: 20
  pop: 10

Multi-threaded test:
  Pushed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  Popped: [5, 6, 7, 8, 9, 10]
  Remaining in stack: [1, 2, 3, 4]
  All accounted for: True
```

## Progress Guarantee

The Treiber stack is **lock-free**: if any thread is suspended mid-operation, other threads can still complete their pushes and pops. This is stronger than a mutex-based stack, where a thread holding the lock can block all others indefinitely.

However, it is not **wait-free**: a single thread may retry its CAS arbitrarily many times if other threads keep succeeding first. In practice, under moderate contention, the retry count is very small.

## ABA Problem

The Treiber stack is susceptible to the ABA problem:

1. Thread A reads `top = X`, prepares to CAS `top` from `X` to `X.next`.
2. Thread B pops `X`, pops `Y`, pushes `X` back (same node, different stack state).
3. Thread A's CAS succeeds (sees `X`), but `X.next` now points to the wrong node.

!!! warning "Solutions to ABA"
    - **Tagged pointers**: Pair each pointer with a version counter. CAS checks both pointer and counter.
    - **Hazard pointers**: Prevent memory reclamation while any thread holds a reference.
    - **Epoch-based reclamation**: Delay freeing nodes until all threads have passed through a quiescent state.

## Elimination Stack

Under high contention, a **back-off** or **elimination** optimization can help. Threads that fail a CAS attempt to directly exchange values (one pusher and one popper cancel each other out without touching the shared stack). This converts contention into throughput.

## Complexity

| Operation | Expected Time |
|---|---|
| `push` | $O(1)$ amortized |
| `pop` | $O(1)$ amortized |
| Space | $O(n)$ |

## Reference

- Treiber, R. K. (1986). "Systems programming: Coping with parallelism." *IBM Research Report RJ 5118*.
- Herlihy, M. and Shavit, N. *The Art of Multiprocessor Programming*, Chapter 11.
