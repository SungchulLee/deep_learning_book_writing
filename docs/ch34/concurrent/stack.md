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

## Exercises

**Exercise 1.**
Describe the Treiber stack's `push` and `pop` operations using CAS. Why is the stack lock-free but not wait-free?

??? success "Solution to Exercise 1"
    **Push**: allocate a new node, set `node.next = top`. CAS `top` from the current value to the new node. If CAS fails (another thread modified `top`), reload `top`, update `node.next`, and retry. **Pop**: read `top`. If null, stack is empty. Otherwise, read `top.next`. CAS `top` from `top` to `top.next`. If CAS fails, retry. On success, return the value from the old `top`. The stack is lock-free because at least one thread's CAS succeeds in every contention round -- the thread whose CAS fails knows another thread succeeded, guaranteeing system-wide progress. It is not wait-free because an individual thread's CAS can fail unboundedly many times if other threads keep succeeding: in theory, one thread could starve while others continuously push and pop. Wait-freedom would require every thread to complete in a bounded number of steps. $\square$

---

**Exercise 2.**
Explain the ABA problem in the context of the Treiber stack and show a concrete scenario where it causes incorrect behavior.

??? success "Solution to Exercise 2"
    Scenario: stack contains $[A \to B \to C]$ (top = A). Thread 1 begins `pop`: reads `top = A`, `next = B`. Thread 1 is preempted. Thread 2 pops A (top becomes B), pops B (top becomes C), pushes A back (top = A, with `A.next = C`). Thread 1 resumes: CAS `top` from A to B succeeds (top is A). Now `top = B`, but B was already freed/popped -- it's a dangling pointer. The stack is corrupted. The CAS succeeded because the pointer value was the same (A), but the stack structure changed underneath. Solutions: (1) use double-width CAS with a version counter (each CAS increments the counter, so the A-with-counter-1 differs from A-with-counter-3); (2) use hazard pointers to prevent A's memory from being reused; (3) use epoch-based reclamation. $\square$

---

**Exercise 3.**
An elimination back-off stack combines a Treiber stack with an elimination array. Describe how it achieves higher throughput under contention.

??? success "Solution to Exercise 3"
    Under high contention, CAS retries on the Treiber stack's `top` pointer become the bottleneck because all threads compete for a single cache line. An elimination back-off stack adds an auxiliary array where threads can pair up: a `push` and a `pop` that collide in the array can exchange their values directly without touching the shared stack. On CAS failure, a thread picks a random slot in the elimination array and advertises its operation (push with value, or pop requesting value). If a complementary thread arrives at the same slot within a timeout, they exchange and both complete. If no match is found, the thread retries on the main stack. This scales throughput because paired operations bypass the contention point entirely. Under low contention, the elimination array is rarely used, and the Treiber stack handles operations directly. $\square$

---

**Exercise 4.**
Prove that the Treiber stack is linearizable: every concurrent execution is equivalent to some sequential execution where each push and pop takes effect at its successful CAS.

??? success "Solution to Exercise 4"
    Assign each operation a linearization point: the successful CAS instruction. For `push`, the linearization point is the CAS that swings `top` from old to the new node. For `pop`, it is the CAS that swings `top` from the current node to `top.next`. For an empty-stack `pop` (returning null), the linearization point is the read of `top == null`. Since CAS is an atomic instruction, the linearization points are totally ordered by their hardware execution times. In this ordering: after a push's CAS, the pushed element is on top of the stack; after a pop's CAS, the top element is removed. Any concurrent execution can be replayed by ordering operations at their linearization points, producing a valid sequential stack execution. This holds because the CAS atomically verifies and updates the stack state, ensuring no intermediate state is visible. $\square$

---

**Exercise 5.**
Compare the Treiber stack with a lock-based stack in terms of: (a) correctness guarantees, (b) performance under no contention, (c) performance under high contention, and (d) implementation complexity.

??? success "Solution to Exercise 5"
    (a) **Correctness**: both are linearizable. The lock-based stack is deadlock-free (with a single lock) and starvation-free (with a fair lock). The Treiber stack is lock-free (guaranteed system-wide progress) but not starvation-free for individual threads. (b) **No contention**: the lock-based stack incurs lock acquisition/release overhead ($\sim$20 ns for an uncontended mutex). The Treiber stack performs one CAS ($\sim$10 ns). The Treiber stack is slightly faster. (c) **High contention**: the lock-based stack serializes all operations at the lock, with throughput bounded by $\sim$1 / (lock cost + operation cost). The Treiber stack also serializes at the CAS point, with similar throughput, but CAS retries waste additional CPU cycles. Neither scales well; the elimination back-off stack is needed. (d) **Complexity**: the lock-based stack is trivial (wrap operations in `lock`/`unlock`). The Treiber stack requires careful memory reclamation (hazard pointers or epoch-based) to avoid ABA and use-after-free, significantly increasing complexity. $\square$
