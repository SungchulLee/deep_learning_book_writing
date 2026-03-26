# Lock-Free Queue

Queues are fundamental building blocks in concurrent systems -- message passing, task scheduling, and producer-consumer patterns all rely on shared queues. A standard queue with a global lock serializes all operations, becoming a bottleneck under high contention. A **lock-free queue** uses atomic compare-and-swap (CAS) operations instead of locks, guaranteeing that at least one thread makes progress at all times, even when other threads are delayed or preempted.

## Michael-Scott Queue

The most widely used lock-free queue is the **Michael-Scott queue** (1996), which uses a singly-linked list with atomic head and tail pointers.

### Structure

- A sentinel (dummy) node separates the head from the tail.
- **Head** points to the sentinel node; the next element to dequeue is `head.next`.
- **Tail** points to the last node (or a node close to the last).
- All pointer updates use CAS: `CAS(addr, expected, new)` atomically writes `new` to `addr` only if its current value equals `expected`.

### Enqueue

1. Create a new node with the value to enqueue.
2. Read `tail` and `tail.next`.
3. If `tail.next` is null, attempt `CAS(tail.next, null, new_node)`.
    - On success, swing `tail` forward with `CAS(tail, old_tail, new_node)`.
    - On failure, retry from step 2.
4. If `tail.next` is not null (another thread already appended), help by swinging `tail` forward, then retry.

### Dequeue

1. Read `head`, `tail`, and `head.next`.
2. If `head.next` is null, the queue is empty.
3. Attempt `CAS(head, old_head, head.next)`.
    - On success, return the value from the old `head.next`.
    - On failure, retry from step 1.

## Simulation

Since Python lacks hardware CAS, we simulate the lock-free queue logic with a threading lock, focusing on the algorithm structure and correctness.

```python
"""
Lock-free queue simulation (Michael-Scott queue).

Simulates the CAS-based enqueue/dequeue algorithm. In a real
implementation, CAS would be a hardware atomic instruction.
"""

import threading

# ===================================================================
# Lock-Free Queue (Simulated)
# ===================================================================

class Node:
    """Queue node."""
    def __init__(self, value=None):
        self.value = value
        self.next = None


class LockFreeQueue:
    """Michael-Scott lock-free queue (simulated with locks).

    The algorithm structure follows the CAS-based design.
    Python's GIL and an explicit lock simulate atomic CAS.
    """

    def __init__(self):
        sentinel = Node()  # dummy node
        self.head = sentinel
        self.tail = sentinel
        self._lock = threading.Lock()  # simulates CAS

    def enqueue(self, value):
        """Add value to the back of the queue.

        Args:
            value: item to enqueue
        """
        new_node = Node(value)
        with self._lock:
            self.tail.next = new_node
            self.tail = new_node

    def dequeue(self):
        """Remove and return the front item.

        Returns:
            Value from the front, or None if empty
        """
        with self._lock:
            if self.head.next is None:
                return None
            value = self.head.next.value
            self.head = self.head.next
            return value

    def is_empty(self):
        """Check if queue is empty."""
        return self.head.next is None

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    queue = LockFreeQueue()

    # Single-threaded test
    for x in [10, 20, 30, 40]:
        queue.enqueue(x)

    print("Single-threaded dequeue:")
    while not queue.is_empty():
        print(f"  {queue.dequeue()}")

    # Multi-threaded producer-consumer
    queue = LockFreeQueue()
    produced = []
    consumed = []
    barrier = threading.Barrier(3)

    def producer(items):
        barrier.wait()
        for item in items:
            queue.enqueue(item)
            produced.append(item)

    def consumer(count):
        barrier.wait()
        local = []
        attempts = 0
        while len(local) < count and attempts < count * 10:
            val = queue.dequeue()
            if val is not None:
                local.append(val)
            attempts += 1
        consumed.extend(local)

    t1 = threading.Thread(target=producer, args=([1, 2, 3, 4, 5],))
    t2 = threading.Thread(target=producer, args=([6, 7, 8, 9, 10],))
    t3 = threading.Thread(target=consumer, args=(10,))

    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()

    print(f"\nProducer-consumer test:")
    print(f"  Produced: {sorted(produced)}")
    print(f"  Consumed: {sorted(consumed)}")
    print(f"  All consumed: {sorted(consumed) == list(range(1, 11))}")
```

**Output:**
```
Single-threaded dequeue:
  10
  20
  30
  40

Producer-consumer test:
  Produced: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  Consumed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  All consumed: True
```

## Progress Guarantees

| Guarantee | Definition |
|---|---|
| **Lock-free** | At least one thread completes its operation in a finite number of steps, regardless of other threads' progress |
| **Wait-free** | Every thread completes its operation in a bounded number of steps |
| **Obstruction-free** | A thread completes its operation in a finite number of steps if executed in isolation |

The Michael-Scott queue is **lock-free**: if one thread is delayed mid-operation, other threads can still make progress. It is not wait-free because a single thread may retry its CAS indefinitely under high contention.

## Complexity

| Operation | Time (amortized) |
|---|---|
| `enqueue` | $O(1)$ expected |
| `dequeue` | $O(1)$ expected |

Under high contention, CAS retries add overhead, but the expected number of retries is constant when contention is bounded.

## ABA Problem

A subtle correctness issue with CAS-based data structures:

1. Thread A reads value `X` from a pointer.
2. Thread B changes the pointer from `X` to `Y` to `X` (same bit pattern, different allocation).
3. Thread A's CAS succeeds because it sees `X`, but the underlying object has changed.

!!! warning "Preventing ABA"
    Common solutions include tagged pointers (append a version counter to each pointer) and hazard pointers (prevent memory reclamation while a thread holds a reference). Java's `AtomicStampedReference` implements tagged pointers.

## Reference

- Michael, M. M. and Scott, M. L. (1996). "Simple, fast, and practical non-blocking and blocking concurrent queue algorithms." *PODC*.
- Herlihy, M. and Shavit, N. *The Art of Multiprocessor Programming*, Chapter 10.
