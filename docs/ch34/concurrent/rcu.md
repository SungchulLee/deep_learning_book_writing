# Read-Copy-Update

Many concurrent data structures are read far more often than they are written. In these read-dominated workloads, paying synchronization overhead on every read operation wastes performance. **Read-Copy-Update (RCU)** eliminates read-side synchronization entirely: readers access shared data without locks, memory barriers, or atomic instructions. Writers create a modified copy of the data and atomically swap the pointer, then wait for all pre-existing readers to finish before reclaiming the old version.

## Core Principles

RCU rests on three mechanisms:

1. **Publish-subscribe**: Writers atomically publish a new version of the data structure by updating a shared pointer. Readers subscribe by loading this pointer.
2. **Grace period**: After publishing a new version, the writer waits until all readers that might still reference the old version have completed. This waiting period is the *grace period*.
3. **Reclamation**: After the grace period ends, the old version can be safely freed because no reader holds a reference to it.

## How It Works

### Read Side

A reader enters an **RCU read-side critical section**, reads the shared pointer, uses the data, and exits the critical section. No locks are acquired. The critical section simply marks the reader as "active" so that writers know not to reclaim data yet.

### Write Side

A writer performs four steps:

1. **Read** the current pointer to the data structure.
2. **Copy** the data and apply modifications to the copy.
3. **Publish** the copy by atomically replacing the shared pointer.
4. **Synchronize**: Call a grace-period mechanism to wait until all pre-existing readers have exited their critical sections.
5. **Reclaim** the old copy.

!!! warning "Readers see old or new, never partial"
    Because the pointer swap is atomic, a reader always sees either the complete old version or the complete new version -- never a partially modified state. This provides a consistency guarantee without locks.

## Implementation

```python
"""
Read-Copy-Update (RCU) simulation.

Simulates RCU with a shared immutable data snapshot. Writers
create new snapshots; readers access the current snapshot
without locking. A grace period ensures safe reclamation.
"""

import threading
import time
import copy

# ===================================================================
# RCU Simulation
# ===================================================================

class RCUProtected:
    """RCU-protected shared data.

    Readers access data lock-free. Writers create copies,
    modify them, and atomically swap the reference.

    Args:
        initial_data: initial value of the shared data
    """

    def __init__(self, initial_data):
        self._data = initial_data
        self._write_lock = threading.Lock()
        self._reader_count = 0
        self._reader_lock = threading.Lock()
        self._versions_reclaimed = 0

    def read(self):
        """Begin an RCU read-side critical section.

        Returns:
            Reference to current data snapshot (immutable view)
        """
        with self._reader_lock:
            self._reader_count += 1
        return self._data

    def read_done(self):
        """Exit an RCU read-side critical section."""
        with self._reader_lock:
            self._reader_count -= 1

    def update(self, modify_fn):
        """Perform an RCU update.

        Args:
            modify_fn: function that takes old data, returns new data

        Returns:
            The new data version
        """
        with self._write_lock:
            old_data = self._data
            new_data = modify_fn(copy.deepcopy(old_data))
            # Publish (atomic pointer swap)
            self._data = new_data
            # Grace period: wait for readers of old version
            self._synchronize()
            # Reclaim old version (in Python, garbage collector handles this)
            self._versions_reclaimed += 1
            return new_data

    def _synchronize(self):
        """Wait until all pre-existing readers have finished."""
        while True:
            with self._reader_lock:
                if self._reader_count == 0:
                    return
            time.sleep(0.001)

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    data = {"users": ["Alice", "Bob"], "count": 2}
    rcu = RCUProtected(data)

    print("RCU simulation:")

    # Reader: access without locking
    snapshot = rcu.read()
    print(f"  Reader sees: {snapshot}")
    rcu.read_done()

    # Writer: create new version
    def add_user(data):
        data["users"].append("Charlie")
        data["count"] += 1
        return data

    new_data = rcu.update(add_user)
    print(f"  After update: {new_data}")

    # Multiple readers during an update
    results = []
    barrier = threading.Barrier(3)

    def reader(reader_id):
        barrier.wait()
        snap = rcu.read()
        results.append((reader_id, len(snap["users"])))
        time.sleep(0.01)  # simulate work
        rcu.read_done()

    def writer():
        barrier.wait()
        rcu.update(lambda d: {**d, "users": d["users"] + ["Dave"],
                               "count": d["count"] + 1})

    threads = [
        threading.Thread(target=reader, args=(1,)),
        threading.Thread(target=reader, args=(2,)),
        threading.Thread(target=writer),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = rcu.read()
    rcu.read_done()

    print(f"\n  Concurrent readers saw user counts: "
          f"{[r[1] for r in sorted(results)]}")
    print(f"  Final state: {final}")
    print(f"  Versions reclaimed: {rcu._versions_reclaimed}")
```

**Output:**
```
RCU simulation:
  Reader sees: {'users': ['Alice', 'Bob'], 'count': 2}
  After update: {'users': ['Alice', 'Bob', 'Charlie'], 'count': 3}

  Concurrent readers saw user counts: [3, 3]
  Final state: {'users': ['Alice', 'Bob', 'Charlie', 'Dave'], 'count': 4}
  Versions reclaimed: 2
```

## Complexity

| Operation | Cost |
|---|---|
| Read-side entry/exit | $O(1)$, no synchronization |
| Write (copy + publish) | $O(n)$ for data of size $n$ |
| Grace period (synchronize) | $O(1)$ amortized with quiescent-state tracking |

## Trade-offs

| Property | RCU | Reader-Writer Lock |
|---|---|---|
| Read overhead | Zero (no lock, no barrier) | Acquire/release shared lock |
| Write overhead | Copy data + grace period | Acquire exclusive lock |
| Best for | Read-dominated workloads | Balanced read-write |
| Memory | Extra copy during update | No extra copy |
| Staleness | Readers may see old version briefly | All see same version |

## Applications

- **Linux kernel**: RCU is used extensively for routing tables, file system caches, and module lists. The kernel's RCU implementation handles millions of read operations per second.
- **Concurrent data structures**: RCU-protected linked lists, hash tables, and trees enable lock-free reads.
- **Configuration updates**: Application config can be updated via RCU: readers always see a consistent snapshot.

## Reference

- McKenney, P. E. (2004). "Exploiting Deferred Destruction: An Analysis of Read-Copy-Update Techniques in Operating System Kernels." *PhD Thesis, OGI*.
- McKenney, P. E. and Slingwine, J. D. (1998). "Read-Copy Update: Using Execution History to Solve Concurrency Problems." *PDCS*.
