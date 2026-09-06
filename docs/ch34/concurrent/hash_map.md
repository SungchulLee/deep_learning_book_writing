# Concurrent Hash Map

A standard hash map provides $O(1)$ average-case lookup, insertion, and deletion, but these operations are not thread-safe. When multiple threads access the same hash map concurrently, data races cause corruption. A **concurrent hash map** provides the same $O(1)$ interface while guaranteeing correct behavior under concurrent access. The key design challenge is achieving high throughput by minimizing the scope and duration of synchronization.

## Concurrency Strategies

### Global Lock

The simplest approach wraps every operation with a single mutex. This is correct but serializes all access, eliminating any benefit from parallelism.

### Striped Locking

Partition the hash table into $k$ segments (stripes), each with its own lock. An operation on key $x$ acquires only the lock for segment $h(x) \bmod k$. Operations on different segments proceed in parallel.

**Throughput**: With $k$ stripes and $p$ threads, the expected throughput scales as $\min(p, k)$ when keys are uniformly distributed.

### Lock-Free Approach

Use atomic compare-and-swap (CAS) operations for insertions and deletions. No locks are held, so threads never block each other. This provides the highest throughput but is significantly harder to implement correctly.

## Striped Hash Map Implementation

```python
"""
Concurrent hash map with striped locking.

Uses multiple locks (stripes) to allow parallel access to
different segments of the hash table. Operations on different
stripes proceed concurrently.
"""

import threading
from collections import defaultdict

# ===================================================================
# Concurrent Hash Map
# ===================================================================

class ConcurrentHashMap:
    """Hash map with striped locking for thread safety.

    Args:
        num_stripes: number of lock stripes
        initial_capacity: initial number of buckets
    """

    def __init__(self, num_stripes=16, initial_capacity=64):
        self.num_stripes = num_stripes
        self.capacity = initial_capacity
        self.buckets = [[] for _ in range(self.capacity)]
        self.locks = [threading.Lock() for _ in range(num_stripes)]
        self.size = 0

    def _stripe(self, key):
        """Return the stripe index for a key."""
        return hash(key) % self.num_stripes

    def _bucket_index(self, key):
        """Return the bucket index for a key."""
        return hash(key) % self.capacity

    def get(self, key, default=None):
        """Thread-safe get.

        Args:
            key: lookup key
            default: value to return if key not found

        Returns:
            Stored value or default
        """
        stripe = self._stripe(key)
        with self.locks[stripe]:
            idx = self._bucket_index(key)
            for k, v in self.buckets[idx]:
                if k == key:
                    return v
            return default

    def put(self, key, value):
        """Thread-safe put.

        Args:
            key: key to insert or update
            value: value to associate with key
        """
        stripe = self._stripe(key)
        with self.locks[stripe]:
            idx = self._bucket_index(key)
            for i, (k, v) in enumerate(self.buckets[idx]):
                if k == key:
                    self.buckets[idx][i] = (key, value)
                    return
            self.buckets[idx].append((key, value))
            self.size += 1

    def delete(self, key):
        """Thread-safe delete.

        Args:
            key: key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        stripe = self._stripe(key)
        with self.locks[stripe]:
            idx = self._bucket_index(key)
            for i, (k, v) in enumerate(self.buckets[idx]):
                if k == key:
                    self.buckets[idx].pop(i)
                    self.size -= 1
                    return True
            return False

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    hmap = ConcurrentHashMap(num_stripes=4)

    # Single-threaded correctness check
    hmap.put("name", "Alice")
    hmap.put("age", 30)
    hmap.put("city", "NYC")
    print(f"get('name') = {hmap.get('name')}")
    print(f"get('age')  = {hmap.get('age')}")
    print(f"get('city') = {hmap.get('city')}")
    print(f"get('zip')  = {hmap.get('zip', 'N/A')}")

    # Multi-threaded insertion
    results = {}
    barrier = threading.Barrier(4)

    def worker(thread_id, count):
        barrier.wait()
        for i in range(count):
            key = f"t{thread_id}_k{i}"
            hmap.put(key, thread_id * 100 + i)
        results[thread_id] = count

    threads = [threading.Thread(target=worker, args=(t, 100))
               for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nMulti-threaded insertion:")
    print(f"  Threads: 4, items per thread: 100")
    print(f"  Total items: {hmap.size}")

    # Verify all items present
    all_found = all(
        hmap.get(f"t{t}_k{i}") is not None
        for t in range(4) for i in range(100)
    )
    print(f"  All items found: {all_found}")
```

**Output:**
```
get('name') = Alice
get('age')  = 30
get('city') = NYC
get('zip')  = N/A

Multi-threaded insertion:
  Threads: 4, items per thread: 100
  Total items: 403
  All items found: True
```

## Complexity

| Operation | Average Time | Amortized |
|---|---|---|
| `get` | $O(1 + n/m)$ | $O(1)$ with good load factor |
| `put` | $O(1 + n/m)$ | $O(1)$ amortized |
| `delete` | $O(1 + n/m)$ | $O(1)$ amortized |

Where $n$ is the number of entries and $m$ is the number of buckets. With a load factor $\alpha = n/m < 1$, all operations are $O(1)$.

## Resizing

When the load factor exceeds a threshold (typically 0.75), the table must be resized. In a concurrent setting, resizing is challenging:

- **Stop-the-world resize**: Acquire all stripe locks, double the table, rehash all entries, release locks. Simple but causes a pause.
- **Incremental resize**: Maintain old and new tables simultaneously. Migrate entries lazily during normal operations. More complex but avoids pauses.

## Comparison of Strategies

| Strategy | Throughput | Complexity | Blocking |
|---|---|---|---|
| Global lock | Low | Simple | Yes |
| Striped locking | Medium-High | Moderate | Per-stripe |
| Lock-free (CAS) | Highest | High | No |
| Read-write lock | Medium | Moderate | Writers block |

!!! tip "Choosing the right strategy"
    For read-heavy workloads, a read-write lock or lock-free design provides the best throughput. For balanced read-write workloads, striped locking offers a good compromise between performance and implementation complexity.

## Reference

- Herlihy, M. and Shavit, N. *The Art of Multiprocessor Programming*, Chapter 13 (Concurrent Hashing).
- Lea, D. (2003). "Overview of package java.util.concurrent." (Java ConcurrentHashMap design).

## Exercises

**Exercise 1.**
Explain the striped locking design used in Java's `ConcurrentHashMap`. How does it achieve higher throughput than a single global lock?

??? success "Solution to Exercise 1"
    Striped locking partitions the hash table into $S$ segments (stripes), each protected by its own lock. A key's stripe is determined by a hash of the key modulo $S$. Operations on keys in different stripes proceed in parallel without contention. With $S$ stripes and $T$ threads, the probability of two threads contending on the same lock is approximately $1/S$ per operation pair, compared to 1.0 with a global lock. Throughput scales nearly linearly with the number of threads up to $S$ (assuming uniform key distribution). Java's original `ConcurrentHashMap` used 16 segments by default, later replaced by a finer-grained per-bucket CAS approach in Java 8 that eliminates the fixed segment count entirely. $\square$

---

**Exercise 2.**
Describe the ABA problem in the context of a lock-free concurrent hash map that uses CAS on bucket pointers. Propose a solution.

??? success "Solution to Exercise 2"
    The ABA problem occurs when a CAS operation succeeds spuriously: thread 1 reads pointer value A, is suspended; thread 2 changes the pointer from A to B then back to A (e.g., deletes a node and inserts a new node at the same address); thread 1 resumes and CAS succeeds because the pointer is A again, even though the underlying data has changed. In a hash map, this can cause a thread to link a new node to a stale or freed node. Solutions: (1) **Tagged pointers**: pack a monotonically increasing version counter into the unused bits of the pointer (or use a double-width CAS). CAS compares both pointer and version, so ABA is detected. (2) **Hazard pointers**: defer memory reclamation until no thread holds a reference, preventing freed addresses from being reused. (3) **Epoch-based reclamation**: batch deferred frees by epoch, ensuring no concurrent reader sees a recycled address. $\square$

---

**Exercise 3.**
A concurrent hash map uses open addressing with linear probing. Explain why deletion is problematic and how tombstone markers solve the issue.

??? success "Solution to Exercise 3"
    In linear probing, a key $k$ is found by starting at $h(k)$ and scanning consecutive slots until finding $k$ or an empty slot. If we delete a key by marking its slot empty, a subsequent lookup for a different key that probed past the deleted slot will stop at the newly empty slot, falsely concluding the key is absent. Tombstones solve this: instead of emptying the slot, mark it as "deleted." Lookup treats tombstones as occupied (continues scanning past them), while insertion treats tombstones as available (can reuse the slot). In a concurrent setting, tombstones must be set atomically, and a thread performing lookup must handle the case where a slot transitions from occupied to tombstone during its scan. The downside is that tombstones degrade probe-chain length over time; periodic rehashing (under a write lock) is needed to reclaim them. $\square$

---

**Exercise 4.**
Prove that a concurrent hash map with $n$ buckets, load factor $\alpha = n_{\text{items}}/n$, and $k$ hash functions (cuckoo hashing) has expected $O(1)$ lookup time regardless of the number of concurrent readers.

??? success "Solution to Exercise 4"
    In cuckoo hashing, each key is stored at one of $k$ possible locations: $h_1(\text{key}), h_2(\text{key}), \ldots, h_k(\text{key})$. A lookup checks these $k$ locations and returns the key if found in any of them. Since $k$ is a constant (typically 2 or 3), lookup performs $k = O(1)$ memory accesses regardless of $\alpha$ or table size. Concurrent readers do not interfere with each other because reads are side-effect-free. Even without locks, each reader independently accesses the $k$ positions. The only concern is that a concurrent writer might relocate a key between two of its positions during a read, causing a false negative. This is resolved by having the reader retry (check all $k$ positions again) or by using a version counter. The expected $O(1)$ bound holds because the number of positions checked is always exactly $k$, independent of the number of concurrent readers. $\square$

---

**Exercise 5.**
Compare the throughput characteristics of three concurrent hash map designs -- global lock, striped locking, and lock-free CAS -- under read-heavy (95% reads) and write-heavy (50% writes) workloads. What design is best for each scenario?

??? success "Solution to Exercise 5"
    Under read-heavy (95% reads): global lock serializes all operations, making throughput inversely proportional to thread count due to contention. Striped locking allows parallel reads on different stripes but still serializes readers within a stripe. Lock-free CAS (or RCU-enhanced) allows fully parallel reads with zero synchronization overhead, achieving near-linear scaling. Best: lock-free or RCU. Under write-heavy (50% writes): global lock throughput collapses. Striped locking provides moderate parallelism -- writes to different stripes proceed in parallel, and with $S = 64$ stripes, contention probability per operation pair is $\approx 1.5\%$. Lock-free CAS suffers from high CAS retry rates under contention (each failed CAS wastes a retry cycle), but avoids deadlocks and priority inversion. Best: striped locking for simplicity and predictable performance; lock-free for maximum throughput if the retry rate is manageable. $\square$
