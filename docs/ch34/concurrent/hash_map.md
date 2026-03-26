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
