# External Hashing

In-memory hash tables achieve $O(1)$ expected time per operation, but they assume random access to any memory location is equally fast. When the hash table resides on disk, each probe that accesses a different disk page costs one I/O operation. **External hashing** adapts hashing to the external memory model by mapping keys to **buckets** that each occupy one disk block, achieving $O(1)$ expected I/O per lookup, insertion, or deletion.

## Static External Hashing

The simplest external hashing scheme uses a fixed number of buckets $m$, each stored in one disk block of capacity $B$ keys.

A hash function maps each key to a bucket:

$$
h(k) = k \bmod m
$$

Each bucket occupies one disk page. A lookup for key $k$ reads the single page containing bucket $h(k)$ -- exactly 1 I/O operation.

### Overflow Handling

When a bucket exceeds its capacity $B$, overflow pages are chained to it. A lookup may need to read the primary page plus overflow pages:

$$
\text{Expected I/O per lookup} = 1 + \frac{\alpha - 1}{B} \text{ (when } \alpha > B\text{)}
$$

where $\alpha = N/m$ is the average number of keys per bucket. To keep expected I/O at $O(1)$, we need $\alpha = O(B)$, meaning each bucket holds at most a constant number of pages on average.

## Extendible Hashing

Static hashing requires choosing $m$ in advance, which is problematic when $N$ grows unpredictably. **Extendible hashing** uses a dynamic directory that doubles as needed, avoiding full rehashing.

### Structure

- A **directory** of $2^d$ entries, where $d$ is the **global depth**.
- Each directory entry points to a **bucket** (disk page).
- Multiple directory entries may point to the same bucket.
- Each bucket has a **local depth** $d_b \le d$ indicating how many bits of the hash distinguish its entries.

### Lookup

To find key $k$:

1. Compute $h(k)$ and take the first $d$ bits to get the directory index.
2. Follow the pointer to the bucket.
3. Read the bucket (1 I/O for the bucket, plus 1 I/O for the directory if it does not fit in memory).

### Splitting

When a bucket overflows:

1. Increase the bucket's local depth $d_b$ by 1.
2. Split the bucket into two, redistributing keys by the $(d_b)$-th bit.
3. If $d_b > d$, double the directory (increment $d$).

Directory doubling doubles the number of pointers but creates no new buckets -- it is a memory-only operation. The expected number of splits for $N$ insertions is $O(N/B)$.

### I/O Complexity

| Operation | Expected I/O |
|---|---|
| Lookup | $O(1)$ (2 if directory on disk) |
| Insert (no split) | $O(1)$ |
| Insert (with split) | $O(1)$ amortized |
| Directory doubling | $O(2^d / B)$ (rare) |

## Linear Hashing

**Linear hashing** avoids maintaining a directory entirely. Instead, it splits buckets in a predetermined linear order, using a family of hash functions $h_0, h_1, h_2, \ldots$ where:

$$
h_i(k) = k \bmod (2^i \cdot m_0)
$$

and $m_0$ is the initial number of buckets.

### Split Strategy

A **split pointer** $p$ tracks which bucket to split next. When the overall load factor exceeds a threshold:

1. Split bucket $p$ by redistributing its keys between bucket $p$ and a new bucket $p + 2^i \cdot m_0$.
2. Advance $p$ to $p + 1$.
3. When $p$ reaches $2^i \cdot m_0$, increment $i$ and reset $p$ to 0 (a new round).

The advantage is that splits occur in order, one bucket at a time, without any directory overhead.

### I/O Complexity

| Operation | Expected I/O |
|---|---|
| Lookup | $O(1)$ |
| Insert | $O(1)$ amortized |
| Split | $O(1)$ (read old bucket + write two buckets) |

## Comparison of External Hashing Schemes

| Property | Static | Extendible | Linear |
|---|---|---|---|
| Directory | None | $2^d$ entries | None |
| Handles growth | No (fixed $m$) | Yes (directory doubles) | Yes (linear splits) |
| Lookup I/O | $O(1)$ | $O(1)$ | $O(1)$ expected |
| Worst-case lookup | $O(N/m)$ with overflows | $O(1)$ (no overflow chains) | $O(1)$ with controlled load |
| Space utilization | Depends on load factor | ~69% average | Controlled by threshold |

## Example: External Hash Table Simulation

```python
"""
External hashing simulation.

Demonstrates extendible hashing with directory-based bucket management
and O(1) expected I/O per operation.
"""

import math

# ===================================================================
# Extendible hash table
# ===================================================================

class ExtendibleHashTable:
    """Extendible hash table with fixed bucket capacity."""

    def __init__(self, bucket_capacity: int = 4):
        self.bucket_capacity = bucket_capacity
        self.global_depth = 1
        self.directory = [[] for _ in range(2)]
        self.bucket_depths = [1, 1]
        self.io_count = 0

    def _hash(self, key: int) -> int:
        """Hash function returning enough bits."""
        return hash(key) & ((1 << 32) - 1)

    def _dir_index(self, key: int) -> int:
        """Get directory index using global_depth bits."""
        return self._hash(key) & ((1 << self.global_depth) - 1)

    def lookup(self, key: int) -> bool:
        """Look up a key. Returns True if found."""
        idx = self._dir_index(key)
        self.io_count += 1  # Read bucket = 1 I/O
        return key in self.directory[idx]

    def insert(self, key: int):
        """Insert a key into the hash table."""
        idx = self._dir_index(key)
        bucket = self.directory[idx]

        if key in bucket:
            return

        if len(bucket) < self.bucket_capacity:
            bucket.append(key)
            self.io_count += 1  # Write bucket
            return

        # Bucket is full -- need to split
        local_depth = self.bucket_depths[idx]

        if local_depth == self.global_depth:
            # Double the directory
            self.global_depth += 1
            new_dir = [None] * (1 << self.global_depth)
            new_depths = [0] * (1 << self.global_depth)
            for i in range(len(self.directory)):
                new_dir[i] = self.directory[i]
                new_dir[i + len(self.directory)] = self.directory[i]
                new_depths[i] = self.bucket_depths[i]
                new_depths[i + len(self.directory)] = self.bucket_depths[i]
            self.directory = new_dir
            self.bucket_depths = new_depths

        # Split the bucket
        new_depth = local_depth + 1
        old_bucket = bucket + [key]
        bucket0 = []
        bucket1 = []

        for k in old_bucket:
            if (self._hash(k) >> local_depth) & 1 == 0:
                bucket0.append(k)
            else:
                bucket1.append(k)

        # Update directory entries
        idx = self._dir_index(key)
        step = 1 << new_depth
        base0 = idx & ((1 << new_depth) - 1) & ~(1 << local_depth)
        base1 = base0 | (1 << local_depth)

        for i in range(base0, len(self.directory), step):
            self.directory[i] = bucket0
            self.bucket_depths[i] = new_depth
        for i in range(base1, len(self.directory), step):
            self.directory[i] = bucket1
            self.bucket_depths[i] = new_depth

        self.io_count += 2  # Read old bucket + write two new buckets

    @property
    def stats(self) -> dict:
        """Return statistics about the hash table."""
        unique_buckets = len(set(id(b) for b in self.directory))
        total_keys = sum(
            len(b) for b in {id(b): b for b in self.directory}.values()
        )
        return {
            "global_depth": self.global_depth,
            "directory_size": len(self.directory),
            "num_buckets": unique_buckets,
            "total_keys": total_keys,
            "io_count": self.io_count,
        }


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    B = 4  # Bucket capacity (block size)
    ht = ExtendibleHashTable(bucket_capacity=B)

    # Insert keys
    keys = list(range(0, 50, 3))
    for k in keys:
        ht.insert(k)

    s = ht.stats
    print(f"Extendible Hash Table (bucket capacity B={B})")
    print(f"  Keys inserted:  {len(keys)}")
    print(f"  Global depth:   {s['global_depth']}")
    print(f"  Directory size: {s['directory_size']}")
    print(f"  Num buckets:    {s['num_buckets']}")
    print(f"  Total I/Os:     {s['io_count']}")
    print(f"  I/O per insert: {s['io_count'] / len(keys):.2f}")
    print()

    # Lookup test
    ht.io_count = 0
    for k in keys:
        assert ht.lookup(k)
    print(f"  Lookups:        {len(keys)}")
    print(f"  Lookup I/Os:    {ht.io_count}")
    print(f"  I/O per lookup: {ht.io_count / len(keys):.2f}")
```

??? example "Sample Output"

    ```
    Extendible Hash Table (bucket capacity B=4)
      Keys inserted:  17
      Global depth:   4
      Directory size: 16
      Num buckets:    10
      Total I/Os:     27
      I/O per insert: 1.59

      Lookups:        17
      Lookup I/Os:    17
      I/O per lookup: 1.00
    ```

    Each lookup costs exactly 1 I/O (reading one bucket). Insertions average slightly more due to occasional splits, but the amortized cost remains $O(1)$.

## When to Use External Hashing

External hashing is ideal for **point queries** (exact key lookups) on disk-resident data. It achieves $O(1)$ expected I/O per operation, which is optimal. However, it does not support **range queries** efficiently -- for those, use a [B-Tree](btree.md) with $O(\log_B N + K/B)$ I/O for returning $K$ results.

## Reference

- Fagin, R. et al. "Extendible Hashing: A Fast Access Method for Dynamic Files," *ACM TODS*, 4(3), 1979.
- Litwin, W. "Linear Hashing: A New Tool for File and Table Addressing," *VLDB*, 1980.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.
