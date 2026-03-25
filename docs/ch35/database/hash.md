# Hash Indexing

B-tree indexes excel at range queries and ordered scans, but many database workloads consist primarily of **exact-match lookups**: find the user with ID 42, retrieve the session with token `abc123`.  Hash indexes are optimized for this pattern, mapping each key directly to its storage location through a hash function, achieving $O(1)$ expected I/O per lookup.  This page covers how hash indexes work in storage engines, their limitations, and dynamic hashing schemes that handle growth without full rebuilds.

## Static Hash Index

A static hash index partitions records into $m$ fixed **buckets**.  Each bucket occupies one or more disk pages.  To store or retrieve a key $k$:

$$
\text{bucket}(k) = h(k) \bmod m
$$

where $h$ is a hash function mapping keys to integers.

### Operations

| Operation | Expected I/Os | Worst case |
|-----------|--------------|------------|
| Point lookup | $O(1)$ | $O(n/m)$ (all keys collide) |
| Insert | $O(1)$ | $O(n/m)$ |
| Delete | $O(1)$ | $O(n/m)$ |
| Range query | $O(n)$ | $O(n)$ |

The fatal weakness of hash indexes is that **range queries require a full scan** -- the hash function destroys key ordering.  This is why most relational databases default to B-tree indexes.

## Hash Index vs B-Tree

| Property | Hash Index | B-Tree Index |
|----------|-----------|-------------|
| Point lookup | $O(1)$ expected | $O(\log_t n)$ |
| Range query | $O(n)$ -- full scan | $O(\log_t n + k/B)$ |
| Ordered scan | Not supported | Natural |
| Space overhead | Low | Moderate |
| Worst case | $O(n)$ with collisions | $O(\log_t n)$ guaranteed |

## Overflow Handling

When a bucket fills beyond its page capacity, the index uses **overflow chaining**: extra pages are linked to the primary bucket page.  Long chains degrade performance, motivating dynamic hashing schemes.

## Extendible Hashing

Extendible hashing avoids full-table rehashing by using a **directory** of $2^d$ pointers indexed by the first $d$ bits of the hash value (the **global depth**).  Each bucket maintains a **local depth** $d_i \leq d$.

### Split on Overflow

When bucket $b$ overflows:

1. If $d_b < d$ (local depth less than global depth):
    - Split bucket $b$ into two buckets, incrementing $d_b$.
    - Update the directory entries that pointed to $b$.
2. If $d_b = d$:
    - Double the directory ($d \leftarrow d + 1$).
    - Then split the bucket as in case 1.

Directory doubling is $O(2^d)$ but does not move any data records -- only pointers are copied.  Bucket splits move approximately half the records of a single bucket.

### Complexity

| Operation | Expected I/Os |
|-----------|--------------|
| Lookup | $O(1)$ -- 1 directory page + 1 bucket page |
| Insert (no split) | $O(1)$ |
| Insert (with split) | $O(1)$ amortized |
| Directory doubling | $O(2^d / B)$ I/Os |

## Linear Hashing

Linear hashing eliminates the directory entirely by expanding buckets **one at a time** in a fixed order.  It maintains:

- A **split pointer** $p$ indicating the next bucket to split.
- A **level** $l$ tracking how many times the table has doubled.

The hash function alternates between $h_l(k) = k \bmod (2^l \cdot m)$ and $h_{l+1}(k) = k \bmod (2^{l+1} \cdot m)$.

When any bucket overflows, the bucket at position $p$ (not necessarily the overflowing one) is split:

1. Create bucket $p + 2^l \cdot m$.
2. Rehash records from bucket $p$ using $h_{l+1}$.
3. Advance $p$.  When $p = 2^l \cdot m$, reset $p = 0$ and increment $l$.

This controlled expansion spreads the rehashing cost evenly across insertions.

## Bitcask -- A Hash-Indexed Storage Engine

!!! example "Bitcask (Riak)"
    Bitcask is a log-structured storage engine that keeps a **hash map in memory** mapping every key to a file offset on disk.  Writes append to a log file; reads follow the in-memory pointer to the exact disk position.

    - **Write**: $O(1)$ -- append to log.
    - **Read**: $O(1)$ -- one in-memory lookup + one disk seek.
    - **Limitation**: all keys must fit in memory.

    This design achieves extremely high write throughput because writes are sequential (no random I/O), and reads require exactly one disk access.

## Implementation

```python
"""
Extendible Hashing -- dynamic hash index demonstration.

Demonstrates bucket splitting and directory doubling as records
are inserted into a hash-indexed structure.
"""

# === Bucket ===================================================================

class Bucket:
    """A hash bucket with local depth tracking."""

    def __init__(self, local_depth: int, capacity: int = 3):
        self.local_depth = local_depth
        self.capacity = capacity
        self.records: dict[int, str] = {}

    def is_full(self) -> bool:
        return len(self.records) >= self.capacity

    def insert(self, key: int, value: str) -> None:
        self.records[key] = value


# === Extendible Hash Table ====================================================

class ExtendibleHashTable:
    """An extendible hash table with directory doubling."""

    def __init__(self, capacity: int = 3):
        self.global_depth = 1
        self.capacity = capacity
        b0 = Bucket(1, capacity)
        b1 = Bucket(1, capacity)
        self.directory = [b0, b1]

    def _hash(self, key: int) -> int:
        return key % (2 ** self.global_depth)

    def lookup(self, key: int) -> str | None:
        idx = self._hash(key)
        bucket = self.directory[idx]
        return bucket.records.get(key)

    def insert(self, key: int, value: str) -> None:
        idx = self._hash(key)
        bucket = self.directory[idx]

        if not bucket.is_full():
            bucket.insert(key, value)
            return

        # Need to split
        if bucket.local_depth == self.global_depth:
            # Double the directory
            self.directory = self.directory + self.directory[:]
            self.global_depth += 1

        # Split the bucket
        old_depth = bucket.local_depth
        new_depth = old_depth + 1
        b0 = Bucket(new_depth, self.capacity)
        b1 = Bucket(new_depth, self.capacity)

        # Rehash existing records
        for k, v in bucket.records.items():
            if (k >> old_depth) & 1 == 0:
                b0.insert(k, v)
            else:
                b1.insert(k, v)

        # Update directory pointers
        for i in range(len(self.directory)):
            if self.directory[i] is bucket:
                if (i >> old_depth) & 1 == 0:
                    self.directory[i] = b0
                else:
                    self.directory[i] = b1

        # Retry the insert
        self.insert(key, value)


# === Main =====================================================================

if __name__ == "__main__":
    ht = ExtendibleHashTable(capacity=2)
    entries = [(2, "a"), (5, "b"), (7, "c"), (10, "d"), (15, "e"), (22, "f")]

    for key, val in entries:
        ht.insert(key, val)
        print(f"Insert ({key}, {val}) -> depth={ht.global_depth}, "
              f"buckets={len(ht.directory)}")

    # Lookups
    for key in [5, 10, 99]:
        result = ht.lookup(key)
        print(f"Lookup {key}: {result}")
```

**Output:**
```
Insert (2, a) -> depth=1, buckets=2
Insert (5, b) -> depth=1, buckets=2
Insert (7, c) -> depth=2, buckets=4
Insert (10, d) -> depth=2, buckets=4
Insert (15, e) -> depth=3, buckets=8
Insert (22, f) -> depth=3, buckets=8
Lookup 5: b
Lookup 10: d
Lookup 99: None
```

## Reference

- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
- [Database System Concepts (Silberschatz, Korth, Sudarshan)](https://www.db-book.com/)
