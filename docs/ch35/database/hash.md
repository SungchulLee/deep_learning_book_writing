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

## Exercises

**Exercise 1.**
Explain why hash indexes do not support range queries. What structural property of hash functions prevents ordered access?

??? success "Solution to Exercise 1"
    Hash functions are designed to distribute keys uniformly and pseudorandomly across buckets. Two keys that are close in value (e.g., 100 and 101) map to unrelated bucket positions. This destroys any ordering: a range query `WHERE id BETWEEN 100 AND 200` would need to check every bucket because the target keys are scattered. B-tree indexes maintain sorted order, so a range scan starts at the lower bound and sequentially reads pages until the upper bound. Hash indexes provide $O(1)$ point lookups but $O(n)$ range scans (equivalent to a full table scan). This is why databases default to B-tree indexes and use hash indexes only when the workload is exclusively exact-match lookups. $\square$

---

**Exercise 2.**
Describe extendible hashing and how it handles bucket overflow without rehashing the entire table.

??? success "Solution to Exercise 2"
    Extendible hashing uses a directory of $2^d$ pointers (where $d$ is the global depth) mapping hash prefixes to buckets. Each bucket has a local depth $d_b \le d$. On overflow: if $d_b < d$, split the bucket into two, increment $d_b$, and update the relevant directory entries to point to the new buckets. If $d_b = d$, double the directory ($d \to d + 1$), copying all pointers, then split the bucket. Crucially, only one bucket is split and at most one directory doubling occurs -- no existing records are moved except those in the overflowing bucket. This is $O(B)$ work where $B$ is the bucket size, compared to $O(n)$ for full rehashing. The directory fits in memory (it is small -- $2^d$ pointers), so only the split bucket requires disk I/O. $\square$

---

**Exercise 3.**
A hash index uses linear probing with a load factor of $\alpha$. Derive the expected number of probes for a successful lookup and an unsuccessful lookup.

??? success "Solution to Exercise 3"
    Under the uniform hashing assumption, Knuth's analysis gives: expected probes for a **successful** lookup: $\frac{1}{2}(1 + \frac{1}{1 - \alpha})$. Expected probes for an **unsuccessful** lookup: $\frac{1}{2}(1 + \frac{1}{(1-\alpha)^2})$. At $\alpha = 0.5$: successful $= 1.5$ probes, unsuccessful $= 2.5$ probes. At $\alpha = 0.75$: successful $= 2.5$, unsuccessful $= 8.5$. At $\alpha = 0.9$: successful $= 5.5$, unsuccessful $= 50.5$. The sharp degradation above $\alpha = 0.75$ is why hash tables are typically resized at 70--75% load. For database hash indexes, each probe is a disk I/O, making the cost even more critical than in-memory hash tables. $\square$

---

**Exercise 4.**
Compare static hash indexing, extendible hashing, and linear hashing for a database that grows from 1000 to 10 million records. Which approach handles growth best?

??? success "Solution to Exercise 4"
    **Static hashing**: fixed number of buckets chosen at creation. As records grow, buckets overflow, creating long chains. Eventually requires a full rebuild (costly $O(n)$ operation that locks the table). Poor for growing databases. **Extendible hashing**: directory doubles when needed ($O(2^d)$ to double), but only one bucket splits at a time. Handles growth smoothly. Drawback: directory can become very large ($2^d$ entries) if data distribution is skewed. **Linear hashing**: splits buckets one at a time in round-robin order, triggered when the average load exceeds a threshold. No directory needed -- the number of buckets grows linearly. Overflow chains are temporary (resolved when the overflowing bucket is eventually split). Linear hashing handles growth best for large-scale databases because it avoids both full rebuilds and large directories, with predictable $O(1)$ amortized cost per split. $\square$

---

**Exercise 5.**
Bitcask (used in Riak) stores an in-memory hash index mapping every key to a file offset. Analyze its space requirements for $10^8$ keys of average length 20 bytes and discuss why it is effective for write-heavy workloads.

??? success "Solution to Exercise 5"
    Each hash index entry stores: key (20 bytes average), file ID (4 bytes), offset (8 bytes), value size (4 bytes), timestamp (4 bytes). Total per entry: $\approx 40$ bytes. For $10^8$ keys: $40 \times 10^8 = 4$ GB of RAM for the index. This is feasible on modern servers but limits the number of keys. Effectiveness for writes: Bitcask writes are always sequential appends to the active data file -- no random I/O, no in-place updates. This maximizes disk throughput (sequential writes are 100x faster than random on HDD). Reads are always one random I/O (look up the offset in the in-memory hash map, then seek to that offset). The tradeoff: the entire key set must fit in RAM, and range queries are not supported. For use cases with a bounded key count and high write throughput (session stores, caches), Bitcask is highly effective. $\square$
