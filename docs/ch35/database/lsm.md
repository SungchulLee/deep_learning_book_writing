# LSM Trees

B-tree indexes optimize for read-heavy workloads by keeping data sorted on disk, but every update requires a random write to the page containing the affected key.  For write-heavy workloads -- time-series databases, logging systems, message queues -- these random writes become a bottleneck.  **Log-Structured Merge Trees (LSM trees)** trade read performance for write performance by converting random writes into sequential writes, achieving dramatically higher write throughput.  LSM trees power modern storage engines including LevelDB, RocksDB, Cassandra, and HBase.

## Core Architecture

An LSM tree organizes data into multiple **levels**, each an order of magnitude larger than the previous:

1. **Memtable (Level 0 in memory)**: An in-memory balanced tree (red-black tree or skip list) that absorbs all incoming writes.
2. **SSTables (Level 1, 2, ..., $L$ on disk)**: Sorted String Tables -- immutable, sorted files on disk.

### Write Path

1. Write the key-value pair to a **write-ahead log (WAL)** for crash recovery.
2. Insert into the in-memory **memtable**.
3. When the memtable reaches its size threshold $M$, flush it to disk as a new SSTable at Level 1.

All writes go to memory first, so write latency is $O(\log M)$ -- a red-black tree insertion.  The flush to disk is a single sequential write.

### Read Path

To read key $k$:

1. Check the memtable.
2. Check each level's SSTables, from newest to oldest.
3. Return the first match found (most recent version).

Without optimization, a read might check $O(L)$ SSTables, where $L$ is the number of levels.  **Bloom filters** attached to each SSTable quickly eliminate SSTables that definitely do not contain $k$, reducing most reads to 1--2 disk accesses.

## Compaction

As SSTables accumulate, they must be merged to bound read amplification and reclaim space from deleted/overwritten keys.  Compaction merges multiple SSTables into fewer, larger ones.

### Size-Tiered Compaction

- Group SSTables of similar size together.
- When a level accumulates $T$ SSTables, merge them all into one SSTable at the next level.
- Simple but can cause temporary space amplification (up to 2x).

### Leveled Compaction

- Each level $i$ has a size limit: $|L_i| \leq T^i \cdot |L_1|$ where $T$ is the size ratio.
- When Level $i$ exceeds its limit, pick one SSTable and merge it into Level $i + 1$.
- Provides bounded space amplification at the cost of higher write amplification.

## Amplification Analysis

LSM tree performance is characterized by three amplification factors:

| Factor | Definition | Size-tiered | Leveled |
|--------|-----------|-------------|---------|
| Write amplification | Bytes written to disk / bytes written by user | $O(T \cdot L)$ | $O(T \cdot L)$ |
| Read amplification | Disk reads per user read | $O(T \cdot L)$ | $O(L)$ |
| Space amplification | Total disk usage / logical data size | $O(T)$ | $O(1 + 1/T)$ |

With $T = 10$ (typical), $L = \log_T(n/M) \approx 3\text{--}5$ levels, and $n$ total keys:

- **Write amplification**: each byte is rewritten roughly $T \cdot L \approx 30\text{--}50$ times across its lifetime.
- **Read amplification (leveled)**: at most $L$ SSTables checked, often just 1--2 with Bloom filters.

## B-Tree vs LSM Tree

| Property | B-Tree | LSM Tree |
|----------|--------|----------|
| Write pattern | Random | Sequential |
| Write throughput | Moderate | High |
| Read latency | Low (1 seek) | Moderate ($O(L)$ seeks) |
| Range scans | Efficient (sorted leaves) | Efficient (sorted SSTables) |
| Space amplification | Low | Low (leveled) to moderate (size-tiered) |
| Write amplification | Moderate | Higher |

!!! tip "Rule of thumb"
    Use B-trees when reads dominate, LSM trees when writes dominate.  Many modern systems offer both: PostgreSQL uses B-trees, while RocksDB (used as a storage backend by CockroachDB) uses LSM trees.

## Bloom Filters for Read Optimization

Each SSTable maintains a Bloom filter -- a space-efficient probabilistic data structure that answers "is key $k$ possibly in this SSTable?" with no false negatives but a tunable false positive rate $\epsilon$.

With $m$ bits per key, the false positive rate is approximately

$$
\epsilon \approx \left(1 - e^{-km/n}\right)^k
$$

where $k$ is the number of hash functions and $n$ is the number of keys.  Setting $k = (m/n) \ln 2$ minimizes $\epsilon$.  At 10 bits per key, $\epsilon \approx 1\%$, meaning only 1% of unnecessary disk reads occur.

## Implementation

```python
"""
LSM Tree -- simplified in-memory simulation.

Demonstrates the core LSM write path (memtable + flush to SSTables)
and read path (check memtable, then SSTables newest-to-oldest).
"""

# === SSTable (immutable sorted file) ==========================================

class SSTable:
    """An immutable sorted string table."""

    def __init__(self, data: dict[str, str]):
        self.data = dict(sorted(data.items()))

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def __len__(self) -> int:
        return len(self.data)


# === LSM Tree =================================================================

class LSMTree:
    """A simplified LSM tree with memtable and SSTable levels."""

    def __init__(self, memtable_limit: int = 4):
        self.memtable_limit = memtable_limit
        self.memtable: dict[str, str] = {}
        self.sstables: list[SSTable] = []  # newest first

    def put(self, key: str, value: str) -> None:
        """Write a key-value pair."""
        self.memtable[key] = value
        if len(self.memtable) >= self.memtable_limit:
            self._flush()

    def get(self, key: str) -> str | None:
        """Read a key, checking memtable then SSTables."""
        if key in self.memtable:
            return self.memtable[key]
        for sst in self.sstables:
            val = sst.get(key)
            if val is not None:
                return val
        return None

    def _flush(self) -> None:
        """Flush memtable to a new SSTable."""
        sst = SSTable(self.memtable)
        self.sstables.insert(0, sst)  # newest first
        self.memtable = {}

    def delete(self, key: str) -> None:
        """Delete by inserting a tombstone."""
        self.put(key, "__TOMBSTONE__")


# === Main =====================================================================

if __name__ == "__main__":
    lsm = LSMTree(memtable_limit=3)

    # Write some data
    writes = [("a", "1"), ("b", "2"), ("c", "3"),
              ("d", "4"), ("e", "5"), ("a", "updated")]
    for k, v in writes:
        lsm.put(k, v)
        print(f"PUT({k}, {v}) -> memtable={dict(lsm.memtable)}, "
              f"sstables={len(lsm.sstables)}")

    # Read data
    for key in ["a", "b", "d", "z"]:
        val = lsm.get(key)
        print(f"GET({key}) = {val}")
```

**Output:**
```
PUT(a, 1) -> memtable={'a': '1'}, sstables=0
PUT(b, 2) -> memtable={'a': '1', 'b': '2'}, sstables=0
PUT(c, 3) -> memtable={}, sstables=1
PUT(d, 4) -> memtable={'d': '4'}, sstables=1
PUT(e, 5) -> memtable={'d': '4', 'e': '5'}, sstables=1
PUT(a, updated) -> memtable={}, sstables=2
GET(a) = updated
GET(b) = 2
GET(d) = 4
GET(z) = None
```

## Reference

- [The Log-Structured Merge-Tree (O'Neil et al., 1996)](https://www.cs.umb.edu/~poneil/lsmtree.pdf)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)

## Exercises

**Exercise 1.**
Describe the write path in an LSM tree: from the client's insert to the data reaching the lowest level on disk.

??? success "Solution to Exercise 1"
    (1) The insert is written to a write-ahead log (WAL) for durability. (2) The key-value pair is added to the in-memory **memtable** (typically a balanced BST or skip list). (3) When the memtable exceeds a size threshold, it is frozen (becomes immutable) and a new empty memtable is created. (4) The frozen memtable is flushed to disk as a sorted SSTable (Sorted String Table) at **level 0**. (5) When level 0 accumulates too many SSTables, **compaction** merges overlapping SSTables into larger, non-overlapping SSTables at level 1. (6) When level 1 exceeds its size limit, its SSTables are merged into level 2, and so on. Each level is typically 10x larger than the previous. Data eventually reaches the lowest level after multiple compaction rounds. All disk writes are sequential (append or bulk-write new SSTables), which is the key to LSM's write performance. $\square$

---

**Exercise 2.**
Explain read amplification, write amplification, and space amplification in LSM trees. How does the leveled compaction strategy affect each?

??? success "Solution to Exercise 2"
    **Read amplification**: the number of SSTables checked per read. In the worst case, a key might not exist and the reader checks one SSTable per level plus Bloom filters. With $L$ levels: up to $L$ reads (reduced by Bloom filters to $\approx 1$--$2$ actual I/Os). **Write amplification**: the total bytes written to disk per byte of user data. Each compaction rewrites data. With a size ratio of 10 between levels and $L$ levels, data is rewritten $\sim 10$ times per level, giving write amplification $\approx 10 \times L$ (e.g., $\approx 30$--$50$ for typical configurations). **Space amplification**: the ratio of total disk space used to the logical data size. During compaction, old and new SSTables coexist temporarily. Leveled compaction keeps space amplification near 1.1x (each level has non-overlapping SSTables). Size-tiered compaction can have $2$x space amplification during major compaction but has lower write amplification. $\square$

---

**Exercise 3.**
A Bloom filter is attached to each SSTable. With a false positive rate of $1\%$ and 5 levels, what is the expected number of unnecessary disk reads per point query?

??? success "Solution to Exercise 3"
    For a point query on a key that exists: the Bloom filter at the correct SSTable always returns true (no false negatives). Bloom filters at other SSTables return false positive with probability $0.01$. Expected unnecessary reads: $(5 - 1) \times 0.01 = 0.04$. For a key that does not exist: all 5 Bloom filters are checked. Expected false positives: $5 \times 0.01 = 0.05$ unnecessary reads. With optimized per-level Bloom filter sizing (higher accuracy for deeper levels, which are larger), this can be reduced further. In practice, point queries on existing keys require $\approx 1.04$ disk reads, making LSM trees competitive with B-trees for read-heavy workloads when Bloom filters are well-tuned. $\square$

---

**Exercise 4.**
Compare leveled compaction and size-tiered compaction. Which is better for write-heavy workloads and which for read-heavy workloads?

??? success "Solution to Exercise 4"
    **Size-tiered compaction**: SSTables at each level are not required to be non-overlapping. When enough similar-sized SSTables accumulate, they are merged into one larger SSTable. Write amplification is lower ($\approx L$ vs. $10L$ for leveled) because data is merged less frequently. Read amplification is higher because multiple overlapping SSTables per level must be checked. Space amplification can be $2$x during compaction. Better for write-heavy workloads (e.g., time-series ingestion). **Leveled compaction**: each level has non-overlapping SSTables. Compaction picks one SSTable from level $i$ and merges it with overlapping SSTables in level $i+1$. Read amplification is lower (one SSTable per level to check). Write amplification is higher. Space amplification is lower ($\sim$1.1x). Better for read-heavy and balanced workloads. $\square$

---

**Exercise 5.**
RocksDB supports a feature called "prefix Bloom filters." Explain how this differs from a standard Bloom filter and why it is useful for scan queries with a known key prefix.

??? success "Solution to Exercise 5"
    A standard Bloom filter tests membership of exact keys. A prefix Bloom filter hashes only a prefix of the key (e.g., the first 8 bytes) and tests whether any key with that prefix exists in the SSTable. For a scan query like "find all keys starting with 'user:123:'," the prefix Bloom filter can quickly determine whether an SSTable contains any relevant keys. If the filter returns negative, the entire SSTable is skipped. Without prefix Bloom filters, the reader would need to check the SSTable's min/max key range (which is coarse) or perform a seek (which requires reading index blocks). Prefix Bloom filters provide fine-grained filtering for scan queries at the cost of higher false positive rates (multiple keys share a prefix) and inability to skip within a prefix. They are especially useful in key-value stores where keys are hierarchically structured (e.g., `tenant:table:row:column`). $\square$
