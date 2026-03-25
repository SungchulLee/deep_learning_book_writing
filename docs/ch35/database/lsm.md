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
