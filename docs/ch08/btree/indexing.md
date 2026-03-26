# Database Indexing with B-Trees

A database table may contain millions of rows stored on disk.  Without an index, answering a query like "find the row with id = 42" requires scanning every disk block — a **full table scan** with cost proportional to the table size.  An **index** is an auxiliary data structure that maps search keys to the disk locations of matching rows, enabling lookups in $O(\log n)$ disk accesses.  [B-trees](definition.md) and [B+ trees](b_plus.md) are the dominant index structures in virtually every relational database system because their high branching factor minimizes disk I/O.

## Why B-Trees for Indexing

The cost of a disk access (seek time + rotational latency + transfer time) is roughly $10^5$ times slower than a memory access.  An index must therefore minimize the **number of disk reads** per query.  B-trees achieve this by:

1. **High fanout:** a node with minimum degree $t$ stores up to $2t - 1$ keys, so a tree with $n$ keys has height $O(\log_t n)$.  With $t = 1000$, a billion keys fit in a tree of height 3.
2. **Node size = block size:** each B-tree node is sized to fill exactly one disk block, so reading a node requires exactly one disk access.
3. **Balanced structure:** all leaves are at the same depth, guaranteeing worst-case $O(\log_t n)$ accesses.

## Primary vs Secondary Indexes

A **primary index** (also called a clustered index) determines the physical order of rows on disk.  Each table can have at most one primary index because the rows can be sorted in only one way.

A **secondary index** (unclustered index) provides an alternative access path without controlling the physical row order.  It maps search keys to row identifiers (pointers to disk locations).

| Property | Primary index | Secondary index |
|----------|--------------|-----------------|
| Physical ordering | Matches index order | Independent |
| Range queries | Sequential disk reads | Random disk reads |
| Number per table | At most 1 | Unlimited |
| Leaf content | Rows or row pointers | Row pointers |

!!! tip "Clustered advantage for range queries"
    A range query on a primary index scans consecutive disk blocks because rows are physically sorted.  The same range query on a secondary index may require a separate random disk access for each matching row, making it much slower when the result set is large.

## Dense vs Sparse Indexes

A **dense index** contains an entry for every search key value in the table.  A **sparse index** contains entries for only some key values — typically one entry per disk block, pointing to the first record in that block.

$$
\text{Sparse index entries} = \frac{n}{B}
$$

where $n$ is the number of records and $B$ is the blocking factor (records per block).  Sparse indexes are smaller and faster to search but require the data to be sorted on the search key.

## Multi-Level Indexes

When a single-level index is itself too large to fit in memory, a second index can be built on top of the first.  This is exactly the structure of a B-tree: each level of the tree is an index on the level below.

The number of levels needed to index $n$ keys with blocking factor $f$ (keys per index block) is:

$$
\text{levels} = \lceil \log_f n \rceil
$$

Each level requires one disk access during a search, giving the familiar $O(\log_f n)$ bound.

## Index Operations Cost

For a B-tree index of minimum degree $t$ on $n$ keys:

| Operation | Disk accesses | In-memory work per node |
|-----------|---------------|------------------------|
| Point query | $O(\log_t n)$ | $O(\log t)$ binary search |
| Range query ($k$ results) | $O(\log_t n + k/B)$ | $O(k)$ |
| Insert | $O(\log_t n)$ | $O(t)$ key shifting |
| Delete | $O(\log_t n)$ | $O(t)$ key shifting |

The dominant cost is the number of disk accesses. The in-memory work within each node is fast relative to the disk I/O.

## Practical Considerations

!!! warning "Write amplification"
    Every insertion or deletion may modify multiple B-tree nodes (due to splits, merges, or rotations).  Each modified node must be written back to disk.  This **write amplification** is a significant concern for write-heavy workloads and is one motivation for log-structured merge trees (LSM-trees) used in systems like LevelDB and RocksDB.

**Buffer pool caching.**  Database systems maintain a buffer pool that caches frequently accessed disk blocks in memory.  The root and upper levels of a B-tree index are almost always cached, so most searches require only 1–2 actual disk reads even for trees with height 4–5.

**Bulk loading.**  Building a B-tree index on an existing table by repeated insertion is inefficient because each insert requires $O(\log_t n)$ random disk accesses.  Instead, databases sort the keys first and build the tree bottom-up, level by level, achieving $O(n / B)$ sequential disk accesses.

## Reference

- Ramakrishnan, R., & Gehrke, J. (2003). *Database Management Systems* (3rd ed.), Chapters 9–10. McGraw-Hill.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 18. MIT Press.
- Graefe, G. (2011). Modern B-tree techniques. *Foundations and Trends in Databases*, 3(4), 203–402.
