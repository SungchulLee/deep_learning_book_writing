# File System Trees

Operating systems organize files on disk using tree-structured data to balance two competing demands: fast lookup by name (directory traversal) and efficient sequential access to file contents (block allocation). The choice of data structure determines how quickly files can be created, opened, and extended, especially as the file system scales to millions of entries.

## Directory Trees

A file system directory maps names to file metadata (inodes). The simplest directory is a linear list of (name, inode) pairs, but lookups take $O(n)$ time. Modern file systems use tree structures:

- **B-trees / B+ trees**: ext4, NTFS, and HFS+ store directory entries in balanced B+ trees keyed by filename hash or name. Lookup, insert, and delete take $O(\log_B n)$ disk I/Os where $B$ is the branching factor (typically hundreds).
- **Hash trees (HTree)**: ext3/ext4 use hash-based directory indexing with a B-tree structure, achieving $O(1)$ expected lookup for typical directories.

For a B+ tree with branching factor $B$ and $n$ entries:

$$
\text{I/O per lookup} = O(\log_B n)
$$

With $B = 256$ and $n = 10^6$ entries, this is about $\log_{256}(10^6) \approx 2.5$ disk reads -- effectively constant.

## Inode Structure

An **inode** stores file metadata and block pointers:

- **Direct pointers**: $d$ pointers to data blocks (typically $d = 12$).
- **Single indirect**: pointer to a block of $B$ pointers.
- **Double indirect**: pointer to a block of $B$ pointers, each pointing to a block of $B$ pointers.
- **Triple indirect**: one more level of indirection.

Maximum file size with block size $b$ and $B = b / \text{pointer\_size}$ pointers per block:

$$
\text{max size} = (d + B + B^2 + B^3) \times b
$$

With $b = 4\text{KB}$, $d = 12$, $B = 1024$: max size $\approx 4\text{TB}$.

## Extent-Based Allocation

Modern file systems (ext4, XFS, NTFS) replace per-block pointers with **extents** -- contiguous ranges described by (start\_block, length) pairs. An extent tree (B+ tree of extents) supports:

$$
\text{I/O per random read} = O(\log_B E)
$$

where $E$ is the number of extents. For mostly sequential files, $E$ is small (often 1), making access nearly $O(1)$.

!!! tip "B-trees and disk I/O"
    B-trees are the natural choice for disk-based data structures because their high branching factor ($B = 100$--$1000$) matches the disk block size, minimizing the number of I/O operations. A tree with branching factor 256 and height 3 can index billions of entries.

## Implementation

```python
"""
File System Trees -- inode block lookup and B+ tree directory.

Demonstrates how multi-level inode indirection and B+ tree
directory indexing work for file systems.
"""

from __future__ import annotations
import math


# === Inode Block Lookup =======================================================

class InodeCalculator:
    """Calculate which block pointer to follow for a given file offset."""

    def __init__(self, block_size: int = 4096, pointer_size: int = 4,
                 direct_count: int = 12):
        self.block_size = block_size
        self.pointers_per_block = block_size // pointer_size
        self.direct = direct_count
        B = self.pointers_per_block
        self.single_max = self.direct + B
        self.double_max = self.single_max + B * B
        self.triple_max = self.double_max + B * B * B

    def max_file_size(self) -> int:
        """Maximum file size in bytes."""
        return self.triple_max * self.block_size

    def lookup_depth(self, block_number: int) -> str:
        """Determine the indirection level for a logical block number."""
        if block_number < self.direct:
            return "direct"
        elif block_number < self.single_max:
            return "single indirect"
        elif block_number < self.double_max:
            return "double indirect"
        elif block_number < self.triple_max:
            return "triple indirect"
        else:
            return "beyond max file size"


# === Simple B+ Tree Directory =================================================

class BPlusDirectory:
    """Simplified B+ tree directory index (in-memory simulation)."""

    def __init__(self, order: int = 4):
        self.order = order
        self.entries: dict[str, int] = {}  # name -> inode

    def insert(self, name: str, inode: int) -> None:
        """Add a directory entry."""
        self.entries[name] = inode

    def lookup(self, name: str) -> int | None:
        """Look up an inode by filename."""
        return self.entries.get(name)

    def list_entries(self) -> list[tuple[str, int]]:
        """List all entries sorted by name."""
        return sorted(self.entries.items())

    def io_cost(self, n: int) -> float:
        """Estimated disk I/Os for lookup in a B+ tree with n entries."""
        if n <= 1:
            return 1.0
        return math.ceil(math.log(n) / math.log(self.order))


# === Main =====================================================================

if __name__ == "__main__":
    # Inode structure analysis
    inode = InodeCalculator()
    print("Inode structure (4KB blocks, 4-byte pointers):")
    print(f"  Pointers per block: {inode.pointers_per_block}")
    print(f"  Max file size: {inode.max_file_size() / (1024**4):.1f} TB")

    for block in [0, 11, 12, 1035, 1036, 100000]:
        print(f"  Block {block:>7}: {inode.lookup_depth(block)}")

    # Directory B+ tree
    print("\nB+ tree directory (order=256):")
    directory = BPlusDirectory(order=256)
    for i, name in enumerate(["readme.txt", "main.py", "data.csv", "config.yml"]):
        directory.insert(name, inode=100 + i)

    for name in ["main.py", "missing.txt"]:
        result = directory.lookup(name)
        print(f"  lookup('{name}'): inode={result}")

    for n in [100, 10_000, 1_000_000, 1_000_000_000]:
        ios = directory.io_cost(n)
        print(f"  {n:>13,} entries -> ~{ios:.0f} disk I/Os")
```

**Output:**

```
Inode structure (4KB blocks, 4-byte pointers):
  Pointers per block: 1024
  Max file size: 4.0 TB

  Block       0: direct
  Block      11: direct
  Block      12: single indirect
  Block    1035: single indirect
  Block    1036: double indirect
  Block  100000: double indirect

B+ tree directory (order=256):
  lookup('main.py'): inode=101
  lookup('missing.txt'): inode=None
            100 entries -> ~1 disk I/Os
         10,000 entries -> ~2 disk I/Os
      1,000,000 entries -> ~3 disk I/Os
  1,000,000,000 entries -> ~4 disk I/Os
```

The inode analysis shows how different block numbers map to indirection levels. The B+ tree directory demonstrates that even a billion entries require only about 4 disk reads -- the power of high-branching-factor trees for disk-based indexing.

## Reference

- Silberschatz, A., Galvin, P.B., and Gagne, G. *Operating System Concepts*. Wiley
- Cormen, T.H., Leiserson, C.E., Rivest, R.L., and Stein, C. *Introduction to Algorithms*. MIT Press

## Exercises

**Exercise 1.**
Compare inode-based file systems (ext4) with B-tree-based file systems (Btrfs) for directory lookup performance on a directory with 1 million files.

??? success "Solution to Exercise 1"
    **ext4 with htree**: directories use a hash tree (B-tree variant) indexed by filename hash. Lookup: hash the filename, traverse the htree in $O(\log n)$ I/Os where $n$ is the number of entries. For 1 million files, the htree has $\sim$3 levels, so lookup takes $\sim$3 disk reads. **Btrfs**: directories are stored as items in the global B-tree, keyed by (directory inode, hash(filename)). Lookup: one B-tree search in $O(\log N)$ where $N$ is the total number of items in the filesystem. For a typical Btrfs tree with branching factor $\sim$100 and millions of items, this is $\sim$3--4 levels. Performance is similar for single lookups. Btrfs has an advantage for operations spanning multiple directories (copy-on-write snapshots, atomic renames across directories) because everything is in one B-tree. ext4 is simpler and has lower per-I/O overhead. $\square$

---

**Exercise 2.**
Explain how ext4's extent tree replaces the traditional indirect block scheme. What are the performance benefits for large files?

??? success "Solution to Exercise 2"
    Traditional indirect blocks: a file's inode has 12 direct pointers, plus single/double/triple indirect block pointers. For a 1 GB file with 4 KB blocks, $\sim$256K block addresses must be stored across many indirect blocks, requiring multiple I/Os to traverse. Ext4's extent tree: each extent records (logical block, physical start block, length). A contiguous 1 GB file needs just one extent: (0, physical_start, 262144). The inode holds up to 4 extents directly; more extents use a B-tree of extent nodes. Benefits: (1) a contiguous file needs 1 extent vs. 256K block pointers -- dramatically less metadata; (2) sequential reads can issue large I/O requests (the OS knows the file is contiguous); (3) the extent tree is shallow (3--4 levels covers petabytes). For fragmented files, extent trees degrade but are still better than indirect blocks because each extent covers multiple blocks. $\square$

---

**Exercise 3.**
A file system must support $O(1)$ time allocation of a free disk block. Describe how a bitmap-based free space manager achieves this and analyze the space overhead.

??? success "Solution to Exercise 3"
    A bitmap allocates one bit per disk block: 1 = used, 0 = free. For a 1 TB disk with 4 KB blocks: $1 \text{ TB} / 4 \text{ KB} = 2.5 \times 10^8$ blocks, requiring $2.5 \times 10^8$ bits $= 31.25$ MB for the bitmap. Space overhead: $31.25 / (10^6) \approx 0.003\%$. For $O(1)$ allocation: maintain a "hint" pointer to the last allocated position. Search forward from the hint for a 0 bit. On average, the search scans $1/(1-\alpha)$ bits where $\alpha$ is the utilization. At 90% full, this is 10 bits -- effectively $O(1)$. For guaranteed $O(1)$: maintain a free-block stack or linked list. Bitmaps are preferred because they support contiguous allocation (find $k$ consecutive 0 bits for extent allocation) and are compact. $\square$

---

**Exercise 4.**
Explain journaling in file systems (ext4, NTFS). How does it prevent corruption after a crash, and what is the performance cost?

??? success "Solution to Exercise 4"
    Without journaling, a crash during a multi-step file operation (e.g., creating a file requires updating the directory, inode table, and data blocks) can leave the file system in an inconsistent state. Journaling writes a **log** of pending changes before applying them to the main file system. On crash recovery, the journal is replayed: committed transactions are applied, and uncommitted ones are discarded. Three modes: (1) **Full journal**: logs both metadata and data. Safest but slowest (data written twice). (2) **Ordered journal** (ext4 default): logs metadata only but ensures data blocks are written before the metadata journal entry is committed. Good balance. (3) **Writeback journal**: logs metadata only with no ordering guarantee. Fastest but data may be lost on crash. Performance cost: journal writes are sequential (fast), but they add I/O overhead. For metadata-only journaling, the cost is $\sim$5--10% of write throughput. Full journaling costs $\sim$30--50% due to double-writing all data. $\square$

---

**Exercise 5.**
A log-structured file system (LFS) converts all writes to sequential appends. Explain the write path, the garbage collection challenge, and why LFS is well-suited for SSDs.

??? success "Solution to Exercise 5"
    **Write path**: all modifications (data and metadata) are buffered in memory and periodically written as a contiguous segment to the end of the log. The segment contains data blocks, inode updates, and a segment summary. An inode map (also stored in the log) tracks the latest location of each inode. Writes are always sequential -- never in-place. **Garbage collection**: as files are updated, old versions of blocks become dead (orphaned in earlier segments). A cleaner process identifies segments with many dead blocks, copies the live blocks to the current segment, and frees the old segment. This adds write amplification. **SSD suitability**: SSDs cannot overwrite -- they must erase blocks before rewriting. LFS's append-only pattern aligns with this constraint, avoiding the costly erase-before-write cycle. LFS also distributes writes evenly (wear leveling). The garbage collector's segment cleaning mirrors SSD firmware's own garbage collection, potentially reducing redundant work. $\square$
