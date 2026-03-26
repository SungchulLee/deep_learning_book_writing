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
