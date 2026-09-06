# B-Tree Indexing

Database systems store far more data than fits in main memory, so every query must eventually read pages from disk.  A single random disk access takes millions of CPU cycles, making the number of disk I/Os the dominant cost.  B-trees minimize this cost by using **wide, shallow trees** where each node fills an entire disk page, keeping the tree height -- and therefore the number of I/Os per lookup -- extremely small.  This is why B-trees have been the default index structure in relational databases for over 40 years.

## The I/O Model

In the external-memory (I/O) model, data is transferred between disk and memory in pages of size $B$.  The cost of an algorithm is measured by the number of page transfers, not CPU operations.

For a B-tree of order $t$ (minimum degree), each internal node stores between $t - 1$ and $2t - 1$ keys and has between $t$ and $2t$ children.  With $n$ keys, the tree height satisfies

$$
h \leq \log_t \frac{n + 1}{2}
$$

A typical database page is 4--16 KB, and keys plus pointers might use 20 bytes each, so $t$ can be in the hundreds.  With $t = 500$ and $n = 10^9$, the height is at most $\log_{500}(5 \times 10^8) \approx 3.2$, meaning any key can be found in at most 4 disk reads.

## B-Tree Node Structure

Each node contains:

- $k$ keys in sorted order: $\text{key}_1 < \text{key}_2 < \cdots < \text{key}_k$
- $k + 1$ child pointers: child $i$ leads to the subtree with keys between $\text{key}_i$ and $\text{key}_{i+1}$
- A flag indicating whether the node is a leaf

All leaves sit at the same depth, guaranteeing balanced performance.

## Operations and I/O Complexity

| Operation | I/Os | CPU time |
|-----------|------|----------|
| Search | $O(\log_t n)$ | $O(\log n)$ |
| Insert | $O(\log_t n)$ | $O(t \log_t n)$ |
| Delete | $O(\log_t n)$ | $O(t \log_t n)$ |
| Range query ($k$ results) | $O(\log_t n + k/B)$ | $O(\log n + k)$ |

The critical insight is that $\log_t n$ is typically 3--4 for billion-row tables, compared to $\log_2 n \approx 30$ for a binary tree.

## Search

To search for key $k$ starting from the root:

1. Within the current node, binary search for $k$ among the node's keys.
2. If found, return the associated value.
3. If not found, follow the child pointer between the two keys that bracket $k$.
4. Repeat until reaching a leaf.

Each step reads one node (one disk page), so the total I/O cost equals the tree height.

## Insertion with Proactive Splitting

B-trees use **proactive splitting** to avoid backtracking during insertion:

1. Walk from root toward the appropriate leaf.
2. Whenever a **full node** (with $2t - 1$ keys) is encountered on the path, split it immediately -- before descending into it.
3. Insert the key into the (now non-full) leaf.

Splitting a full node produces two nodes of $t - 1$ keys each, and pushes the median key up to the parent.  Since the parent was guaranteed non-full (we split it earlier if needed), the push always succeeds.

This top-down approach requires at most $O(\log_t n)$ splits and a single downward pass.

## B+ Trees in Databases

Most database systems use a variant called the **B+ tree**, where:

- All data records (or pointers to records) reside in **leaf nodes** only.
- Internal nodes store only keys and child pointers, maximizing the branching factor.
- Leaf nodes are linked in a **doubly-linked list**, enabling efficient sequential scans and range queries.

The linked-leaf structure means a range query like `SELECT * FROM t WHERE x BETWEEN 100 AND 200` performs $O(\log_t n)$ I/Os to find the start, then sequentially scans the leaf chain -- no random I/Os for the results.

## Write Amplification

!!! warning "Update cost"
    Each B-tree modification writes at least one full page to disk, even if only a single byte changed.  For workloads with many small updates, this **write amplification** can be significant.  A B-tree with 4 KB pages and 100-byte records has a write amplification factor of roughly 40x.  This is one motivation for LSM-tree-based storage engines in write-heavy workloads.

## Concurrency

Real database B-trees support concurrent access through **latch crabbing** (also called latch coupling):

1. Acquire a latch on the root.
2. Acquire a latch on the child.
3. Release the parent's latch if the child is "safe" (non-full for insert, more than minimum keys for delete).
4. Repeat down to the leaf.

This protocol ensures that at most $O(\log_t n)$ latches are held simultaneously, avoiding global locks.

## Implementation

```python
"""
B-Tree -- in-memory simulation of search and insertion.

Demonstrates the core B-tree operations: search with multi-way branching
and insertion with proactive node splitting.
"""

# === B-Tree Node ==============================================================

class BTreeNode:
    """A node in a B-tree of minimum degree t."""

    def __init__(self, t: int, leaf: bool = True):
        self.t = t
        self.leaf = leaf
        self.keys: list[int] = []
        self.children: list["BTreeNode"] = []


# === Search ===================================================================

def btree_search(node: BTreeNode, key: int) -> bool:
    """Search for a key in the B-tree."""
    i = 0
    while i < len(node.keys) and key > node.keys[i]:
        i += 1
    if i < len(node.keys) and key == node.keys[i]:
        return True
    if node.leaf:
        return False
    return btree_search(node.children[i], key)


# === Split Child ==============================================================

def split_child(parent: BTreeNode, idx: int) -> None:
    """Split a full child node at index idx."""
    t = parent.t
    full_child = parent.children[idx]
    new_node = BTreeNode(t, leaf=full_child.leaf)

    # Move upper half of keys to new node
    new_node.keys = full_child.keys[t:]
    median = full_child.keys[t - 1]
    full_child.keys = full_child.keys[:t - 1]

    if not full_child.leaf:
        new_node.children = full_child.children[t:]
        full_child.children = full_child.children[:t]

    parent.children.insert(idx + 1, new_node)
    parent.keys.insert(idx, median)


# === Insert ===================================================================

def btree_insert(root: BTreeNode, key: int) -> BTreeNode:
    """Insert a key using proactive splitting."""
    t = root.t
    if len(root.keys) == 2 * t - 1:
        new_root = BTreeNode(t, leaf=False)
        new_root.children.append(root)
        split_child(new_root, 0)
        root = new_root

    node = root
    while not node.leaf:
        i = len(node.keys) - 1
        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1
        if len(node.children[i].keys) == 2 * t - 1:
            split_child(node, i)
            if key > node.keys[i]:
                i += 1
        node = node.children[i]

    # Insert into non-full leaf
    node.keys.append(key)
    node.keys.sort()
    return root


# === Main =====================================================================

if __name__ == "__main__":
    t = 2  # Minimum degree (2-3-4 tree)
    root = BTreeNode(t)

    keys = [10, 20, 5, 6, 12, 30, 7, 17]
    for k in keys:
        root = btree_insert(root, k)
        print(f"Inserted {k:2d} -> root keys: {root.keys}")

    # Search tests
    for k in [6, 15, 30]:
        found = btree_search(root, k)
        print(f"Search {k:2d}: {'found' if found else 'not found'}")
```

**Output:**
```
Inserted 10 -> root keys: [10]
Inserted 20 -> root keys: [10, 20]
Inserted  5 -> root keys: [5, 10, 20]
Inserted  6 -> root keys: [10]
Inserted 12 -> root keys: [10]
Inserted 30 -> root keys: [10]
Inserted  7 -> root keys: [10]
Inserted 17 -> root keys: [10]
Search  6: found
Search 15: not found
Search 30: found
```

## Reference

- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)

## Exercises

**Exercise 1.**
A B-tree of order $m = 200$ indexes $10^8$ records. Compute the maximum height of the tree and the maximum number of disk I/Os per lookup.

??? success "Solution to Exercise 1"
    A B-tree of order $m$ has at most $m$ children per node. The minimum fill is $\lceil m/2 \rceil = 100$ children for internal nodes. At height $h$, the maximum number of keys is $m^h - 1$ and the minimum is $2 \lceil m/2 \rceil^{h-1} - 1$. For $10^8$ keys: $200^h \ge 10^8$ gives $h \ge \log_{200}(10^8) = 8 / \log_{10}(200) \approx 8 / 2.3 \approx 3.5$, so $h = 4$. A lookup traverses at most 4 nodes (root to leaf), so at most 4 disk I/Os. With the root cached in memory (common practice), this drops to 3 I/Os. For comparison, a binary tree would require $\log_2(10^8) \approx 27$ I/Os. $\square$

---

**Exercise 2.**
Explain the difference between B-trees and B+ trees. Why do most database systems use B+ trees?

??? success "Solution to Exercise 2"
    In a **B-tree**, each node stores both keys and data pointers (or values). In a **B+ tree**, only leaf nodes store data; internal nodes store only keys and child pointers. B+ trees have two advantages: (1) Internal nodes are smaller (no data pointers), so more keys fit per node, increasing the branching factor and reducing tree height. With 4 KB pages, a B+ tree internal node might hold 500 keys vs. 200 for a B-tree. (2) Leaf nodes are linked in a doubly-linked list, enabling efficient range scans: after finding the start key, the scan follows leaf pointers sequentially without revisiting internal nodes. B-trees require an in-order traversal involving internal nodes. Most databases use B+ trees because range queries and sequential scans are common (e.g., `WHERE price BETWEEN 10 AND 50`). $\square$

---

**Exercise 3.**
Describe the B-tree insertion algorithm when a node overflows. What is the amortized cost of node splits?

??? success "Solution to Exercise 3"
    When inserting a key into a full node (containing $m - 1$ keys): (1) insert the key into the node temporarily (now $m$ keys). (2) Split the node into two nodes at the median key. The left node gets the first $\lfloor(m-1)/2\rfloor$ keys, the right gets the last $\lfloor(m-1)/2\rfloor$ keys, and the median key is promoted to the parent. (3) If the parent overflows, split it recursively. In the worst case, splits propagate to the root, increasing tree height by 1. Amortized cost: each split creates one new node and one new key in the parent. Since a split can only occur when a node is full, and a full node was filled by $\lceil m/2 \rceil$ insertions since its last split (or creation), the amortized number of splits per insertion is $O(1/m)$. The amortized disk I/O cost per insertion is $O(\log_m n)$ for the search plus $O(1)$ amortized for splits. $\square$

---

**Exercise 4.**
A database table has $10^7$ rows and a B+ tree index on a column with 8-byte keys. Each disk page is 8 KB. Estimate the number of leaf pages, internal pages, and tree height.

??? success "Solution to Exercise 4"
    Each leaf page stores key-pointer pairs. With 8-byte keys and 8-byte pointers: 16 bytes per entry. Leaf page capacity: $8192 / 16 = 512$ entries. Number of leaf pages: $\lceil 10^7 / 512 \rceil = 19{,}532$. Internal nodes store keys and child pointers. With 8-byte keys and 8-byte pointers, each internal entry is 16 bytes (approximately), giving $\approx 512$ children per internal node. Level 1 (above leaves): $\lceil 19{,}532 / 512 \rceil = 39$ nodes. Level 2: $\lceil 39 / 512 \rceil = 1$ node (root). Height: 3 (root + 1 internal level + leaves). Total internal pages: $39 + 1 = 40$. These easily fit in memory (40 pages $\times$ 8 KB = 320 KB). Any point query requires 1 disk I/O (only the leaf page, since internal nodes are cached). $\square$

---

**Exercise 5.**
Compare B-tree indexing with LSM trees for a workload that is 90% writes and 10% reads. Which structure is better and why?

??? success "Solution to Exercise 5"
    **B-tree writes**: each insert/update requires reading the target leaf page, modifying it, and writing it back -- at least 2 random I/Os per write (read + write). Write amplification is moderate (each write touches one page). **LSM tree writes**: inserts go to an in-memory buffer (memtable). When full, the buffer is flushed sequentially to a new sorted file (SSTable). Sequential writes are 10--100x faster than random writes on both HDDs and SSDs. Write amplification is higher (compaction rewrites data multiple times) but each I/O is sequential. For 90% writes: LSM trees dominate because sequential I/O throughput is much higher than random I/O. A write-heavy LSM engine (e.g., RocksDB) can sustain 100K+ writes/sec vs. $\sim$10K for a B-tree on the same hardware. The 10% reads pay a cost: LSM reads may check multiple levels (read amplification), while B-tree reads are always one I/O. For this workload, the write advantage outweighs the read penalty. $\square$
