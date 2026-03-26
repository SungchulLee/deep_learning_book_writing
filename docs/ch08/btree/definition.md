# B-Tree Definition

Binary search trees provide $O(\log n)$ operations when balanced, but each comparison touches a separate node that may reside on a different disk block.  On storage devices where a single block read is orders of magnitude slower than a CPU instruction, the number of **disk accesses** dominates performance.  B-trees address this by packing many keys into each node so that one disk read loads an entire node, and the tree's branching factor is high enough to keep the height — and therefore the number of disk accesses — extremely small.

## Minimum Degree

A B-tree is parameterized by a **minimum degree** $t \ge 2$.  This single parameter controls both the minimum and maximum number of keys that each node can hold.

## B-Tree Properties

A B-tree of minimum degree $t$ satisfies the following properties:

1. **Every node** $x$ stores $x.n$ keys in non-decreasing order: $x.key_1 \le x.key_2 \le \cdots \le x.key_{x.n}$.
2. **Every internal node** $x$ has $x.n + 1$ children.  The keys act as separators: for child $c_i$, all keys $k$ in the subtree rooted at $c_i$ satisfy $x.key_{i-1} \le k \le x.key_i$ (with appropriate boundary handling for the first and last children).
3. **All leaves are at the same depth**, which equals the height $h$ of the tree.
4. **Key bounds for non-root nodes:** every node other than the root has at least $t - 1$ keys and at most $2t - 1$ keys.
5. **Root bound:** the root has at least 1 key (if the tree is non-empty) and at most $2t - 1$ keys.

A node with $2t - 1$ keys is called **full**.

??? example "B-tree of minimum degree t = 2 (a 2-3-4 tree)"
    When $t = 2$, each non-root node holds 1 to 3 keys and has 2 to 4 children.  This special case is called a **2-3-4 tree**.

    ```
            [  8  ]
           /       \
       [3, 5]     [10, 12, 15]
      / |   \     / |   \    \
    [1] [4] [6] [9] [11] [13] [16,17]
    ```

    - The root has 1 key and 2 children.
    - The internal node `[10, 12, 15]` is full (3 keys = $2t - 1$).
    - All leaves reside at depth 2.

## Height Bound

The most important consequence of the B-tree properties is that the height grows logarithmically with the number of keys.

!!! note "B-tree height theorem"
    For a B-tree of minimum degree $t \ge 2$ containing $n \ge 1$ keys, the height $h$ satisfies:

    $$
    h \le \log_t \frac{n + 1}{2}
    $$

**Proof sketch.**  The tree has the fewest keys when every node contains exactly the minimum number.  The root contributes at least 1 key, the root's children contribute at least 2 nodes with $t - 1$ keys each, the next level has at least $2t$ nodes with $t - 1$ keys each, and so on.  Summing over all levels:

$$
n \ge 1 + (t - 1) \sum_{i=1}^{h} 2t^{i-1} = 1 + 2(t - 1) \cdot \frac{t^h - 1}{t - 1} = 2t^h - 1
$$

Solving for $h$ gives $t^h \le (n + 1)/2$, so $h \le \log_t \frac{n+1}{2}$. $\square$

For a typical disk-based system with $t = 1000$ and $n = 10^9$ keys, the height is at most $\log_{1000}(5 \times 10^8) \approx 3$.  Three disk reads suffice to locate any key among a billion entries.

## Node Structure

Each B-tree node stores:

| Field | Description |
|-------|-------------|
| $x.n$ | Number of keys currently stored |
| $x.key_1, \ldots, x.key_{x.n}$ | Keys in sorted order |
| $x.c_1, \ldots, x.c_{x.n+1}$ | Child pointers (internal nodes only) |
| $x.leaf$ | Boolean flag: is the node a leaf? |

In practice, each key may have an associated satellite data pointer (or the data may be stored inline for small records).

## Choosing the Minimum Degree

The minimum degree $t$ is typically chosen so that a full node fits into a single disk block.  If a disk block holds $B$ bytes, a key occupies $k$ bytes, and a child pointer occupies $p$ bytes, then:

$$
(2t - 1) \cdot k + 2t \cdot p \le B
$$

Solving for $t$ maximizes the branching factor while keeping each node within one block.

!!! tip "Practical values"
    Database systems commonly use $t$ values in the range 50–2000, depending on the block size (typically 4 KB or 8 KB) and the key size.  With $t = 500$ and 8 KB blocks, each node holds up to 999 keys.

## Comparison with Binary Search Trees

| Property | BST | B-tree (degree $t$) |
|----------|-----|---------------------|
| Keys per node | 1 | $t-1$ to $2t-1$ |
| Children per node | 2 | $t$ to $2t$ |
| Height | $O(\log_2 n)$ | $O(\log_t n)$ |
| Disk accesses per search | $O(\log_2 n)$ | $O(\log_t n)$ |
| Balanced? | Only if self-balancing | Always |

The reduction from $\log_2 n$ to $\log_t n$ disk accesses is the fundamental advantage of B-trees.

## Reference

- Bayer, R., & McCreight, E. (1972). Organization and maintenance of large ordered indexes. *Acta Informatica*, 1(3), 173–189.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 18. MIT Press.
