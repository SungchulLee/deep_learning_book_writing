# B-Tree Search

Searching in a [B-tree](definition.md) generalizes binary search tree lookup to multi-way branching.  At each node, the algorithm examines up to $2t - 1$ keys to determine which child subtree to descend into, and this process repeats until the key is found or a leaf is reached without a match.  Because a B-tree of minimum degree $t$ has height $O(\log_t n)$, each search touches at most $O(\log_t n)$ nodes — and on a disk-based system, that translates directly to $O(\log_t n)$ disk reads.

## Algorithm

Given a B-tree node $x$ and a search key $k$:

1. **Linear scan within the node:** find the smallest index $i$ such that $k \le x.key_i$.
2. If $k = x.key_i$, the key is found — return $(x, i)$.
3. If $x$ is a leaf, the key does not exist in the tree — return `nil`.
4. Otherwise, **recurse** into child $x.c_i$ (the subtree whose keys are between $x.key_{i-1}$ and $x.key_i$).

The search at each node can also use **binary search** instead of a linear scan, reducing the per-node work from $O(t)$ to $O(\log t)$.

## Pseudocode

```
B-TREE-SEARCH(x, k):
    i = 1
    while i <= x.n and k > x.key[i]:
        i = i + 1
    if i <= x.n and k == x.key[i]:
        return (x, i)                 # key found at position i
    elif x.leaf:
        return nil                     # key not in tree
    else:
        DISK-READ(x.c[i])
        return B-TREE-SEARCH(x.c[i], k)
```

The `DISK-READ` call reflects the fact that child nodes reside on disk and must be loaded into memory before they can be examined.

??? example "Searching for key 14 in a B-tree with t = 3"
    ```
              [  8, 17  ]
             /     |      \
       [2,4,6]  [10,13,15]  [20,25]
    ```

    | Step | Node | Keys examined | Action |
    |------|------|---------------|--------|
    | 1 | Root | 8, 17 | $14 > 8$ and $14 < 17$ → descend to middle child |
    | 2 | [10,13,15] | 10, 13, 15 | $14 > 13$ and $14 < 15$ → would descend, but this is a leaf |
    | 3 | — | — | Key 14 not found → return `nil` |

## Binary Search Within a Node

When $t$ is large (hundreds or thousands in a disk-based system), a linear scan through $O(t)$ keys per node is wasteful.  Binary search within each node reduces the per-node comparison count:

$$
\text{Per-node cost: } O(\log(2t - 1)) = O(\log t)
$$

The total search time becomes:

$$
O(\log_t n) \cdot O(\log t) = O(\log n)
$$

This shows that the total number of key comparisons is the same as in a balanced binary search tree — but the number of **disk accesses** is $O(\log_t n)$, which is dramatically smaller.

## Complexity

| Metric | Cost |
|--------|------|
| Disk accesses | $O(\log_t n)$ |
| Key comparisons (linear scan) | $O(t \log_t n)$ |
| Key comparisons (binary search) | $O(\log n)$ |

!!! note "Disk accesses dominate"
    In practice, the disk access cost dwarfs the CPU cost of key comparisons.  A search in a B-tree with $t = 1000$ and $n = 10^9$ requires at most 3 disk reads.  The $O(\log t)$ binary search within each node happens entirely in memory and is negligible compared to the millisecond-scale disk latency.

## Search vs Traversal

B-tree search follows a single root-to-leaf path, visiting $O(\log_t n)$ nodes.  A **range query** or **full traversal** visits all $O(n/t)$ nodes via an inorder walk.  For range queries on sorted data, [B+ trees](b_plus.md) are more efficient because their leaf-level linked list supports sequential scanning without backtracking through internal nodes.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 18.1–18.2. MIT Press.
- Knuth, D. E. (1998). *The Art of Computer Programming*, Volume 3, Section 6.2.4. Addison-Wesley.
