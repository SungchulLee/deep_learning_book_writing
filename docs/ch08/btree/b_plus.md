# B+ Trees

A [B-tree](definition.md) stores keys and data in every node — both internal nodes and leaves.  This design works well for point lookups, but range queries suffer because consecutive keys may live in nodes scattered across different subtrees.  The **B+ tree** solves this problem by pushing all data records to the leaves and linking the leaves into a sorted linked list.  Internal nodes store only keys that serve as routing guides, and a sequential scan of any key range reduces to a simple linked-list traversal at the leaf level.  This structure makes B+ trees the dominant index in virtually all relational database systems.

## Structure

A B+ tree of order $m$ satisfies the following properties:

1. **Internal nodes** contain between $\lceil m/2 \rceil$ and $m$ children, with $k - 1$ keys guiding the search among $k$ children.  The root may have as few as 2 children (unless it is also a leaf).
2. **Leaf nodes** contain between $\lceil (m-1)/2 \rceil$ and $m - 1$ key–value pairs.  Every data record resides in a leaf.
3. **All leaves are at the same depth**, ensuring $O(\log n)$ search in the worst case.
4. **Leaf linked list:** every leaf stores a pointer to the next leaf in key order, enabling efficient range scans.

The key difference from a standard B-tree is that internal nodes do **not** store data — they store only separator keys that direct searches toward the correct leaf.

??? example "B+ tree of order 4"
    ```
    Internal:       [  17  |  35  ]
                   /        |       \
    Leaves:   [3,5,12] → [17,20,30] → [35,42,50]
    ```

    - Internal node holds separator keys 17 and 35.
    - All data resides in the three leaf nodes.
    - Arrows (`→`) represent the leaf-level linked list.

## Search

Searching for a key $k$ starts at the root and descends through internal nodes using the separator keys.  At each internal node with keys $k_1, k_2, \ldots, k_{j}$, the search follows child pointer $c_i$ where $k_{i-1} \le k < k_i$ (with appropriate boundary handling).  The search always reaches a leaf, where a linear or binary scan determines whether $k$ is present.

$$
\text{Search cost} = O(\log_m n) \cdot O(\log m) = O(\log n)
$$

The first factor counts the levels, and the second accounts for binary search within each node.

## Insertion

Inserting a key–value pair follows these steps:

1. **Search** for the correct leaf node $L$.
2. **Insert** the key–value pair into $L$ in sorted order.
3. If $L$ has fewer than $m$ entries, insertion is complete.
4. If $L$ **overflows** (reaches $m$ entries), **split** $L$ into two leaves $L_1$ and $L_2$:
      - $L_1$ keeps the first $\lceil m/2 \rceil$ entries.
      - $L_2$ gets the remaining entries.
      - The smallest key of $L_2$ is **copied up** to the parent as a new separator.
5. If the parent overflows, split it recursively.  When an internal node splits, the middle key is **pushed up** (not copied), since internal nodes hold only routing keys.

!!! warning "Copy up vs push up"
    In B+ trees, leaf splits **copy** the separator up to the parent (the key remains in the leaf because all data must stay at the leaf level).  Internal node splits **push** the middle key up (removing it from the splitting node).  This differs from standard B-trees, where splits always push the middle key up.

## Deletion

Deleting a key from a B+ tree:

1. **Search** for the leaf $L$ containing the key and remove it.
2. If $L$ still has at least $\lceil (m-1)/2 \rceil$ entries, deletion is complete.
3. If $L$ **underflows**, try to **borrow** an entry from an adjacent sibling.
4. If borrowing is not possible (sibling is at minimum occupancy), **merge** $L$ with its sibling and remove the corresponding separator from the parent.
5. If the parent underflows, apply the same borrow-or-merge logic recursively.

A subtle point: if the deleted key also appears as a separator in an internal node, the separator must be updated to reflect the new smallest key of the right subtree.

## Range Queries

Range queries are the primary motivation for B+ trees.  To find all keys in the range $[a, b]$:

1. Search for key $a$ to locate the starting leaf.
2. Scan forward through the leaf linked list, collecting all keys $\le b$.

$$
\text{Range query cost} = O(\log_m n + k)
$$

where $k$ is the number of keys in the range.  The $O(\log_m n)$ term locates the starting leaf, and the $O(k)$ term covers the sequential scan.  Because leaves are stored contiguously on disk (or in cache-friendly blocks), this scan is extremely efficient in practice.

## Complexity

| Operation | Time complexity |
|-----------|----------------|
| Search | $O(\log n)$ |
| Insert | $O(\log n)$ |
| Delete | $O(\log n)$ |
| Range query | $O(\log n + k)$ |

All operations have the same asymptotic cost as a standard B-tree.  The practical advantage of B+ trees lies in the constant factors: higher fanout (since internal nodes carry no data) and sequential leaf access for range queries.

## B+ Tree vs B-Tree

| Property | B-tree | B+ tree |
|----------|--------|---------|
| Data location | All nodes | Leaves only |
| Internal node content | Keys + data + child pointers | Keys + child pointers |
| Leaf linked list | No | Yes |
| Range query | $O(\log n + k \log n)$ | $O(\log n + k)$ |
| Fanout | Lower (data occupies space) | Higher (more keys per node) |

The higher fanout of B+ trees translates directly to fewer levels in the tree, which means fewer disk seeks per operation.

## Reference

- Comer, D. (1979). The ubiquitous B-tree. *ACM Computing Surveys*, 11(2), 121–137.
- Ramakrishnan, R., & Gehrke, J. (2003). *Database Management Systems* (3rd ed.), Chapter 10. McGraw-Hill.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 18. MIT Press.
