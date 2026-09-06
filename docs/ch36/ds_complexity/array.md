# Array Complexities

Arrays are the most fundamental data structure in computing. They store elements in
contiguous memory, providing $O(1)$ random access by index -- a property that no other
data structure can match. Understanding the complexity of each array operation is
essential because arrays underlie nearly every other data structure.

## Static Array Operations

A static array has a fixed size determined at allocation. Elements cannot be added or
removed; only the values at existing indices can be modified.

| Operation | Time | Notes |
|---|---|---|
| Access by index | $O(1)$ | Direct memory offset calculation |
| Update by index | $O(1)$ | Same as access |
| Search (unsorted) | $O(n)$ | Must scan all elements |
| Search (sorted) | $O(\log n)$ | Binary search |
| Find min/max | $O(n)$ | Must scan all elements |

The $O(1)$ access time comes from the address formula: element at index $i$ is located
at memory address

$$
\text{base} + i \times \text{element\_size}
$$

This formula requires contiguous memory allocation, which is the defining property of
arrays.

## Dynamic Array Operations

A dynamic array (Python `list`, C++ `vector`, Java `ArrayList`) automatically resizes
when capacity is exceeded, typically by doubling.

| Operation | Average | Worst | Amortized | Notes |
|---|---|---|---|---|
| Access by index | $O(1)$ | $O(1)$ | -- | Same as static |
| Append (push back) | $O(1)$ | $O(n)$ | $O(1)$ | Resize triggers copy |
| Insert at index $i$ | $O(n)$ | $O(n)$ | -- | Shifts $n - i$ elements |
| Delete at index $i$ | $O(n)$ | $O(n)$ | -- | Shifts $n - i$ elements |
| Pop last | $O(1)$ | $O(1)$ | $O(1)$ | No shifting needed |
| Search | $O(n)$ | $O(n)$ | -- | Linear scan |

!!! tip "Why Amortized O(1) for Append"
    When the array is full, doubling the capacity costs $O(n)$ to copy all
    elements. But this happens only after $n$ cheap $O(1)$ appends. Spreading the
    expensive copy over the preceding cheap operations gives $O(1)$ amortized cost
    per append.

## Sorted Array Operations

Maintaining a sorted array enables binary search but makes insertion expensive.

| Operation | Time | Notes |
|---|---|---|
| Search | $O(\log n)$ | Binary search |
| Insert (maintaining order) | $O(n)$ | Find position + shift elements |
| Delete (maintaining order) | $O(n)$ | Find element + shift elements |
| Find min | $O(1)$ | First element |
| Find max | $O(1)$ | Last element |
| Find predecessor/successor | $O(\log n)$ | Binary search variant |
| Merge two sorted arrays | $O(m + n)$ | Two-pointer merge |

## Multidimensional Arrays

A 2D array of size $m \times n$ uses either row-major or column-major layout.

| Operation | Time | Notes |
|---|---|---|
| Access `A[i][j]` | $O(1)$ | Offset = $i \times n + j$ (row-major) |
| Row traversal | $O(n)$ | Cache-friendly in row-major |
| Column traversal | $O(m)$ | Cache-unfriendly in row-major |
| Full traversal | $O(mn)$ | Row-by-row is faster due to caching |
| Transpose | $O(mn)$ | Must visit every element |

!!! warning "Cache Performance"
    Traversing a 2D array column-by-column in a row-major language (C, C++, Python
    NumPy default) causes cache misses at every access. For large matrices, this
    can be 10--100x slower than row-by-row traversal.

## Space Complexity

| Array Type | Space | Notes |
|---|---|---|
| Static array of $n$ elements | $O(n)$ | Exact: $n \times$ element size |
| Dynamic array | $O(n)$ | Up to $2n$ allocated due to doubling |
| 2D array $m \times n$ | $O(mn)$ | Contiguous or array of pointers |
| Sparse array (hash map) | $O(k)$ | $k$ = number of non-zero elements |

## Common Array Algorithms and Their Complexities

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Prefix sum (build) | $O(n)$ | $O(n)$ | Enables $O(1)$ range sum queries |
| Prefix sum (query) | $O(1)$ | -- | $\text{sum}(l, r) = P[r+1] - P[l]$ |
| Kadane's algorithm | $O(n)$ | $O(1)$ | Maximum subarray sum |
| Dutch National Flag | $O(n)$ | $O(1)$ | 3-way partition |
| Rotate by $k$ | $O(n)$ | $O(1)$ | Three reversals trick |
| Remove duplicates (sorted) | $O(n)$ | $O(1)$ | Two pointers |

## Comparison with Other Structures

| Operation | Array | Linked List | Hash Table | BST (balanced) |
|---|---|---|---|---|
| Access by index | $O(1)$ | $O(n)$ | -- | -- |
| Search | $O(n)$ | $O(n)$ | $O(1)$ avg | $O(\log n)$ |
| Insert at end | $O(1)$ amort. | $O(1)$ | $O(1)$ avg | $O(\log n)$ |
| Insert at beginning | $O(n)$ | $O(1)$ | -- | $O(\log n)$ |
| Delete | $O(n)$ | $O(1)$ given pointer | $O(1)$ avg | $O(\log n)$ |
| Ordered traversal | $O(n)$ | $O(n)$ | $O(n \log n)$ sort | $O(n)$ |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.

## Exercises

**Exercise 1.**
Compare the time complexities of array operations: access by index, search, insertion at the end, insertion at position $i$, and deletion at position $i$.

??? success "Solution to Exercise 1"
    | Operation | Time | Explanation |
    |---|---|---|
    | Access by index | $O(1)$ | Direct address computation: base + i * element_size |
    | Search (unsorted) | $O(n)$ | Must scan all elements |
    | Search (sorted) | $O(\log n)$ | Binary search |
    | Insert at end | $O(1)$ amortized | Append; $O(n)$ when resizing |
    | Insert at position $i$ | $O(n)$ | Shift elements $i, i+1, \ldots, n-1$ right |
    | Delete at position $i$ | $O(n)$ | Shift elements $i+1, \ldots, n-1$ left |

    The key advantage of arrays is $O(1)$ random access. The key disadvantage is $O(n)$ insertion/deletion in the middle. $\square$

---

**Exercise 2.**
A dynamic array doubles its capacity when full. Prove that the amortized cost of $n$ append operations is $O(1)$ per operation.

??? success "Solution to Exercise 2"
    Use the aggregate method. Over $n$ appends, resizing occurs when the array size reaches $1, 2, 4, 8, \ldots, 2^k$ where $2^k \le n$. Each resize copies $2^j$ elements. Total copies: $\sum_{j=0}^{k} 2^j = 2^{k+1} - 1 < 2n$. Adding the $n$ individual insertions (each $O(1)$ without resizing): total work $< 3n$. Amortized cost: $3n / n = O(1)$ per append. Alternatively, using the banker's method: charge 3 units per append (1 for the insertion itself, 2 saved for future copying). When a resize doubles from $m$ to $2m$, the $m$ new elements since the last resize have saved $2m$ units, which pays for copying $m$ elements. $\square$

---

**Exercise 3.**
Explain why arrays have better cache performance than linked lists for sequential access, despite both requiring $O(n)$ time.

??? success "Solution to Exercise 3"
    Arrays store elements contiguously in memory. When the CPU accesses `arr[i]`, the hardware prefetcher loads the entire cache line (typically 64 bytes) containing `arr[i]` and its neighbors. Subsequent accesses to `arr[i+1], arr[i+2], ...` hit the L1 cache ($\sim$1 ns per access). Linked list nodes are scattered in memory (allocated by `malloc` at arbitrary addresses). Each `node->next` traversal is a pointer chase to a potentially distant memory location, causing an L1/L2 cache miss ($\sim$5--100 ns per access). For sequential traversal of $n$ elements: arrays make $\sim n/16$ cache line fetches (16 ints per 64-byte line); linked lists make $\sim n$ independent memory accesses. The practical speedup is 5--20x for arrays over linked lists, despite identical $O(n)$ asymptotic complexity. $\square$

---

**Exercise 4.**
Given $n = 10^5$ elements, compare the practical performance of: (a) searching an unsorted array, (b) binary searching a sorted array, (c) hash table lookup. Include estimated times.

??? success "Solution to Exercise 4"
    (a) Unsorted array linear search: $O(n)$, average $n/2 = 50{,}000$ comparisons. At $\sim$1 ns per comparison (cached): $\sim$50 microseconds. (b) Sorted array binary search: $O(\log n)$, $\lceil \log_2(10^5) \rceil = 17$ comparisons. Each comparison may cause a cache miss ($\sim$10 ns for L2): $\sim$170 ns. (c) Hash table lookup: $O(1)$ expected, $\sim$2--3 memory accesses (hash computation + bucket access + possible collision resolution). At $\sim$10 ns per access: $\sim$30 ns. Ranking: hash table (30 ns) < binary search (170 ns) < linear search (50 microseconds). For repeated lookups, the hash table is $\sim$6x faster than binary search and $\sim$1700x faster than linear search. The preprocessing costs (sorting: $O(n \log n)$; hash table: $O(n)$) are amortized over many lookups. $\square$

---

**Exercise 5.**
A problem requires frequent insertions and deletions at arbitrary positions while maintaining sorted order. An array requires $O(n)$ per operation. Propose a data structure with better complexity.

??? success "Solution to Exercise 5"
    A **balanced BST** (AVL tree, red-black tree, or treap) supports insert, delete, and search in $O(\log n)$. It maintains sorted order through the BST invariant. Alternatively, a **skip list** provides $O(\log n)$ expected time for all operations. For competitive programming, a **Fenwick tree with coordinate compression** can support order-statistic queries (rank, select) in $O(\log n)$, acting as a sorted multiset. The tradeoff compared to arrays: $O(\log n)$ vs. $O(n)$ for modifications, but $O(\log n)$ vs. $O(1)$ for access by index (unless the BST is augmented with subtree sizes for order-statistic operations). For this problem, the $O(\log n)$ modification time outweighs the slower access time. $\square$
