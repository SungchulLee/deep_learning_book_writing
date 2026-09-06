# Comparison Tables

Choosing the right data structure requires comparing operations side by side. This page
consolidates the time complexity of fundamental operations across all major data
structures, organized by use case. Use these tables as a quick reference when deciding
which structure fits your problem's access pattern.

## Core Operations Comparison

The following table compares the four most fundamental operations across basic data
structures. All complexities are worst-case unless noted.

| Structure | Access | Search | Insert | Delete |
|---|---|---|---|---|
| Array | $O(1)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| Sorted array | $O(1)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| Dynamic array | $O(1)$ | $O(n)$ | $O(1)$ amort. end | $O(n)$ |
| Singly linked list | $O(n)$ | $O(n)$ | $O(1)$ at head | $O(n)$ |
| Doubly linked list | $O(n)$ | $O(n)$ | $O(1)$ given node | $O(1)$ given node |
| Hash table | -- | $O(1)$ avg / $O(n)$ worst | $O(1)$ avg | $O(1)$ avg |
| BST (balanced) | -- | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
| Heap (binary) | $O(1)$ top only | $O(n)$ | $O(\log n)$ | $O(\log n)$ |
| Trie | -- | $O(L)$ | $O(L)$ | $O(L)$ |

Here $L$ is the key length for trie operations. A dash indicates the operation is not
naturally supported.

## Ordered Operations Comparison

When you need elements in sorted order or need predecessor/successor queries, only
certain structures qualify.

| Structure | Find Min | Find Max | Predecessor | Successor | Range Query |
|---|---|---|---|---|---|
| Sorted array | $O(1)$ | $O(1)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n + k)$ |
| BST (balanced) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n + k)$ |
| Min-heap | $O(1)$ | $O(n)$ | -- | -- | -- |
| Max-heap | $O(n)$ | $O(1)$ | -- | -- | -- |
| Skip list | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(\log n + k)$ |

Here $k$ is the number of elements in the query range.

!!! tip "When Order Matters"
    If you need both fast search and ordered iteration, a balanced BST or skip list
    is the right choice. Hash tables provide faster search but cannot enumerate
    elements in sorted order without a separate sort step.

## Space Comparison

| Structure | Space | Overhead per Element | Notes |
|---|---|---|---|
| Array | $O(n)$ | 0 (just the element) | Contiguous memory |
| Dynamic array | $O(n)$ | Up to $n$ wasted slots | Doubling policy |
| Singly linked list | $O(n)$ | 1 pointer | Non-contiguous |
| Doubly linked list | $O(n)$ | 2 pointers | More flexible deletion |
| Hash table (chaining) | $O(n + m)$ | 1 pointer + node | $m$ = table size |
| Hash table (open addr.) | $O(m)$ | None beyond element | Load factor $< 1$ |
| BST | $O(n)$ | 2 pointers | Left and right children |
| AVL tree | $O(n)$ | 2 pointers + balance | 1 byte extra |
| Red-Black tree | $O(n)$ | 2 pointers + color | 1 bit extra |
| Binary heap | $O(n)$ | 0 (array-based) | Implicit structure |
| B-tree (order $m$) | $O(n)$ | $m$ pointers per node | Large nodes |
| Trie | $O(N \cdot \Sigma)$ | $\Sigma$ pointers per node | $\Sigma$ = alphabet size |

## Priority Queue Implementations

Different structures offer different trade-offs for priority queue operations.

| Structure | Insert | Extract-Min | Decrease-Key | Merge | Build |
|---|---|---|---|---|---|
| Unsorted array | $O(1)$ | $O(n)$ | $O(1)$ | $O(1)$ | $O(n)$ |
| Sorted array | $O(n)$ | $O(1)$ | $O(n)$ | $O(n)$ | $O(n \log n)$ |
| Binary heap | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| Binomial heap | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| Fibonacci heap | $O(1)$ | $O(\log n)$ amort. | $O(1)$ amort. | $O(1)$ | $O(n)$ |
| Pairing heap | $O(1)$ | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(1)$ | $O(n)$ |

!!! warning "Fibonacci Heap Caveat"
    Fibonacci heaps have the best theoretical bounds but large constant factors.
    In practice, binary heaps outperform Fibonacci heaps for $n < 10^6$ due to
    cache efficiency and simpler operations.

## Dictionary/Map Implementations

| Structure | Search | Insert | Delete | Ordered? |
|---|---|---|---|---|
| Hash table | $O(1)$ avg | $O(1)$ avg | $O(1)$ avg | No |
| Balanced BST | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | Yes |
| Skip list | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(\log n)$ exp. | Yes |
| Trie | $O(L)$ | $O(L)$ | $O(L)$ | Yes (lexicographic) |
| Sorted array | $O(\log n)$ | $O(n)$ | $O(n)$ | Yes |

## Choosing by Access Pattern

| Access Pattern | Best Structure | Why |
|---|---|---|
| Random access by index | Array | $O(1)$ access |
| Fast membership test | Hash set | $O(1)$ average |
| Ordered enumeration | Balanced BST | In-order traversal |
| Fast min/max extraction | Heap | $O(1)$ peek, $O(\log n)$ extract |
| FIFO processing | Queue (linked or circular) | $O(1)$ enqueue and dequeue |
| LIFO processing | Stack (array) | $O(1)$ push and pop |
| Prefix matching | Trie | $O(L)$ per query |
| Range queries on points | Segment tree / BIT | $O(\log n)$ per query |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.

## Exercises

**Exercise 1.**
You need a data structure supporting insert, delete, search, and ordered iteration. Compare hash tables, balanced BSTs, and skip lists for this use case.

??? success "Solution to Exercise 1"
    | Operation | Hash Table | Balanced BST | Skip List |
    |---|---|---|---|
    | Insert | $O(1)$ avg | $O(\log n)$ | $O(\log n)$ exp |
    | Delete | $O(1)$ avg | $O(\log n)$ | $O(\log n)$ exp |
    | Search | $O(1)$ avg | $O(\log n)$ | $O(\log n)$ exp |
    | Ordered iter. | $O(n \log n)$ sort | $O(n)$ in-order | $O(n)$ level-0 scan |
    | Min/Max | $O(n)$ | $O(\log n)$ | $O(1)$ with pointers |

    Hash tables are fastest for point operations but cannot iterate in order. BSTs and skip lists support all four operations in $O(\log n)$ with $O(n)$ ordered iteration. BSTs are the best choice for this combined requirement. $\square$

---

**Exercise 2.**
A system needs to support: (a) insert a number, (b) delete a number, (c) find the median. Which data structure provides the best time complexity for all three operations?

??? success "Solution to Exercise 2"
    Use **two heaps**: a max-heap for the lower half and a min-heap for the upper half, maintaining the invariant that their sizes differ by at most 1. Insert: add to the appropriate heap and rebalance if sizes differ by more than 1. $O(\log n)$. Median: the top of the larger heap (or the average of both tops). $O(1)$. Delete: requires an indexed heap or augmented structure for $O(\log n)$ deletion by value. Alternative: an **order-statistic tree** (balanced BST with subtree sizes) supports all three in $O(\log n)$: insert/delete via BST operations, median via rank query for $\lfloor n/2 \rfloor$. The order-statistic tree is more versatile; the two-heap approach has a smaller constant factor for median queries. $\square$

---

**Exercise 3.**
For each scenario, select the optimal data structure: (a) frequent membership tests, (b) maintaining a sorted collection with range queries, (c) LIFO processing of tasks, (d) priority-based event scheduling.

??? success "Solution to Exercise 3"
    (a) **Hash set**: $O(1)$ expected membership test. If approximate answers are acceptable, a Bloom filter uses less memory. (b) **Balanced BST** (or B-tree for disk): $O(\log n)$ insert, delete, search; $O(\log n + k)$ range query returning $k$ results. (c) **Stack** (array-based): $O(1)$ push and pop, minimal overhead. (d) **Binary heap** (priority queue): $O(\log n)$ insert and extract-min. If decrease-key is frequent, use a Fibonacci heap or indexed priority queue. Each choice minimizes the complexity of the dominant operations for that scenario. $\square$

---

**Exercise 4.**
Explain why there is no single "best" data structure. Use the concept of tradeoffs to justify why multiple structures exist.

??? success "Solution to Exercise 4"
    Every data structure makes tradeoffs between: (1) **Time for different operations**: arrays have $O(1)$ access but $O(n)$ insertion; linked lists have $O(1)$ insertion but $O(n)$ access. (2) **Space**: hash tables use $O(n)$ with overhead for empty buckets; arrays are more compact. (3) **Worst-case vs. average-case**: hash tables are $O(1)$ average but $O(n)$ worst case; BSTs are $O(\log n)$ guaranteed. (4) **Static vs. dynamic**: sorted arrays are optimal for static data (binary search); BSTs are optimal for dynamic data. No single structure can be simultaneously optimal for all operations, all input distributions, and all memory constraints. The choice depends on the workload: which operations are frequent, whether worst-case guarantees are needed, and how much memory is available. This is why data structures is a rich field -- each structure occupies a different point in the tradeoff space. $\square$

---

**Exercise 5.**
A database index must support: point lookups ($O(1)$), range scans ($O(\log n + k)$), and ordered iteration ($O(n)$). Can any single data structure achieve all three? If not, what combination works?

??? success "Solution to Exercise 5"
    No single standard structure achieves $O(1)$ point lookup and $O(\log n + k)$ range scan simultaneously. Hash tables provide $O(1)$ lookups but $O(n \log n)$ range scans. B-trees provide $O(\log n)$ lookups and $O(\log n + k)$ range scans. The combination: maintain both a **hash index** for point lookups and a **B-tree index** on the same column. Point queries use the hash index; range queries use the B-tree. The cost: double the memory for indexes and maintaining consistency on updates. Many databases support this: PostgreSQL allows creating both a hash index and a B-tree index on the same column. The query optimizer automatically selects the appropriate index for each query type. $\square$
