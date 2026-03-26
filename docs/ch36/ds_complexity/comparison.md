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
