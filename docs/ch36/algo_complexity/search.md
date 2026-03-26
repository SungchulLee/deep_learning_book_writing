# Search Algorithm Complexities

Searching is the most fundamental operation in computing: given a collection of
elements, find the one that matches a target. The complexity of a search algorithm
depends on the structure of the data (sorted vs. unsorted, array vs. tree vs. graph)
and whether we can exploit ordering to eliminate candidates in bulk.

## Array Search

Linear search makes no assumptions about order, while binary search requires a sorted
array and eliminates half the remaining candidates at each step.

| Algorithm | Best | Average | Worst | Space | Requirement |
|---|---|---|---|---|---|
| Linear search | $O(1)$ | $O(n)$ | $O(n)$ | $O(1)$ | None |
| Binary search | $O(1)$ | $O(\log n)$ | $O(\log n)$ | $O(1)$ | Sorted array |
| Binary search (recursive) | $O(1)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | Sorted array |
| Interpolation search | $O(1)$ | $O(\log \log n)$ | $O(n)$ | $O(1)$ | Sorted, uniform distribution |
| Exponential search | $O(1)$ | $O(\log i)$ | $O(\log n)$ | $O(1)$ | Sorted array; $i$ = target position |
| Ternary search | $O(1)$ | $O(\log_3 n)$ | $O(\log_3 n)$ | $O(1)$ | Unimodal function |

!!! tip "Binary vs. Ternary Search"
    Although ternary search divides the range into three parts, it performs
    $2\log_3 n \approx 1.26 \log_2 n$ comparisons, which is worse than binary
    search's $\log_2 n$. Ternary search is useful only for unimodal function
    optimization, not for sorted array search.

## Why Binary Search is Optimal

Any comparison-based search on a sorted array of $n$ elements requires at least
$\lceil \log_2(n + 1) \rceil$ comparisons in the worst case. This follows from the
decision tree model: a binary tree with $n + 1$ leaves (one for each gap between
elements plus the two ends) must have height at least $\lceil \log_2(n + 1) \rceil$.
Binary search achieves this lower bound.

## Tree-Based Search

Search trees organize elements to support efficient lookup, insertion, and deletion.
The complexity depends on the tree's balance guarantee.

| Data Structure | Search | Insert | Delete | Space |
|---|---|---|---|---|
| BST (unbalanced) | $O(n)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| BST (random) | $O(\log n)$ expected | $O(\log n)$ expected | $O(\log n)$ expected | $O(n)$ |
| AVL tree | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| Red-Black tree | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| B-tree (order $m$) | $O(\log_m n)$ | $O(\log_m n)$ | $O(\log_m n)$ | $O(n)$ |
| Splay tree | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(n)$ |
| Trie | $O(L)$ | $O(L)$ | $O(L)$ | $O(N \cdot \Sigma)$ |

Here $L$ is the key length, $N$ is the number of stored keys, and $\Sigma$ is the
alphabet size. Trie space can be reduced to $O(N \cdot L)$ with compressed (Patricia)
tries.

## Hash-Based Search

Hash tables provide $O(1)$ expected-time search by mapping keys to array indices via a
hash function.

| Method | Search (avg) | Search (worst) | Insert (avg) | Space |
|---|---|---|---|---|
| Chaining | $O(1 + \alpha)$ | $O(n)$ | $O(1)$ | $O(n + m)$ |
| Open addressing | $O\!\left(\frac{1}{1 - \alpha}\right)$ | $O(n)$ | $O\!\left(\frac{1}{1 - \alpha}\right)$ | $O(m)$ |

Here $\alpha = n/m$ is the load factor, $n$ is the number of stored elements, and $m$
is the table size. Performance degrades as $\alpha \to 1$ for open addressing.

## Specialized Search

Some search problems require algorithms beyond simple comparison or hashing.

| Algorithm | Time | Space | Application |
|---|---|---|---|
| KD-tree (construction) | $O(n \log n)$ | $O(n)$ | Multidimensional points |
| KD-tree (nearest neighbor) | $O(\log n)$ avg, $O(n)$ worst | $O(\log n)$ | Nearest neighbor query |
| KD-tree (range search) | $O(n^{1-1/d} + k)$ | $O(\log n)$ | $k$ results in $d$ dimensions |
| A* search | $O(b^d)$ | $O(b^d)$ | Graph with admissible heuristic |
| Bidirectional BFS | $O(b^{d/2})$ | $O(b^{d/2})$ | Unweighted shortest path |

!!! warning "KD-tree Degradation"
    KD-trees degrade to $O(n)$ per query in high dimensions ($d > 20$). For
    high-dimensional nearest-neighbor search, approximate methods like locality-sensitive
    hashing (LSH) are preferred.

## Practical Input Size Guidelines

| $n$ | Linear $O(n)$ | Binary $O(\log n)$ | Hash $O(1)$ | BST $O(\log n)$ |
|---|---|---|---|---|
| $10^3$ | fast | instant | instant | instant |
| $10^6$ | fast | fast | fast | fast |
| $10^9$ | slow | fast | fast | fast |
| $10^{12}$ | infeasible | fast | fast | fast |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Knuth, D. *The Art of Computer Programming, Vol. 3: Sorting and Searching*. 2nd ed. Addison-Wesley, 1998.
