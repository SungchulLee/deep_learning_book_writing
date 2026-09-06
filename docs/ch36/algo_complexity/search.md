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

## Exercises

**Exercise 1.**
Compare linear search $O(n)$, binary search $O(\log n)$, and hash lookup $O(1)$. What preprocessing does each require?

??? success "Solution to Exercise 1"
    **Linear search**: no preprocessing. Works on unsorted data. Scans each element sequentially. Best for small collections or one-time searches. **Binary search**: requires sorted data ($O(n \log n)$ preprocessing for sorting). Halves the search space at each step. Best for repeated searches on static sorted data. **Hash lookup**: requires building a hash table ($O(n)$ preprocessing). Provides $O(1)$ expected-time lookups. Best for repeated exact-match queries. Binary search is preferable over hashing when: (1) range queries are needed (find all elements in $[a, b]$); (2) the data is already sorted; (3) worst-case $O(\log n)$ is needed (hashing has $O(n)$ worst case). Hashing is preferable for point queries on large unsorted datasets. $\square$

---

**Exercise 2.**
Prove that any comparison-based search algorithm on a sorted array of $n$ elements requires $\Omega(\log n)$ comparisons in the worst case.

??? success "Solution to Exercise 2"
    A comparison-based algorithm can be modeled as a binary decision tree: each internal node represents a comparison with two outcomes (less/greater or equal/not equal). Each leaf represents a possible answer (one of $n$ elements or "not found": $n + 1$ outcomes). A binary tree with $L$ leaves has height $\ge \lceil \log_2 L \rceil$. With $L = n + 1$ leaves: height $\ge \lceil \log_2(n + 1) \rceil = \Omega(\log n)$. The worst-case number of comparisons equals the height of the decision tree, so any comparison-based search requires $\Omega(\log n)$ comparisons. Binary search achieves this bound with exactly $\lceil \log_2(n + 1) \rceil$ comparisons, making it optimal. $\square$

---

**Exercise 3.**
Interpolation search achieves $O(\log \log n)$ expected time on uniformly distributed data. Explain the algorithm and why it degrades to $O(n)$ on adversarial data.

??? success "Solution to Exercise 3"
    Interpolation search estimates the position of the target $x$ in sorted array $A[lo..hi]$ by linear interpolation: $\text{mid} = lo + \lfloor (x - A[lo]) / (A[hi] - A[lo]) \times (hi - lo) \rfloor$. For uniformly distributed data, this estimate is close to the true position, so each step reduces the search space by a square-root factor: $n \to \sqrt{n} \to n^{1/4} \to \ldots$, giving $O(\log \log n)$ steps. On adversarial data (e.g., $A = [1, 2, 3, \ldots, 999, 10^9]$ searching for 999): interpolation estimates mid $\approx 0$ (999 is tiny relative to $10^9$), so the algorithm scans nearly linearly from the start, taking $O(n)$ steps. The algorithm has no worst-case guarantee better than $O(n)$ because the interpolation can be arbitrarily misleading. $\square$

---

**Exercise 4.**
Describe exponential search and analyze its time complexity. When is it preferable to standard binary search?

??? success "Solution to Exercise 4"
    Exponential search finds the range containing the target, then binary searches within it. Steps: (1) Starting from position 1, double the index: 1, 2, 4, 8, 16, ... until $A[2^k] \ge x$ or $2^k > n$. (2) Binary search in the range $[2^{k-1}, \min(2^k, n)]$. Phase 1 takes $O(\log i)$ steps where $i$ is the target's position. Phase 2 takes $O(\log(2^k - 2^{k-1})) = O(k) = O(\log i)$. Total: $O(\log i)$. Exponential search is preferable when: (1) the target is near the beginning of a large array -- $O(\log i)$ is much better than binary search's $O(\log n)$; (2) the array size is unknown (infinite or streaming); (3) the data structure supports efficient sequential access but not random access (e.g., a linked list with skip pointers). $\square$

---

**Exercise 5.**
A sorted array supports binary search in $O(\log n)$. If the array is modified (insertions/deletions), maintaining sorted order costs $O(n)$ per modification. Describe a data structure that supports both searches and modifications in $O(\log n)$.

??? success "Solution to Exercise 5"
    A **balanced BST** (AVL tree, red-black tree) stores elements in sorted order with $O(\log n)$ search, insert, and delete. Alternatively, a **skip list** provides $O(\log n)$ expected time for all operations. For the specific case of an array that needs both binary search and insertions, an **order-statistic tree** (balanced BST augmented with subtree sizes) supports: find the $k$-th element in $O(\log n)$, insert in $O(\log n)$, delete in $O(\log n)$, and rank query (position of an element) in $O(\log n)$. This replaces the sorted array + binary search combination when modifications are frequent. The tradeoff: BSTs have higher constant factors than arrays (pointer overhead, cache misses) but avoid the $O(n)$ shift cost of array insertions. $\square$
