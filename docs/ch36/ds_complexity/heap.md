# Heap Family

A heap is a tree-based data structure that satisfies the heap property: in a min-heap,
every parent is smaller than or equal to its children. This simple invariant gives
$O(1)$ access to the minimum element and $O(\log n)$ insertion and extraction,
making heaps the standard implementation for priority queues.

## Binary Heap

The binary heap stores elements in an array where the children of index $i$ are at
indices $2i + 1$ and $2i + 2$ (0-indexed). This implicit tree structure eliminates
pointer overhead and provides excellent cache performance.

| Operation | Time | Notes |
|---|---|---|
| Find min (or max) | $O(1)$ | Root of the heap |
| Insert | $O(\log n)$ | Append + sift up |
| Extract min (or max) | $O(\log n)$ | Swap root with last, sift down |
| Decrease key | $O(\log n)$ | Update value + sift up |
| Delete arbitrary | $O(\log n)$ | Decrease to $-\infty$, then extract |
| Build heap | $O(n)$ | Bottom-up heapify |
| Heap sort | $O(n \log n)$ | Build + $n$ extractions |
| Merge two heaps | $O(n)$ | Build heap on concatenation |

The $O(n)$ build-heap bound follows from the fact that most nodes are near the
bottom of the tree. Specifically, the total work is:

$$
\sum_{h=0}^{\lfloor \log n \rfloor} \left\lceil \frac{n}{2^{h+1}} \right\rceil \cdot O(h) = O(n)
$$

where $h$ is the height of each node.

!!! tip "Why Build Heap is O(n), Not O(n log n)"
    A common mistake is assuming build-heap costs $O(n \log n)$ by inserting
    elements one at a time. The bottom-up approach (Floyd's algorithm) is faster
    because it performs sift-down on each node, and most nodes are at low heights
    where sift-down is cheap.

## d-ary Heap

A $d$-ary heap generalizes the binary heap by allowing each node to have $d$ children.

| Operation | Time | Trade-off |
|---|---|---|
| Find min | $O(1)$ | Same as binary |
| Insert | $O(\log_d n)$ | Faster sift-up |
| Extract min | $O(d \log_d n)$ | Compare $d$ children per level |
| Decrease key | $O(\log_d n)$ | Faster than binary for large $d$ |

Increasing $d$ makes insertions faster (shallower tree) but extractions slower (more
comparisons per level). For Dijkstra's algorithm, $d = E/V$ balances the number of
decrease-key and extract-min operations.

## Advanced Heap Variants

| Heap Type | Insert | Extract-Min | Decrease-Key | Merge | Space |
|---|---|---|---|---|---|
| Binary | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| Binomial | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| Fibonacci | $O(1)$ | $O(\log n)$ amort. | $O(1)$ amort. | $O(1)$ | $O(n)$ |
| Pairing | $O(1)$ | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(1)$ | $O(n)$ |
| Leftist | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| Skew | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(n)$ |

!!! warning "Fibonacci Heap Practicality"
    Fibonacci heaps achieve the best theoretical bounds for decrease-key and merge,
    making them optimal for Dijkstra ($O(V \log V + E)$) and Prim ($O(E + V \log V)$).
    However, the large constant factors and complex implementation make binary
    heaps faster in practice for $n < 10^6$.

## Heap Applications and Their Complexities

| Application | Algorithm | Heap Operations | Total Time |
|---|---|---|---|
| Priority queue | -- | Insert + extract-min | $O(\log n)$ each |
| Heap sort | Build + extract all | $n$ extract-min | $O(n \log n)$ |
| $k$ largest elements | Build min-heap of size $k$ | $n$ conditional inserts | $O(n \log k)$ |
| Merge $k$ sorted lists | Min-heap of size $k$ | $N$ extract + insert | $O(N \log k)$ |
| Running median | Two heaps (max + min) | Insert + rebalance | $O(\log n)$ per element |
| Dijkstra's algorithm | Min-heap | $V$ extract + $E$ decrease-key | $O((V+E) \log V)$ |

Here $N$ is the total number of elements across all lists.

## Binary Heap vs Sorted Array for Priority Queue

| Operation | Binary Heap | Sorted Array | Unsorted Array |
|---|---|---|---|
| Insert | $O(\log n)$ | $O(n)$ | $O(1)$ |
| Extract-min | $O(\log n)$ | $O(1)$ | $O(n)$ |
| Peek min | $O(1)$ | $O(1)$ | $O(n)$ |
| Build from $n$ | $O(n)$ | $O(n \log n)$ | $O(n)$ |
| Decrease-key | $O(\log n)$ | $O(n)$ | $O(1)$ |

The binary heap provides the best balance: $O(\log n)$ for both insert and extract.
Use an unsorted array only when inserts dominate and extractions are rare.

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Fredman, M. L. and Tarjan, R. E. "Fibonacci heaps and their uses in improved network optimization algorithms." *JACM*, 34(3), 1987.
