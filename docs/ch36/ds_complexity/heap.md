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

## Exercises

**Exercise 1.**
Compare binary heaps and Fibonacci heaps in terms of insert, extract-min, and decrease-key operations. When is a Fibonacci heap worth the implementation complexity?

??? success "Solution to Exercise 1"
    | Operation | Binary Heap | Fibonacci Heap |
    |---|---|---|
    | Insert | $O(\log n)$ | $O(1)$ amortized |
    | Extract-min | $O(\log n)$ | $O(\log n)$ amortized |
    | Decrease-key | $O(\log n)$ | $O(1)$ amortized |

    Fibonacci heaps improve decrease-key from $O(\log n)$ to $O(1)$, which matters in algorithms that call decrease-key frequently: Dijkstra's algorithm ($O(E$ decrease-keys) improves from $O((V+E) \log V)$ to $O(E + V \log V)$; Prim's MST similarly. Fibonacci heaps are worth the complexity for dense graphs ($E = \Theta(V^2)$) where the improvement is from $O(V^2 \log V)$ to $O(V^2)$. For sparse graphs ($E = O(V)$), the improvement is marginal. In practice, binary heaps are preferred due to simpler implementation and better cache performance; Fibonacci heaps are mainly of theoretical interest. $\square$

---

**Exercise 2.**
Prove that building a binary heap from $n$ elements using the bottom-up (Floyd's) method takes $O(n)$ time, not $O(n \log n)$.

??? success "Solution to Exercise 2"
    Floyd's algorithm starts at the last internal node and sifts each node down to restore the heap property. A node at height $h$ takes $O(h)$ sift-down operations. The number of nodes at height $h$ in a complete binary tree is $\lceil n / 2^{h+1} \rceil$. Total work: $\sum_{h=0}^{\lfloor \log n \rfloor} \lceil n / 2^{h+1} \rceil \cdot O(h) \le n \sum_{h=0}^{\infty} h / 2^{h+1} = n \cdot 2 = O(n)$. The sum converges because $\sum h / 2^h = 2$. The key insight: most nodes are near the bottom (height 0 or 1), where sift-down is cheap. Only $O(1)$ nodes are at height $\log n$, where sift-down is expensive. This is faster than $n$ individual insertions ($O(n \log n)$) because insertions sift up, and most nodes are at high levels. $\square$

---

**Exercise 3.**
A priority queue supports insert and extract-min. Describe how to efficiently support a "merge" operation that combines two priority queues.

??? success "Solution to Exercise 3"
    **Binary heap**: merging two heaps of sizes $m$ and $n$ requires concatenating the arrays and rebuilding the heap in $O(m + n)$. This is the best possible for binary heaps but expensive for repeated merges. **Binomial heap**: merge in $O(\log n)$ by merging sorted lists of binomial trees (similar to binary addition). Supports insert in $O(1)$ amortized and extract-min in $O(\log n)$. **Fibonacci heap**: merge in $O(1)$ -- just concatenate the root lists. All other operations maintain their amortized bounds. **Leftist/skew heap**: merge in $O(\log n)$ with a simple recursive algorithm. For applications requiring frequent merges (e.g., Huffman coding, external sorting), binomial or Fibonacci heaps are preferred. For simple priority queue operations without merging, a binary heap suffices. $\square$

---

**Exercise 4.**
Explain why a sorted array can serve as a priority queue with $O(1)$ extract-min but $O(n)$ insert. When is this tradeoff acceptable?

??? success "Solution to Exercise 4"
    In a sorted array, the minimum is always at position 0 (or $n-1$), so extract-min is $O(1)$ (read and remove the first/last element). Insertion requires finding the correct position ($O(\log n)$ via binary search) and shifting elements to make room ($O(n)$). The $O(n)$ insert makes sorted arrays unsuitable for general priority queues. This tradeoff is acceptable when: (1) all elements are known in advance and can be sorted once ($O(n \log n)$), then extracted one by one ($O(1)$ each). Example: processing events in sorted order. (2) Insertions are rare but extract-min is frequent. (3) The array is small enough that $O(n)$ insertion is fast in absolute terms. $\square$

---

**Exercise 5.**
A financial system processes market orders by price-time priority (lowest price first, ties broken by earliest arrival). Design a priority queue that supports insert, extract-min, and cancel (delete by order ID) in $O(\log n)$.

??? success "Solution to Exercise 5"
    Use a **binary heap** augmented with a **hash map** from order ID to heap position. Insert: add to the heap and record the position in the hash map. $O(\log n)$. Extract-min: remove the root, update the hash map. $O(\log n)$. Cancel by ID: look up the position in the hash map ($O(1)$), replace the element with the last element, sift up or down to restore the heap property ($O(\log n)$), remove from the hash map. The composite key is (price, timestamp), ensuring price-time priority. This is the standard design used in financial matching engines. Alternative: use an indexed priority queue (a heap that maintains a position array) for $O(\log n)$ decrease-key and cancel operations without an external hash map. $\square$
