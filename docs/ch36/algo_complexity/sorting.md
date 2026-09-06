# Sorting Complexities

Sorting is the single most studied problem in algorithm design. Knowing the best, average,
and worst-case complexity of each sorting algorithm, along with its stability and space
requirements, lets you choose the right tool for every situation -- from small arrays
where simplicity matters to massive datasets where asymptotic guarantees dominate.

## Comparison-Based Sorts

Comparison sorts determine order by pairwise element comparisons. The decision-tree
lower bound proves that any comparison sort must use at least

$$
\Omega(n \log n)
$$

comparisons in the worst case.

| Algorithm | Best | Average | Worst | Space | Stable | Method |
|---|---|---|---|---|---|---|
| Bubble sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | Yes | Exchanging |
| Selection sort | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | No | Selection |
| Insertion sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | Yes | Insertion |
| Shell sort | $O(n \log n)$ | depends on gaps | $O(n^{3/2})$ | $O(1)$ | No | Insertion |
| Merge sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes | Merging |
| Quick sort | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ | No | Partitioning |
| Heap sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ | No | Selection |
| Timsort | $O(n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes | Hybrid |
| Introsort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(\log n)$ | No | Hybrid |

!!! tip "Why Quick Sort Dominates in Practice"
    Despite its $O(n^2)$ worst case, randomized quick sort achieves $O(n \log n)$
    expected time with small constants due to cache-friendly sequential access and
    low overhead. Median-of-three pivot selection makes the worst case exceedingly
    unlikely.

## Non-Comparison Sorts

These algorithms bypass the $\Omega(n \log n)$ lower bound by exploiting the structure
of keys (integers, strings) rather than using pairwise comparisons.

| Algorithm | Time | Space | Stable | Constraint |
|---|---|---|---|---|
| Counting sort | $O(n + k)$ | $O(n + k)$ | Yes | Keys in range $[0, k)$ |
| Radix sort (LSD) | $O(d(n + k))$ | $O(n + k)$ | Yes | $d$ digits, base $k$ |
| Radix sort (MSD) | $O(d(n + k))$ | $O(n + k)$ | Yes | Same as LSD |
| Bucket sort | $O(n + k)$ expected | $O(n + k)$ | Yes | Uniform distribution |

Here $k$ is the range of digit values and $d$ is the number of digits. For $w$-bit
integers sorted in base $n$, radix sort runs in $O(n \cdot w / \log n)$ time.

## The Lower Bound Argument

Any comparison-based sorting algorithm corresponds to a binary decision tree. The tree
must have at least $n!$ leaves (one per permutation). The minimum height of a binary
tree with $n!$ leaves is:

$$
h \ge \lceil \log_2(n!) \rceil = \Theta(n \log n)
$$

The second equality follows from Stirling's approximation:
$\log_2(n!) = n \log_2 n - n \log_2 e + O(\log n)$.

## Space Complexity Details

Sorting space complexity often determines which algorithm is feasible.

| Algorithm | Auxiliary Space | In-place? | Notes |
|---|---|---|---|
| Merge sort | $O(n)$ | No | Needs temporary array for merging |
| Quick sort | $O(\log n)$ | Yes | Stack space for recursion |
| Heap sort | $O(1)$ | Yes | Builds heap in-place |
| Timsort | $O(n)$ | No | Needs merge buffer |
| Radix sort | $O(n + k)$ | No | Needs output and count arrays |
| Block merge sort | $O(1)$ | Yes | Stable, in-place, $O(n \log n)$ |

!!! warning "Quick Sort Stack Depth"
    Naive quick sort can use $O(n)$ stack space on already-sorted input. Tail-call
    optimization (recurring on the smaller partition) guarantees $O(\log n)$ space
    even in the worst case for time.

## Adaptive Sorts

Adaptive algorithms exploit existing order in the input. Timsort, used by Python and
Java, detects pre-sorted runs and merges them efficiently.

| Algorithm | Best (nearly sorted) | Worst | Adaptive? |
|---|---|---|---|
| Insertion sort | $O(n)$ | $O(n^2)$ | Yes -- inversions |
| Timsort | $O(n)$ | $O(n \log n)$ | Yes -- natural runs |
| Smooth sort | $O(n)$ | $O(n \log n)$ | Yes -- Leonardo heaps |
| Shell sort | $O(n \log n)$ | $O(n^{3/2})$ | Partially |

## Practical Feasibility Guide

| $n$ | $O(n^2)$ | $O(n \log n)$ | $O(n)$ (linear) |
|---|---|---|---|
| $10^3$ | fast | instant | instant |
| $10^4$ | moderate | fast | fast |
| $10^5$ | slow | fast | fast |
| $10^6$ | infeasible | fast | fast |
| $10^7$ | infeasible | moderate | fast |
| $10^8$ | infeasible | slow | moderate |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.

## Exercises

**Exercise 1.**
Prove that any comparison-based sorting algorithm requires $\Omega(n \log n)$ comparisons in the worst case.

??? success "Solution to Exercise 1"
    A comparison-based sort can be modeled as a decision tree where each internal node is a comparison and each leaf is a permutation of the input. There are $n!$ possible permutations, so the tree must have at least $n!$ leaves. The height of a binary tree with $L$ leaves is at least $\lceil \log_2 L \rceil$. Therefore, worst-case comparisons $\ge \log_2(n!) = \sum_{i=1}^{n} \log_2 i \ge \int_1^n \log_2 x \, dx = n \log_2 n - n/\ln 2 = \Omega(n \log n)$. This means no comparison sort can beat $O(n \log n)$ in the worst case. Merge sort and heap sort achieve this bound. $\square$

---

**Exercise 2.**
Counting sort runs in $O(n + k)$ where $k$ is the range of values. Explain why it bypasses the $\Omega(n \log n)$ lower bound and when it is practical.

??? success "Solution to Exercise 2"
    Counting sort is not comparison-based: it uses the values themselves as array indices, counting occurrences and computing prefix sums. The $\Omega(n \log n)$ lower bound applies only to comparison-based sorts. Counting sort is practical when $k = O(n)$ (e.g., sorting $n$ integers in $[0, n]$, sorting characters). It becomes impractical when $k \gg n$ (e.g., sorting 1000 64-bit integers: $k = 2^{64}$, requiring an impossibly large count array). Radix sort extends counting sort to large keys by sorting digit-by-digit, achieving $O(d(n + b))$ where $d$ is the number of digits and $b$ is the base. $\square$

---

**Exercise 3.**
Compare quicksort, mergesort, and heapsort in terms of average time, worst-case time, space, and stability.

??? success "Solution to Exercise 3"
    | Property | Quicksort | Mergesort | Heapsort |
    |---|---|---|---|
    | Average time | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ |
    | Worst-case time | $O(n^2)$ | $O(n \log n)$ | $O(n \log n)$ |
    | Space | $O(\log n)$ stack | $O(n)$ | $O(1)$ |
    | Stable | No | Yes | No |
    | Cache perf. | Excellent | Good | Poor |

    Quicksort is fastest in practice (best cache behavior, low constant factors) despite $O(n^2)$ worst case (mitigated by randomized pivot). Mergesort is used when stability is required (Python's Timsort, Java's Arrays.sort for objects). Heapsort is used when $O(1)$ extra space and guaranteed $O(n \log n)$ are both needed. $\square$

---

**Exercise 4.**
Timsort (Python's default sort) achieves $O(n)$ on nearly-sorted data. Explain how it exploits existing order (runs).

??? success "Solution to Exercise 4"
    Timsort identifies maximal ascending or descending runs in the input. Descending runs are reversed to become ascending. Short runs (below a minimum run length, typically 32--64) are extended using insertion sort, which is $O(k^2)$ but fast for small $k$ due to low overhead. The resulting sorted runs are merged using a modified merge sort with a merge policy that maintains a stack of runs and merges them when certain size invariants are violated. For nearly-sorted data: the entire input may be one or two runs, so Timsort performs $O(n)$ work (one scan + zero or one merge). For random data: $\sim n/\text{minrun}$ runs are created, each extended to minrun by insertion sort, then merged in $O(n \log(n/\text{minrun})) = O(n \log n)$. $\square$

---

**Exercise 5.**
A financial system sorts 10 million trade records by timestamp every minute. Each batch is 99% already sorted (only a few new trades are out of order). Which sorting algorithm minimizes wall-clock time?

??? success "Solution to Exercise 5"
    For nearly-sorted data with $k$ out-of-order elements: **Insertion sort** runs in $O(nk)$ -- for $k \ll n$, this is nearly $O(n)$. With 1% out of order ($k = 100{,}000$): $10^7 \times 10^5 = 10^{12}$ -- too slow. **Timsort** identifies the nearly-sorted structure and runs in $O(n + k \log k)$ -- detect the long sorted run, insertion-sort the few unsorted elements, merge. For this workload: $\sim 10^7 + 10^5 \times 17 \approx 1.2 \times 10^7$ operations ($\sim$0.1 seconds). **Merge the new trades**: maintain a sorted array, collect new trades separately, sort the new trades ($O(k \log k)$), and merge the two sorted lists ($O(n)$). Total: $O(n + k \log k) \approx O(n)$. This is the fastest approach for this specific workload pattern. $\square$
