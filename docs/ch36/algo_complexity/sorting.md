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
