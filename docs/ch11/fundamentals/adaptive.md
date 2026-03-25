# Adaptive Sorting

In many practical scenarios, data arrives nearly sorted. Log files are ordered by timestamp except for occasional late entries. A list that was sorted yesterday needs only minor adjustments after a few insertions. Version-controlled records change incrementally. An algorithm that ignores this existing order and performs the same work regardless wastes effort. **Adaptive** sorting algorithms detect and exploit existing order in the input, completing faster when the data is partially sorted. This section defines adaptivity formally, introduces measures of presortedness, and examines how specific algorithms achieve adaptive behavior.

## What Makes an Algorithm Adaptive

A sorting algorithm is **adaptive** if its running time decreases as the input becomes more sorted. More precisely, let $m$ be a measure of the "disorder" or "unsortedness" of the input. An algorithm is adaptive with respect to $m$ if its running time is a function of both $n$ (the number of elements) and $m$, with the running time decreasing as $m$ decreases.

A **non-adaptive** algorithm always performs the same number of operations regardless of the input order. Selection sort, for example, always makes

$$
\frac{n(n-1)}{2}
$$

comparisons whether the input is already sorted, reverse sorted, or randomly permuted.

## Measures of Presortedness

Several measures quantify how far an input is from being sorted. Each captures a different aspect of disorder.

### Inversions

An **inversion** is a pair of indices $(i, j)$ with $i < j$ and $a_i > a_j$. The number of inversions, denoted $\text{Inv}(A)$, ranges from $0$ (sorted) to

$$
\binom{n}{2} = \frac{n(n-1)}{2}
$$

(reverse sorted). This is the most widely used measure of presortedness.

!!! example "Counting Inversions"
    For the sequence $\langle 2, 4, 1, 3, 5 \rangle$, the inversions are:
    $(2, 1)$, $(4, 1)$, $(4, 3)$ — so $\text{Inv}(A) = 3$.

    For the sorted sequence $\langle 1, 2, 3, 4, 5 \rangle$, $\text{Inv}(A) = 0$.

### Runs

A **run** is a maximal contiguous subsequence that is already sorted. The number of runs, denoted $\text{Runs}(A)$, ranges from $1$ (fully sorted) to $n$ (every consecutive pair is out of order, e.g., a strictly decreasing sequence). Timsort and natural merge sort are designed around this measure.

!!! example "Counting Runs"
    For $\langle 1, 3, 5, 2, 4, 6 \rangle$: the runs are $\langle 1, 3, 5 \rangle$ and $\langle 2, 4, 6 \rangle$, so $\text{Runs}(A) = 2$.

    For $\langle 5, 4, 3, 2, 1 \rangle$: every element starts a new ascending run, so $\text{Runs}(A) = 5$ (or $1$ if descending runs are also detected, as in Timsort).

### Displacement

The **displacement** of element $a_i$ is $|i - \sigma(i)|$, where $\sigma(i)$ is the position of $a_i$ in the sorted output. The maximum displacement $\text{Dis}(A) = \max_i |i - \sigma(i)|$ measures how far any element is from its correct position. If $\text{Dis}(A) = k$, then each element needs to move at most $k$ positions, and an algorithm can exploit this locality.

### Removals

The minimum number of elements that must be removed so that the remaining elements are sorted is $n - \text{LIS}(A)$, where $\text{LIS}(A)$ is the length of the longest increasing subsequence. A nearly sorted sequence has a long LIS and requires few removals.

## Adaptive Algorithms

### Insertion Sort: Adaptive via Inversions

Insertion sort performs exactly $\text{Inv}(A)$ swaps (or shifts) plus $n - 1$ comparisons in the outer loop. Its total running time is

$$
\Theta(n + \text{Inv}(A))
$$

When the input is already sorted, $\text{Inv}(A) = 0$ and insertion sort runs in $\Theta(n)$. When the input is reverse sorted, $\text{Inv}(A) = n(n-1)/2$ and insertion sort runs in $\Theta(n^2)$. This smooth interpolation between $O(n)$ and $O(n^2)$ makes insertion sort one of the most naturally adaptive algorithms.

### Timsort: Adaptive via Runs

Python's built-in sort (Timsort) exploits existing runs. It scans the array for natural runs (both ascending and descending), extends short runs using insertion sort, and merges runs using a carefully designed merge strategy. On an input with $r$ runs, Timsort performs

$$
O(n \log r)
$$

comparisons. When the input is already sorted ($r = 1$), Timsort runs in $O(n)$. When the input has many short runs ($r = \Theta(n)$), it performs $O(n \log n)$ comparisons, matching optimal worst-case behavior.

### Natural Merge Sort: Adaptive via Runs

Natural merge sort identifies existing sorted runs in the input and merges them pairwise. Like Timsort, it runs in $O(n \log r)$ time where $r$ is the number of runs, but it lacks Timsort's run-extension and galloping optimizations.

### Bubble Sort: Weakly Adaptive

Bubble sort with an early-termination check (stop if no swaps occur in a pass) is adaptive in a limited sense: it terminates in $O(n)$ on already-sorted input. However, even a single out-of-place element can require $O(n)$ passes, so bubble sort is not as smoothly adaptive as insertion sort.

## Adaptive vs Non-Adaptive Comparison

| Algorithm | Adaptive? | Best Case | Worst Case | Adapts To |
|-----------|-----------|-----------|------------|-----------|
| Insertion sort | Yes | $O(n)$ | $O(n^2)$ | Inversions |
| Timsort | Yes | $O(n)$ | $O(n \log n)$ | Runs |
| Natural merge sort | Yes | $O(n)$ | $O(n \log n)$ | Runs |
| Bubble sort (optimized) | Weakly | $O(n)$ | $O(n^2)$ | Sorted input |
| Shell sort | Partially | $O(n \log n)$ | $O(n^{3/2})$ | Depends on gap sequence |
| Selection sort | No | $O(n^2)$ | $O(n^2)$ | None |
| Heapsort | No | $O(n \log n)$ | $O(n \log n)$ | None |
| Standard merge sort | No | $O(n \log n)$ | $O(n \log n)$ | None |

!!! tip "When Adaptivity Matters"
    Adaptivity is most valuable when:

    - Data is frequently re-sorted after small modifications (e.g., maintaining a sorted list with occasional insertions).
    - Data arrives from multiple sorted streams that need to be merged.
    - The distribution of inputs is skewed toward nearly sorted configurations.

    When input is truly random, adaptive and non-adaptive algorithms perform similarly, so adaptivity provides no advantage.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 8.
- Estivill-Castro, V., & Wood, D. (1992). A survey of adaptive sorting algorithms. *ACM Computing Surveys*, 24(4), 441--476.
