# Sorting Problem

Sorting is one of the most fundamental problems in computer science. Nearly every large software system relies on sorting at some level — databases order records for efficient retrieval, search engines rank results by relevance, and operating systems schedule processes by priority. Understanding the sorting problem precisely is essential because the way we formalize it determines which algorithms are correct, which complexity bounds apply, and which trade-offs matter. This section defines the sorting problem formally, introduces the comparison model, and surveys the basic complexity landscape.

## Formal Definition

The **sorting problem** takes a sequence of $n$ elements and produces a reordering (permutation) of that sequence such that the elements appear in non-decreasing order according to a given ordering relation.

**Input.** A sequence of $n$ elements $\langle a_1, a_2, \ldots, a_n \rangle$.

**Output.** A permutation $\langle a_{\pi(1)}, a_{\pi(2)}, \ldots, a_{\pi(n)} \rangle$ of the input such that

$$
a_{\pi(1)} \leq a_{\pi(2)} \leq \cdots \leq a_{\pi(n)}
$$

Here $\pi$ is a permutation of $\{1, 2, \ldots, n\}$, and $\leq$ is a **total order** on the element type.

## Total Order

A **total order** $\leq$ on a set $S$ is a binary relation satisfying four properties:

1. **Reflexivity.** For all $a \in S$, $a \leq a$.
2. **Antisymmetry.** If $a \leq b$ and $b \leq a$, then $a = b$.
3. **Transitivity.** If $a \leq b$ and $b \leq c$, then $a \leq c$.
4. **Totality.** For all $a, b \in S$, either $a \leq b$ or $b \leq a$.

Totality ensures that every pair of elements is comparable. This is crucial: without totality, some pairs might be incomparable and the notion of a "sorted" sequence would be ill-defined.

!!! example "Common Total Orders"
    - Integers under the usual $\leq$.
    - Strings under lexicographic (dictionary) order.
    - Tuples under lexicographic comparison: compare first components, break ties with second components, and so on.

## Keys and Satellite Data

In practice, each element $a_i$ consists of a **key** used for ordering and **satellite data** that travels with the key. The sorting algorithm compares and rearranges elements based on their keys alone. For example, sorting a list of student records by GPA treats GPA as the key while name, ID, and other fields are satellite data.

This distinction matters when multiple elements share the same key. A sorting algorithm is **stable** if elements with equal keys appear in the output in the same relative order as in the input. Stability is discussed in detail on the [Stability](stability.md) page.

## The Comparison Model

Most classical sorting algorithms operate in the **comparison model**: the only way to determine the relative order of two elements is by comparing them using $\leq$ (or equivalently $<$, $>$, $\geq$). No assumptions are made about the internal structure of the keys.

In this model, every algorithm can be represented as a **decision tree** where each internal node corresponds to a comparison $a_i \leq a_j$ and each leaf corresponds to a particular output permutation. The decision tree model leads to a fundamental lower bound: any comparison-based sorting algorithm must perform at least

$$
\Omega(n \log n)
$$

comparisons in the worst case. This bound is proved on the [Proof](../lower_bound/proof.md) page.

!!! tip "Beyond Comparisons"
    Algorithms like counting sort, radix sort, and bucket sort bypass the $\Omega(n \log n)$ barrier by exploiting the structure of keys (e.g., integers in a known range). These are called **non-comparison-based** sorting algorithms and achieve $O(n)$ time under appropriate assumptions.

## Complexity Landscape

The table below summarizes the time complexity of major sorting algorithms. Here $n$ is the number of elements and $k$ is the range of key values (for integer sorts).

| Algorithm | Best | Average | Worst | In-Place | Stable |
|-----------|------|---------|-------|----------|--------|
| Bubble sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | Yes | Yes |
| Selection sort | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | Yes | No |
| Insertion sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | Yes | Yes |
| Merge sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | No | Yes |
| Heapsort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | Yes | No |
| Quicksort | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | Yes | No |
| Counting sort | $O(n + k)$ | $O(n + k)$ | $O(n + k)$ | No | Yes |
| Radix sort | $O(dn)$ | $O(dn)$ | $O(dn)$ | No | Yes |

The $\Omega(n \log n)$ lower bound means that merge sort, heapsort, and (on average) quicksort are **asymptotically optimal** among comparison-based algorithms. The simple $O(n^2)$ algorithms — bubble, selection, and insertion sort — are useful for small inputs or nearly sorted data but are too slow for large-scale sorting.

## Sorting as a Building Block

Sorting serves as a prerequisite for many other algorithms:

- **Binary search** requires a sorted array and runs in $O(\log n)$ time.
- **Finding duplicates** reduces from $O(n^2)$ with brute force to $O(n \log n)$ by sorting first and scanning adjacent pairs.
- **Computing the median** and other order statistics can be found in $O(n)$ time, but sorting provides them all simultaneously.
- **Greedy algorithms** for scheduling and interval problems typically begin by sorting events by start or finish time.

These applications explain why sorting is studied so extensively: improvements to sorting algorithms cascade into improvements across many domains.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 8.
