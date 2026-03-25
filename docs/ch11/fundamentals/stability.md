# Stability

When sorting a list of records by one field, we often want to preserve the existing order among records that share the same key value. For example, if students are already sorted alphabetically and we re-sort by grade, a **stable** sort keeps students with the same grade in alphabetical order. An **unstable** sort might scramble that alphabetical order arbitrarily. This distinction seems minor at first glance, but it has deep practical consequences for multi-key sorting, database operations, and algorithm composition.

## Formal Definition

A sorting algorithm is **stable** if, whenever two elements $a_i$ and $a_j$ have equal keys and $a_i$ appears before $a_j$ in the input, then $a_i$ also appears before $a_j$ in the output.

More precisely, let $\langle a_1, a_2, \ldots, a_n \rangle$ be the input and let $\text{key}(a_i)$ denote the sort key of element $a_i$. A sorting algorithm produces a permutation $\pi$ such that

$$
\text{key}(a_{\pi(1)}) \leq \text{key}(a_{\pi(2)}) \leq \cdots \leq \text{key}(a_{\pi(n)})
$$

The algorithm is **stable** if, additionally, for all $i < j$ with $\text{key}(a_i) = \text{key}(a_j)$, we have $\pi^{-1}(i) < \pi^{-1}(j)$. In other words, equal-key elements retain their original relative order.

## Why Stability Matters

### Multi-Key Sorting

The most important application of stability is **multi-key sorting**: sorting by a primary key and then by a secondary key. With a stable sort, we can achieve this by sorting twice:

1. First, sort by the secondary key.
2. Then, sort by the primary key using a stable algorithm.

After step 2, elements with the same primary key remain in the order established by step 1 — that is, sorted by the secondary key. This generalizes to any number of keys, and it is exactly the principle behind **radix sort**, which sorts by the least significant digit first and works upward.

!!! example "Sorting Students by Grade, Then Name"
    Consider students with (name, grade) pairs:

    | Input order | Name | Grade |
    |---|---|---|
    | 1 | Alice | B |
    | 2 | Bob | A |
    | 3 | Carol | B |
    | 4 | Dave | A |

    **Stable sort by grade** produces: Bob (A), Dave (A), Alice (B), Carol (B).
    Among the A students, Bob appears before Dave (preserving input order). Among the B students, Alice appears before Carol.

    **Unstable sort by grade** might produce: Dave (A), Bob (A), Carol (B), Alice (B).
    The relative order within each grade group is unpredictable.

### Database Operations

Database systems rely on stability when executing `ORDER BY` with multiple columns. A stable sort ensures that the tie-breaking behavior is predictable and consistent, which is essential for pagination (displaying results across multiple pages).

### Composability

Stable sorts compose well. If a sequence has already been sorted by one criterion, applying a stable sort by a different criterion produces a result sorted by the second criterion with ties broken by the first. Unstable sorts destroy the prior ordering, making composition unreliable.

## Classifying Sorting Algorithms

| Algorithm | Stable | Notes |
|-----------|--------|-------|
| Bubble sort | Yes | Equal elements are never swapped |
| Insertion sort | Yes | Equal elements are not moved past each other |
| Merge sort | Yes | When merging, take from left subarray on ties |
| Counting sort | Yes | By construction, preserves input order |
| Radix sort | Yes | Requires a stable subroutine (typically counting sort) |
| Selection sort | No | Swaps can move equal elements past each other |
| Heapsort | No | Heap extraction does not preserve input order |
| Quicksort | No | Partitioning moves equal elements unpredictably |
| Shell sort | No | Long-range swaps can disrupt relative order |

### Why Some Algorithms Are Unstable

An algorithm becomes unstable when it performs **long-range swaps** that can move an element past another element with an equal key. Consider selection sort: when it finds the minimum and swaps it into position, the swapped element jumps over potentially many equal-key elements.

!!! example "Selection Sort Instability"
    Input: $\langle 3_a, 3_b, 1 \rangle$ (subscripts distinguish equal keys).

    Selection sort finds the minimum ($1$) and swaps it with the first element ($3_a$):

    $\langle 1, 3_b, 3_a \rangle$

    Now $3_b$ appears before $3_a$, but in the input $3_a$ appeared first. The sort is unstable.

### Making Unstable Algorithms Stable

Any sorting algorithm can be made stable by augmenting each key with the element's original index. Instead of comparing keys $k_i$ and $k_j$ alone, compare the pairs $(k_i, i)$ and $(k_j, j)$ lexicographically. Since indices are unique, ties in the original key are broken by input position, guaranteeing stability.

This transformation has a cost: storing the original indices requires $O(n)$ extra space, and the comparisons are slightly more expensive. For this reason, algorithms that are naturally stable are generally preferred when stability is needed.

## Stability in Python

Python's built-in `sorted()` function and the `list.sort()` method both use **Timsort**, which is a stable sorting algorithm. This guarantee is part of the language specification, not just an implementation detail. As a result, the multi-key sorting pattern works reliably in Python:

```python
# Sort by grade (primary), then by name (secondary)
students = [("Carol", "B"), ("Alice", "B"), ("Bob", "A"), ("Dave", "A")]

# Step 1: sort by name (secondary key)
students.sort(key=lambda s: s[0])

# Step 2: sort by grade (primary key) — stable, so name order is preserved
students.sort(key=lambda s: s[1])

# Result: [('Bob', 'A'), ('Dave', 'A'), ('Alice', 'B'), ('Carol', 'B')]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 8.
- Python Documentation. [Sorting HOW TO](https://docs.python.org/3/howto/sorting.html).
