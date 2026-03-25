# In-Place vs Out-of-Place

When choosing a sorting algorithm, time complexity often dominates the discussion, but **space complexity** can be equally important. On embedded systems with limited memory, in server environments processing millions of records, or on GPUs where memory is expensive, the amount of extra space an algorithm requires may determine whether it is feasible at all. This section formalizes the distinction between in-place and out-of-place sorting and examines the trade-offs involved.

## Auxiliary Space

The **auxiliary space** of an algorithm is the extra memory it uses beyond the input itself. This excludes the space occupied by the input array and counts only additional allocations such as temporary arrays, recursion stack frames, and helper data structures.

Auxiliary space differs from **total space**, which includes both the input and any additional memory:

$$
\text{Total space} = \text{Input space} + \text{Auxiliary space}
$$

For sorting $n$ elements, the input space is $\Theta(n)$. The interesting question is how much *additional* space the algorithm requires.

## In-Place Sorting

A sorting algorithm is **in-place** if it uses only $O(1)$ auxiliary space — that is, a constant amount of extra memory regardless of the input size. The algorithm rearranges elements within the original array using swaps and local variables, without allocating a second array.

!!! note "Relaxed Definition"
    Some authors use a relaxed definition that allows $O(\log n)$ auxiliary space, which accounts for the recursion stack in algorithms like quicksort. Under the strict $O(1)$ definition, only iterative algorithms qualify. Throughout this book, we use the relaxed $O(\log n)$ convention unless stated otherwise.

### Examples of In-Place Algorithms

**Insertion sort** maintains a sorted prefix of the array and inserts each new element into its correct position by shifting elements one position to the right. The only extra storage needed is a single temporary variable to hold the element being inserted.

**Heapsort** builds a max-heap in the original array and repeatedly extracts the maximum. The heap structure is maintained using array indices alone, requiring $O(1)$ auxiliary space.

**Quicksort** partitions the array around a pivot element and recursively sorts the two halves. The partitioning is done in place, but the recursion stack requires $O(\log n)$ space in the best and average cases. In the worst case (already sorted input with naive pivot selection), the stack depth grows to $O(n)$.

## Out-of-Place Sorting

A sorting algorithm is **out-of-place** if it requires $\Theta(n)$ or more auxiliary space. Typically, this means allocating a second array of the same size as the input.

### Examples of Out-of-Place Algorithms

**Merge sort** divides the array in half, recursively sorts each half, and merges the two sorted halves into a temporary array. The merge step requires $\Theta(n)$ auxiliary space for the temporary array. Although in-place merge algorithms exist, they are significantly more complex and have worse constant factors.

**Counting sort** allocates a count array of size $k$ (the range of key values) and an output array of size $n$, requiring $\Theta(n + k)$ auxiliary space.

**Radix sort** uses counting sort as a subroutine and inherits its $\Theta(n + k)$ space requirement for each digit pass.

## Space Complexity Classification

The following table classifies common sorting algorithms by their auxiliary space usage.

| Algorithm | Auxiliary Space | Classification |
|-----------|----------------|----------------|
| Bubble sort | $O(1)$ | In-place |
| Selection sort | $O(1)$ | In-place |
| Insertion sort | $O(1)$ | In-place |
| Heapsort | $O(1)$ | In-place |
| Shell sort | $O(1)$ | In-place |
| Quicksort | $O(\log n)$ average | In-place (relaxed) |
| Merge sort | $\Theta(n)$ | Out-of-place |
| Counting sort | $\Theta(n + k)$ | Out-of-place |
| Radix sort | $\Theta(n + k)$ | Out-of-place |
| Bucket sort | $\Theta(n + k)$ | Out-of-place |
| Timsort | $\Theta(n)$ | Out-of-place |

## Trade-Offs

The choice between in-place and out-of-place sorting involves several trade-offs.

### Speed vs Space

Out-of-place algorithms can be faster in practice. Merge sort's $O(n \log n)$ worst-case guarantee is attractive, but it pays for that guarantee with $\Theta(n)$ extra space. Quicksort matches merge sort's average-case performance while using only $O(\log n)$ auxiliary space, but its worst case degrades to $O(n^2)$.

### Stability vs In-Place

Achieving both stability and in-place operation simultaneously is difficult. Among the common $O(n \log n)$ algorithms:

- **Merge sort** is stable but not in-place.
- **Heapsort** is in-place but not stable.
- **Quicksort** is in-place (relaxed) but not stable.

This is one reason Python's Timsort accepts $\Theta(n)$ auxiliary space: it provides both $O(n \log n)$ worst-case time and stability, which are more valuable for a general-purpose library sort than minimizing memory usage.

### Cache Performance

In-place algorithms often have better **cache performance** because they operate on the original array without jumping between separate memory regions. However, this is not always the case: merge sort's sequential access pattern can be cache-friendly despite its out-of-place nature, while heapsort's heap operations exhibit poor locality.

!!! warning "Destructive In-Place Sorting"
    In-place sorting modifies the input array. If the original order must be preserved (e.g., for later use or rollback), the caller must make a copy before sorting. This hidden cost can negate the space savings of an in-place algorithm.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 8.
