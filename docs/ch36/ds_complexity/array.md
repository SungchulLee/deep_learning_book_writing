# Array Complexities

Arrays are the most fundamental data structure in computing. They store elements in
contiguous memory, providing $O(1)$ random access by index -- a property that no other
data structure can match. Understanding the complexity of each array operation is
essential because arrays underlie nearly every other data structure.

## Static Array Operations

A static array has a fixed size determined at allocation. Elements cannot be added or
removed; only the values at existing indices can be modified.

| Operation | Time | Notes |
|---|---|---|
| Access by index | $O(1)$ | Direct memory offset calculation |
| Update by index | $O(1)$ | Same as access |
| Search (unsorted) | $O(n)$ | Must scan all elements |
| Search (sorted) | $O(\log n)$ | Binary search |
| Find min/max | $O(n)$ | Must scan all elements |

The $O(1)$ access time comes from the address formula: element at index $i$ is located
at memory address

$$
\text{base} + i \times \text{element\_size}
$$

This formula requires contiguous memory allocation, which is the defining property of
arrays.

## Dynamic Array Operations

A dynamic array (Python `list`, C++ `vector`, Java `ArrayList`) automatically resizes
when capacity is exceeded, typically by doubling.

| Operation | Average | Worst | Amortized | Notes |
|---|---|---|---|---|
| Access by index | $O(1)$ | $O(1)$ | -- | Same as static |
| Append (push back) | $O(1)$ | $O(n)$ | $O(1)$ | Resize triggers copy |
| Insert at index $i$ | $O(n)$ | $O(n)$ | -- | Shifts $n - i$ elements |
| Delete at index $i$ | $O(n)$ | $O(n)$ | -- | Shifts $n - i$ elements |
| Pop last | $O(1)$ | $O(1)$ | $O(1)$ | No shifting needed |
| Search | $O(n)$ | $O(n)$ | -- | Linear scan |

!!! tip "Why Amortized O(1) for Append"
    When the array is full, doubling the capacity costs $O(n)$ to copy all
    elements. But this happens only after $n$ cheap $O(1)$ appends. Spreading the
    expensive copy over the preceding cheap operations gives $O(1)$ amortized cost
    per append.

## Sorted Array Operations

Maintaining a sorted array enables binary search but makes insertion expensive.

| Operation | Time | Notes |
|---|---|---|
| Search | $O(\log n)$ | Binary search |
| Insert (maintaining order) | $O(n)$ | Find position + shift elements |
| Delete (maintaining order) | $O(n)$ | Find element + shift elements |
| Find min | $O(1)$ | First element |
| Find max | $O(1)$ | Last element |
| Find predecessor/successor | $O(\log n)$ | Binary search variant |
| Merge two sorted arrays | $O(m + n)$ | Two-pointer merge |

## Multidimensional Arrays

A 2D array of size $m \times n$ uses either row-major or column-major layout.

| Operation | Time | Notes |
|---|---|---|
| Access `A[i][j]` | $O(1)$ | Offset = $i \times n + j$ (row-major) |
| Row traversal | $O(n)$ | Cache-friendly in row-major |
| Column traversal | $O(m)$ | Cache-unfriendly in row-major |
| Full traversal | $O(mn)$ | Row-by-row is faster due to caching |
| Transpose | $O(mn)$ | Must visit every element |

!!! warning "Cache Performance"
    Traversing a 2D array column-by-column in a row-major language (C, C++, Python
    NumPy default) causes cache misses at every access. For large matrices, this
    can be 10--100x slower than row-by-row traversal.

## Space Complexity

| Array Type | Space | Notes |
|---|---|---|
| Static array of $n$ elements | $O(n)$ | Exact: $n \times$ element size |
| Dynamic array | $O(n)$ | Up to $2n$ allocated due to doubling |
| 2D array $m \times n$ | $O(mn)$ | Contiguous or array of pointers |
| Sparse array (hash map) | $O(k)$ | $k$ = number of non-zero elements |

## Common Array Algorithms and Their Complexities

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Prefix sum (build) | $O(n)$ | $O(n)$ | Enables $O(1)$ range sum queries |
| Prefix sum (query) | $O(1)$ | -- | $\text{sum}(l, r) = P[r+1] - P[l]$ |
| Kadane's algorithm | $O(n)$ | $O(1)$ | Maximum subarray sum |
| Dutch National Flag | $O(n)$ | $O(1)$ | 3-way partition |
| Rotate by $k$ | $O(n)$ | $O(1)$ | Three reversals trick |
| Remove duplicates (sorted) | $O(n)$ | $O(1)$ | Two pointers |

## Comparison with Other Structures

| Operation | Array | Linked List | Hash Table | BST (balanced) |
|---|---|---|---|---|
| Access by index | $O(1)$ | $O(n)$ | -- | -- |
| Search | $O(n)$ | $O(n)$ | $O(1)$ avg | $O(\log n)$ |
| Insert at end | $O(1)$ amort. | $O(1)$ | $O(1)$ avg | $O(\log n)$ |
| Insert at beginning | $O(n)$ | $O(1)$ | -- | $O(\log n)$ |
| Delete | $O(n)$ | $O(1)$ given pointer | $O(1)$ avg | $O(\log n)$ |
| Ordered traversal | $O(n)$ | $O(n)$ | $O(n \log n)$ sort | $O(n)$ |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.
