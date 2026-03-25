# Array vs Linked List

Arrays and linked lists are the two most fundamental ways to store a
sequence of elements. Every more complex data structure -- stacks, queues,
hash tables, trees -- is ultimately built on one of these two primitives.
Choosing between them requires understanding their structural differences and
how those differences affect the cost of common operations. This page
provides a systematic comparison to guide that choice.

## Memory Layout

The key structural difference is how elements are stored in memory:

- **Array**: Elements occupy a contiguous block of memory. Element $i$ sits
  at address $\text{base} + i \times \text{element\_size}$. This formula
  enables $O(1)$ random access.
- **Linked list**: Each element is stored in a separate node allocated
  independently. Nodes may be scattered anywhere in memory. Each node
  carries a pointer to the next node (and possibly the previous node).

This difference in layout drives nearly every performance trade-off between
the two structures.

## Operation Complexity Comparison

| Operation | Array | Singly linked list | Doubly linked list |
|---|---|---|---|
| Access by index | $O(1)$ | $O(n)$ | $O(n)$ |
| Search (unsorted) | $O(n)$ | $O(n)$ | $O(n)$ |
| Search (sorted) | $O(\log n)$ (binary search) | $O(n)$ | $O(n)$ |
| Insert at front | $O(n)$ (shift elements) | $O(1)$ | $O(1)$ |
| Insert at back | $O(1)$ amortized (dynamic) | $O(n)$ or $O(1)$ with tail | $O(1)$ with tail |
| Insert at position $i$ | $O(n)$ (shift elements) | $O(n)$ (traversal) | $O(n)$ (traversal) |
| Delete at front | $O(n)$ (shift elements) | $O(1)$ | $O(1)$ |
| Delete at back | $O(1)$ | $O(n)$ (need predecessor) | $O(1)$ with tail |
| Delete given node | $O(n)$ (shift + find) | $O(n)$ (need predecessor) | $O(1)$ |

The table reveals a pattern: arrays excel at **random access**, while
linked lists excel at **insertion and deletion** when the position is
already known.

## Memory Overhead

Each structure carries different overhead per element:

| Structure | Storage per element |
|---|---|
| Array (static) | Data only |
| Dynamic array | Data + unused capacity (amortized) |
| Singly linked list | Data + 1 pointer |
| Doubly linked list | Data + 2 pointers |

For small data types (integers, characters), the pointer overhead in linked
lists can exceed the data size itself. A doubly linked list of 32-bit
integers on a 64-bit system uses 20 bytes per element (4 for data + 16 for
two pointers), compared to 4 bytes per element in an array -- a $5\times$
overhead.

Dynamic arrays also waste memory through over-allocation. A typical
growth factor of 2 means that, on average, 25% of the allocated capacity
is unused. However, this overhead is per-array rather than per-element,
making it far more efficient for large collections.

## Structural Flexibility

Linked lists offer structural operations that arrays cannot efficiently
support:

- **Splitting**: A linked list can be split at any node in $O(1)$ time by
  redirecting one pointer. Splitting an array requires copying half the
  elements, taking $O(n)$.
- **Merging**: Two linked lists can be merged in $O(1)$ time by connecting
  one tail to the other head. Merging arrays requires allocating a new
  array and copying all elements, taking $O(n + m)$.
- **Persistent modification**: In a linked list, old versions of the
  structure can share nodes with new versions (structural sharing). Arrays
  must be fully copied to create a snapshot.

## When to Choose Each

**Choose arrays when:**

- Random access by index is frequent.
- The collection size is known in advance or changes rarely.
- Elements are small (minimizing the benefit of per-element allocation).
- Cache performance matters (see [Cache Performance](cache.md)).
- Binary search on sorted data is needed.

**Choose linked lists when:**

- Insertions and deletions at arbitrary positions dominate the workload.
- The collection size fluctuates unpredictably and large contiguous
  allocations are impractical.
- $O(1)$ splitting and merging are required.
- Elements are large, making pointer overhead negligible relative to data
  size.

In practice, arrays and dynamic arrays are the default choice for most
applications. Linked lists are reserved for specialized scenarios where
their structural advantages justify the pointer overhead and cache penalty.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
