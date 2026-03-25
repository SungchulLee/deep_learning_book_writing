# Access Patterns

The best data structure for a problem depends not just on which operations
are needed, but on how those operations are interleaved -- the **access
pattern**. A workload that reads elements by index thousands of times
between modifications demands a different structure than one that
alternates between insertions and deletions at both ends. This page
catalogs common access patterns and matches each one to the data structure
that serves it most efficiently.

## Sequential Access

**Pattern**: Elements are visited one after another, from beginning to end
(or end to beginning).

**Examples**: Summing all elements, printing a list, computing a running
average, streaming data processing.

**Best structure**: Both arrays and linked lists support sequential access
in $O(n)$, but arrays are significantly faster in practice due to cache
locality (see [Cache Performance](cache.md)). Prefer arrays unless
structural modifications occur during the traversal.

## Random Access

**Pattern**: Elements are accessed by index in unpredictable order.

**Examples**: Binary search, hash table probing, matrix operations,
accessing elements by computed indices.

**Best structure**: Arrays provide $O(1)$ access by index. Linked lists
require $O(n)$ traversal to reach position $i$, making them unsuitable for
random access workloads.

## Stack Pattern (LIFO)

**Pattern**: Elements are only added and removed at one end.

**Examples**: Function call stacks, undo operations, expression parsing,
depth-first search.

**Best structure**: Both arrays (using the end as the stack top) and singly
linked lists (using the head as the stack top) provide $O(1)$ push and pop.
Arrays are preferred for their cache behavior unless the maximum size is
truly unpredictable.

## Queue Pattern (FIFO)

**Pattern**: Elements are added at one end and removed from the other.

**Examples**: Breadth-first search, task scheduling, print queues, message
buffers.

**Best structure**: A circular array or a singly linked list with a tail
pointer both provide $O(1)$ enqueue and dequeue. For fixed-capacity queues,
a circular array is more cache-friendly. For unbounded queues, a linked
list avoids the cost of resizing.

## Deque Pattern (Double-Ended)

**Pattern**: Elements are added and removed at both ends.

**Examples**: Sliding window maximum, work-stealing schedulers, palindrome
checking.

**Best structure**: A circular array (deque) or a doubly linked list both
provide $O(1)$ operations at both ends. Python's `collections.deque` uses
a block-based design that combines array cache locality with $O(1)$
double-ended operations.

## Frequent Insertion and Deletion

**Pattern**: Elements are frequently inserted or removed at arbitrary
positions, with the position determined by traversal rather than index.

**Examples**: Text editors (insert/delete at cursor), LRU caches (evict
least recently used), maintaining sorted order by pointer rearrangement.

**Best structure**: Linked lists provide $O(1)$ insertion and deletion
once the position is found. Arrays require $O(n)$ element shifting for
mid-sequence modifications.

## Sorted Order Maintenance

**Pattern**: Elements must remain in sorted order through a mix of
insertions, deletions, and lookups.

**Examples**: Priority queues, ordered dictionaries, database indices.

**Best structure**: Neither plain arrays nor linked lists are ideal.
Sorted arrays support $O(\log n)$ search (binary search) but $O(n)$
insertion. Sorted linked lists support $O(1)$ insertion after finding the
position but $O(n)$ search. For this pattern, balanced BSTs ($O(\log n)$
for all operations) or skip lists are superior.

## Pattern Summary Table

| Access pattern | Best array variant | Best linked variant | Winner |
|---|---|---|---|
| Sequential | Static/dynamic array | Any linked list | Array (cache) |
| Random access | Static/dynamic array | -- | Array |
| Stack (LIFO) | Dynamic array | Singly linked | Array (cache) |
| Queue (FIFO) | Circular array | Singly + tail ptr | Depends |
| Deque | Circular array | Doubly linked | Depends |
| Frequent insert/delete | -- | Doubly linked | Linked list |
| Sorted maintenance | -- | -- | BST / skip list |

!!! tip "Decision heuristic"
    Start with an array. Switch to a linked list only when the workload is
    dominated by insertions and deletions at positions already located by
    pointer, and random access is not needed. Switch to a tree or skip list
    when sorted order must be maintained dynamically.

## Hybrid Approaches

Many real-world systems combine arrays and linked lists to match complex
access patterns:

- **Hash map with chaining**: An array of bucket heads, each pointing to a
  linked list of colliding entries. The array provides $O(1)$ bucket
  lookup; the linked list handles collisions.
- **LRU cache**: A hash map (array-based) for $O(1)$ key lookup combined
  with a doubly linked list for $O(1)$ eviction ordering.
- **Unrolled linked list**: A linked list of small arrays, combining
  sequential cache locality with linked-list flexibility.
- **B-tree**: Each node is an array of keys, and nodes are linked by
  child pointers, optimizing for disk and cache line access patterns.

These hybrids demonstrate that arrays and linked lists are not always
competing choices but often complementary building blocks.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
