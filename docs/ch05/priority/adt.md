# Priority Queue ADT

A regular queue serves elements in the order they arrive.  Many algorithms, however, need to process the most urgent or smallest element first, regardless of arrival time.  Dijkstra's shortest-path algorithm extracts the vertex with the smallest tentative distance; an operating system scheduler picks the highest-priority task; and a Huffman encoder repeatedly merges the two least-frequent symbols.  A **priority queue** is the abstract data type that supports this pattern: it allows inserting elements with associated priorities and efficiently retrieving the element with the extreme (minimum or maximum) priority.

## Definition

A **priority queue** is a collection $S$ of elements, each associated with a **key** (priority).  The ADT comes in two flavors:

- **Min-priority queue**: the element with the *smallest* key is served first.
- **Max-priority queue**: the element with the *largest* key is served first.

The two variants are symmetric.  We present the min-priority queue below; the max variant simply reverses the comparisons.

## Operations

A min-priority queue must support the following operations:

| Operation | Description | Typical complexity |
|---|---|---|
| `insert(x, k)` | Add element $x$ with key $k$ to the queue | $O(\log n)$ |
| `extract_min()` | Remove and return the element with the smallest key | $O(\log n)$ |
| `find_min()` | Return (without removing) the element with the smallest key | $O(1)$ |
| `decrease_key(x, k')` | Reduce the key of element $x$ to $k' \le k$ | $O(\log n)$ |
| `is_empty()` | Return `True` if the queue is empty | $O(1)$ |
| `size()` | Return the number of elements | $O(1)$ |

!!! note "Complexity depends on implementation"
    The complexities listed above are for the standard binary heap implementation.  Other implementations (Fibonacci heaps, sorted/unsorted arrays) offer different trade-offs.  See the [sorted](sorted.md) and [unsorted](unsorted.md) implementation pages, as well as the [heap preview](heap_preview.md) for details.

!!! warning "Preconditions"
    `extract_min()` and `find_min()` require the queue to be non-empty.  `decrease_key(x, k')` requires $k' \le k$; increasing a key is not supported by this operation.

## Min vs Max Priority Queue

The only difference between a min-priority queue and a max-priority queue is the direction of comparison.  A max-priority queue replaces `extract_min` with `extract_max` and `decrease_key` with `increase_key`.  Any min-priority queue can emulate a max-priority queue by negating all keys on insertion and negating the result on extraction.

## Comparison with Other ADTs

| Feature | Stack | Queue | Priority Queue |
|---|---|---|---|
| Access order | LIFO | FIFO | By priority |
| Insertion | $O(1)$ | $O(1)$ | $O(\log n)$ typical |
| Deletion | $O(1)$ | $O(1)$ | $O(\log n)$ typical |
| Find extreme | $O(n)$ | $O(n)$ | $O(1)$ |

The trade-off is clear: a priority queue pays more for insertion and deletion but gains constant-time access to the extreme element.

??? example "Priority queue operations trace"
    Consider a min-priority queue with the following operations:

    | Step | Operation | Queue contents (key) | Returned |
    |------|-----------|---------------------|----------|
    | 1 | `insert(A, 4)` | {A:4} | — |
    | 2 | `insert(B, 1)` | {A:4, B:1} | — |
    | 3 | `insert(C, 3)` | {A:4, B:1, C:3} | — |
    | 4 | `find_min()` | {A:4, B:1, C:3} | B (key 1) |
    | 5 | `extract_min()` | {A:4, C:3} | B (key 1) |
    | 6 | `decrease_key(A, 2)` | {A:2, C:3} | — |
    | 7 | `extract_min()` | {C:3} | A (key 2) |

    Elements are served in order of their keys, not their insertion order.  After decreasing A's key from 4 to 2, A becomes the new minimum.

## Common Applications

- **Graph algorithms**: Dijkstra's algorithm uses a min-priority queue to select the nearest unvisited vertex.  Prim's MST algorithm uses one to pick the lightest crossing edge.
- **Event-driven simulation**: events are scheduled with timestamps as keys; the next event to process has the smallest timestamp.
- **Huffman encoding**: repeatedly extract the two symbols with the lowest frequencies and merge them.
- **Task scheduling**: an OS scheduler may use a max-priority queue to run the highest-priority process first.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 6. MIT Press.
