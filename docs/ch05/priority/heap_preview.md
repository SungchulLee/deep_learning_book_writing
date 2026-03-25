# Heap Preview

The [priority queue ADT](adt.md) requires $O(\log n)$ insertion and extraction of the minimum (or maximum) element.  An [unsorted list](unsorted.md) achieves $O(1)$ insertion but $O(n)$ extraction; a [sorted list](sorted.md) achieves $O(1)$ extraction but $O(n)$ insertion.  A **binary heap** is the data structure that balances both operations at $O(\log n)$ each, making it the standard priority queue implementation.  This page previews the key ideas behind binary heaps; the full treatment appears in the [heap chapter](../../ch09/binary/property.md).

## The Heap Property

A **min-heap** is a complete binary tree in which every node's key is less than or equal to the keys of its children.  Formally, for every non-root node $i$ with parent $p(i)$:

$$
\text{key}(p(i)) \le \text{key}(i)
$$

A **max-heap** reverses the inequality: every parent's key is greater than or equal to its children's keys.

The heap property ensures that the root always holds the minimum (in a min-heap) or maximum (in a max-heap) element, so `find_min()` or `find_max()` runs in $O(1)$.

## Array Representation

A complete binary tree of $n$ nodes can be stored in a flat array $A[0 \ldots n-1]$ without explicit pointers.  For a node at index $i$:

- **Parent**: $\lfloor (i - 1) / 2 \rfloor$
- **Left child**: $2i + 1$
- **Right child**: $2i + 2$

This mapping is compact (no wasted space) and cache-friendly (elements are stored contiguously in memory).

??? example "Array layout of a min-heap"
    Consider the min-heap with keys $[1, 3, 2, 7, 5, 4, 6]$:

    ```
             1
           /   \
          3     2
         / \   / \
        7   5 4   6
    ```

    The array representation is simply `[1, 3, 2, 7, 5, 4, 6]`.  Node at index 0 (key 1) is the root.  Its children are at indices 1 (key 3) and 2 (key 2).  Node at index 1 has children at indices 3 (key 7) and 4 (key 5).

## Key Operations

With the array layout in place, the heap supports four core operations, each relying on a simple "sift" procedure that walks up or down the tree.

| Operation | Idea | Time |
|---|---|---|
| `insert(x)` | Place $x$ at the end; **sift up** to restore the heap property | $O(\log n)$ |
| `extract_min()` | Swap root with last element; remove last; **sift down** the new root | $O(\log n)$ |
| `find_min()` | Return the root | $O(1)$ |
| `build_heap(A)` | Apply sift-down from the last internal node to the root | $O(n)$ |

The $O(n)$ build-heap result is a key theoretical result: a heap can be constructed from an unsorted array in linear time, not $O(n \log n)$.  The intuition is that most nodes live near the bottom of the tree and require only a few swaps during sift-down, while the few nodes near the top that require many swaps are vastly outnumbered.

!!! tip "Sift up and sift down"
    **Sift up** (also called "bubble up" or "percolate up") repeatedly swaps a node with its parent until the heap property is restored.  **Sift down** (also called "heapify down") repeatedly swaps a node with its smaller child.  Each traverses at most the height of the tree, which is $\lfloor \log_2 n \rfloor$ for a complete binary tree.

## Python's heapq Module

Python's standard library provides a min-heap through the `heapq` module.  It operates directly on a regular list:

```python
"""Demonstration of Python's heapq module for priority queue operations."""

import heapq


# === Priority Queue with heapq ===

if __name__ == "__main__":
    # Build a heap from an unsorted list
    data = [5, 3, 8, 1, 9, 2]
    heapq.heapify(data)  # O(n) in-place
    print(f"Heapified: {data}")  # [1, 3, 2, 5, 9, 8]

    # Insert a new element
    heapq.heappush(data, 4)
    print(f"After push(4): {data}")

    # Extract elements in sorted order
    sorted_output = []
    while data:
        sorted_output.append(heapq.heappop(data))
    print(f"Extracted in order: {sorted_output}")
    # Output: [1, 2, 3, 4, 5, 8, 9]
```

!!! note "Min-heap only"
    Python's `heapq` provides only a min-heap.  To emulate a max-heap, negate the keys on insertion and negate the result on extraction.

## Complexity Summary

| Implementation | `insert` | `extract_min` | `find_min` | `build` |
|---|---|---|---|---|
| Unsorted array | $O(1)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| Sorted array | $O(n)$ | $O(1)$ | $O(1)$ | $O(n \log n)$ |
| **Binary heap** | $O(\log n)$ | $O(\log n)$ | $O(1)$ | $O(n)$ |

The binary heap provides the best balance across all operations, which is why it is the default choice for priority queues.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 6. MIT Press.
