# Merge K Sorted Lists

Merging $k$ sorted lists into a single sorted output is a fundamental operation in external sorting, database query processing, and distributed systems where data arrives pre-sorted in chunks. The naive approach of repeatedly picking the minimum from all $k$ list heads costs $O(nk)$ comparisons. A min-heap reduces this to $O(n \log k)$ by maintaining only the $k$ current candidates, where $n$ is the total number of elements across all lists.

## Problem Statement

Given $k$ sorted lists with a combined total of $n$ elements, produce a single sorted list containing all $n$ elements.

## Heap-Based Algorithm

The idea is to maintain a min-heap of size $k$ that holds one element from each list. At each step, extract the minimum from the heap (the globally smallest unprocessed element), append it to the output, and push the next element from the same list that produced the minimum.

### Algorithm

1. Initialize a min-heap with the first element from each of the $k$ lists, along with metadata tracking which list and position each element came from.
2. While the heap is not empty:
    - Extract the minimum element from the heap and append it to the output.
    - If the source list has more elements, push the next element from that list into the heap.
3. Return the output list.

### Why O(n log k)?

Each of the $n$ total elements is inserted into and extracted from the heap exactly once. Each heap operation costs $O(\log k)$ since the heap contains at most $k$ elements. The total cost is therefore:

$$
T(n, k) = n \cdot O(\log k) = O(n \log k)
$$

## Comparison of Approaches

| Approach | Time | Space |
|----------|------|-------|
| Compare all $k$ heads at each step | $O(nk)$ | $O(1)$ |
| Min-heap of size $k$ | $O(n \log k)$ | $O(k)$ |
| Divide-and-conquer pairwise merge | $O(n \log k)$ | $O(n)$ |

The heap approach and the divide-and-conquer approach have the same asymptotic time, but the heap approach uses only $O(k)$ extra space (plus $O(n)$ for the output) and processes elements in a streaming fashion.

## Implementation

```python
"""
Merge k sorted lists using a min-heap.

Demonstrates the O(n log k) heap-based k-way merge algorithm.
"""

import heapq


# === From-Scratch Implementation ===

def merge_k_sorted(lists):
    """Merge k sorted lists into one sorted list using a min-heap.

    Each heap entry is (value, list_index, element_index) to break
    ties deterministically and track the source of each element.

    Time: O(n log k), Space: O(k) for the heap.
    """
    result = []
    heap = []

    # Initialize heap with the first element from each list
    for i, lst in enumerate(lists):
        if lst:
            # (value, list_index, element_index)
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # Push next element from the same list
        next_idx = elem_idx + 1
        if next_idx < len(lists[list_idx]):
            next_val = lists[list_idx][next_idx]
            heapq.heappush(heap, (next_val, list_idx, next_idx))

    return result


# === Using heapq.merge ===

def merge_k_heapq(lists):
    """Merge k sorted iterables using Python's heapq.merge."""
    return list(heapq.merge(*lists))


# === Demonstration ===

if __name__ == "__main__":
    lists = [
        [1, 4, 7, 10],
        [2, 5, 8, 11],
        [3, 6, 9, 12],
    ]

    print("Input lists:")
    for i, lst in enumerate(lists):
        print(f"  List {i}: {lst}")

    result = merge_k_sorted(lists)
    print(f"\nMerged (manual): {result}")

    result2 = merge_k_heapq(lists)
    print(f"Merged (heapq):  {result2}")

    # Unequal length lists
    print("\n--- Unequal Length Lists ---")
    lists2 = [
        [1, 3, 5, 7, 9, 11],
        [2, 4],
        [6, 8, 10],
        [],
        [0, 12, 13, 14],
    ]

    for i, lst in enumerate(lists2):
        print(f"  List {i}: {lst}")

    result3 = merge_k_sorted(lists2)
    print(f"\nMerged: {result3}")

    # Verify sorted
    assert result3 == sorted(result3), "Result is not sorted!"
    print("Correctness verified.")
```

**Output:**
```
Input lists:
  List 0: [1, 4, 7, 10]
  List 1: [2, 5, 8, 11]
  List 2: [3, 6, 9, 12]

Merged (manual): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Merged (heapq):  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

--- Unequal Length Lists ---
  List 0: [1, 3, 5, 7, 9, 11]
  List 1: [2, 4]
  List 2: [6, 8, 10]
  List 3: []
  List 4: [0, 12, 13, 14]

Merged: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Correctness verified.
```

## Linked List Variant

When the $k$ sorted inputs are linked lists (a common interview problem), the same algorithm applies. Each heap entry stores the current node from each list. After extracting the minimum node, push `node.next` if it exists.

??? example "Linked List K-Way Merge"
    ```python
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def merge_k_linked(lists):
        heap = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i, node))

        dummy = ListNode(0)
        current = dummy

        while heap:
            val, idx, node = heapq.heappop(heap)
            current.next = ListNode(val)
            current = current.next
            if node.next:
                heapq.heappush(heap, (node.next.val, idx, node.next))

        return dummy.next
    ```

## Applications

| Application | How K-Way Merge is Used |
|------------|------------------------|
| External sorting | Merge sorted runs from disk |
| MapReduce | Merge sorted outputs from mapper tasks |
| Database joins | Merge sorted index scans |
| Log aggregation | Merge timestamped logs from $k$ servers |
| Tournament trees | Selection networks in hardware |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.5 and Problem 6-2. MIT Press.
- Python Documentation: [heapq.merge](https://docs.python.org/3/library/heapq.html#heapq.merge)
