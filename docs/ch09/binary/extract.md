# Extract Min/Max

The **extract** operation removes and returns the root element of a heap -- the minimum in a min-heap or the maximum in a max-heap. This is the central operation that makes heaps useful as priority queues: it always delivers the highest-priority element in $O(\log n)$ time. Understanding extract also clarifies why heapsort works, since heapsort is simply repeated extraction.

## Algorithm

Extracting the root directly would leave a gap at index 0 and potentially break the complete binary tree shape. The standard approach avoids this by using a three-step procedure:

1. **Save** the root value (the element to return).
2. **Move** the last element in the array to the root position, reducing the heap size by one. This preserves the complete binary tree shape.
3. **Sift down** the new root to restore the heap property.

### Pseudocode for Extract-Max

```
EXTRACT-MAX(A):
    if heap_size < 1:
        error "heap underflow"
    max_val = A[0]
    A[0] = A[heap_size - 1]
    heap_size = heap_size - 1
    MAX-HEAPIFY(A, 0, heap_size)
    return max_val
```

## Step-by-Step Example

Extract the maximum from the max-heap `[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]`:

```
Step 1: Save root value 16.

Step 2: Move last element (1) to root, reduce size to 9.
          1
        /    \
      14      10
     /  \    /  \
    8    7  9    3
   / \
  2   4

Step 3: Sift down 1.
  Compare 1 with children 14 and 10. Swap with 14 (larger child).
          14
        /    \
      1       10
     /  \    /  \
    8    7  9    3
   / \
  2   4

  Compare 1 with children 8 and 7. Swap with 8.
          14
        /    \
      8       10
     /  \    /  \
    1    7  9    3
   / \
  2   4

  Compare 1 with children 2 and 4. Swap with 4.
          14
        /    \
      8       10
     /  \    /  \
    4    7  9    3
   / \
  2   1

Result: returned 16, heap is [14, 8, 10, 4, 7, 9, 3, 2, 1].
```

## Complexity Analysis

The sift-down procedure traverses at most one path from the root to a leaf. In a heap of $n$ elements, the height is $\lfloor \log_2 n \rfloor$, so sift-down performs at most $\lfloor \log_2 n \rfloor$ comparisons and swaps.

| Operation | Time Complexity |
|-----------|----------------|
| Save root | $O(1)$ |
| Move last to root | $O(1)$ |
| Sift down | $O(\log n)$ |
| **Total** | $O(\log n)$ |

The space complexity is $O(1)$ for the iterative version, or $O(\log n)$ for the recursive version due to the call stack.

## Peek Without Extraction

Sometimes we need to inspect the minimum or maximum without removing it. Since the root always holds the extreme element, peeking is a simple $O(1)$ array access:

$$
\text{peek}(A) = A[0]
$$

## Implementation

```python
"""
Extract-min and extract-max operations for binary heaps.

Demonstrates removal of the root element with sift-down
to restore the heap property in O(log n) time.
"""


# === Max-Heap Extract ===

class MaxHeap:
    """A max-heap supporting insert, extract-max, and peek."""

    def __init__(self, items=None):
        """Build a max-heap from an optional list of items."""
        self.heap = list(items) if items else []
        # Build heap using bottom-up sift-down
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    def _sift_down(self, i):
        """Move element at index i down to restore heap property."""
        n = len(self.heap)
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and self.heap[left] > self.heap[largest]:
                largest = left
            if right < n and self.heap[right] > self.heap[largest]:
                largest = right

            if largest == i:
                break
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            i = largest

    def _sift_up(self, i):
        """Move element at index i up to restore heap property."""
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] > self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def insert(self, val):
        """Insert a value into the heap. O(log n)."""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def peek(self):
        """Return the maximum without removing it. O(1)."""
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]

    def extract_max(self):
        """Remove and return the maximum element. O(log n)."""
        if not self.heap:
            raise IndexError("extract from empty heap")

        max_val = self.heap[0]

        # Move last element to root
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)

        return max_val

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return f"MaxHeap({self.heap})"


# === Demonstration ===

if __name__ == "__main__":
    # Build a max-heap
    h = MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
    print(f"Initial heap: {h.heap}")
    print(f"Peek: {h.peek()}")

    # Extract elements one by one (produces sorted order descending)
    print("\nExtracting elements:")
    extracted = []
    while len(h) > 0:
        val = h.extract_max()
        extracted.append(val)
        print(f"  Extracted {val}, heap: {h.heap}")

    print(f"\nExtracted in order: {extracted}")

    # Demonstrate insert + extract interleaving
    print("\n--- Insert and Extract ---")
    h2 = MaxHeap()
    for val in [5, 3, 8]:
        h2.insert(val)
        print(f"  Inserted {val}: {h2.heap}")

    print(f"  Extract max: {h2.extract_max()}, heap: {h2.heap}")
    h2.insert(10)
    print(f"  Inserted 10: {h2.heap}")
    print(f"  Extract max: {h2.extract_max()}, heap: {h2.heap}")
```

**Output:**
```
Initial heap: [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Peek: 16

Extracting elements:
  Extracted 16, heap: [14, 8, 10, 4, 7, 9, 3, 2, 1]
  Extracted 14, heap: [10, 8, 9, 4, 7, 1, 3, 2]
  Extracted 10, heap: [9, 8, 3, 4, 7, 1, 2]
  Extracted 9, heap: [8, 7, 3, 4, 2, 1]
  Extracted 8, heap: [7, 4, 3, 1, 2]
  Extracted 7, heap: [4, 2, 3, 1]
  Extracted 4, heap: [3, 2, 1]
  Extracted 3, heap: [2, 1]
  Extracted 2, heap: [1]
  Extracted 1, heap: []

Extracted in order: [16, 14, 10, 9, 8, 7, 4, 3, 2, 1]

--- Insert and Extract ---
  Inserted 5: [5]
  Inserted 3: [5, 3]
  Inserted 8: [8, 3, 5]
  Extract max: 8, heap: [5, 3]
  Inserted 10: [10, 3, 5]
  Extract max: 10, heap: [5, 3]
```

## Correctness Argument

After moving the last element to the root, both subtrees of the root are still valid heaps (they were not modified). The only potential violation is at the root itself. Sift-down restores the heap property by repeatedly swapping the root with its largest child until the element reaches a position where it is at least as large as both children (or becomes a leaf). This is exactly the precondition that `MAX-HEAPIFY` requires: both subtrees are valid heaps, and only the root may violate the property.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.2 and 6.5: Maintaining the heap property and Priority queues. MIT Press.
