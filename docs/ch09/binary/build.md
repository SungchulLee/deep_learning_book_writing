# Build Heap

Given an unordered array of $n$ elements, we often need to transform it into a valid heap before performing any extract or priority queue operations. A naive approach -- inserting elements one at a time -- costs $O(n \log n)$. The **Build-Heap** algorithm achieves the same result in $O(n)$ time by exploiting a bottom-up strategy: start from the last non-leaf node and apply sift-down at each position, working toward the root.

## Naive Approach: Repeated Insertion

The most straightforward way to build a heap is to start with an empty heap and insert elements one by one. Each insertion calls sift-up, which costs $O(\log n)$ in the worst case. Over $n$ insertions the total cost is:

$$
\sum_{i=1}^{n} O(\log i) = O(n \log n)
$$

This works but is suboptimal. The Build-Heap algorithm does better by a constant factor that matters in practice and by a full asymptotic class in theory.

## Bottom-Up Build-Heap Algorithm

The key insight is that leaves already satisfy the heap property trivially (they have no children to violate it). In a complete binary tree with $n$ nodes, the leaves occupy indices $\lfloor n/2 \rfloor$ through $n-1$ (0-indexed). Build-Heap starts at the last non-leaf node and applies sift-down at each index, moving backward to the root.

### Algorithm

```
BUILD-MAX-HEAP(A):
    n = length(A)
    for i = floor(n/2) - 1 down to 0:
        MAX-HEAPIFY(A, i, n)
```

After each call to `MAX-HEAPIFY(A, i, n)`, the subtree rooted at index $i$ is a valid max-heap. Since the loop processes nodes from bottom to top, by the time we heapify node $i$, both of its children's subtrees are already valid heaps.

### Step-by-Step Example

Consider building a max-heap from the array `[4, 1, 3, 2, 16, 9, 10, 14, 8, 7]`:

```
Initial array (n=10, last non-leaf at index 4):

          4
        /   \
      1       3
     / \     / \
    2   16  9   10
   / \  /
  14 8 7

Step 1: heapify index 4 (value 16) — children: 7. No swap needed.
Step 2: heapify index 3 (value 2)  — children: 14, 8. Swap 2 and 14.
Step 3: heapify index 2 (value 3)  — children: 9, 10. Swap 3 and 10.
Step 4: heapify index 1 (value 1)  — children: 14, 16. Swap 1 and 16, then 1 and 7.
Step 5: heapify index 0 (value 4)  — children: 16, 10. Swap 4 and 16, then 4 and 14, then 4 and 8.

Result:
          16
        /    \
      14      10
     / \     /  \
    8   7   9    3
   / \  /
  2  4 1
```

The final array is `[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]`.

## Why Bottom-Up is O(n)

The $O(n)$ complexity is not obvious because each sift-down can cost up to $O(h)$ where $h$ is the height of the subtree. The key is that most nodes are near the bottom of the tree where subtree heights are small:

- At height 0 (leaves): $\lceil n/2 \rceil$ nodes, each doing 0 work
- At height 1: $\lceil n/4 \rceil$ nodes, each doing at most 1 swap
- At height $h$: $\lceil n/2^{h+1} \rceil$ nodes, each doing at most $h$ swaps

The total work is:

$$
\sum_{h=0}^{\lfloor \log n \rfloor} \left\lceil \frac{n}{2^{h+1}} \right\rceil \cdot O(h) = O\!\left(n \sum_{h=0}^{\infty} \frac{h}{2^h}\right) = O(n)
$$

The infinite series $\sum_{h=0}^{\infty} h/2^h = 2$ converges, so the entire Build-Heap procedure runs in $O(n)$ time. The detailed proof is covered on the sibling page *Build Heap O(n) Proof*.

## Implementation

```python
"""
Build-Heap algorithm implementation.

Constructs a max-heap from an unordered array in O(n) time
using the bottom-up sift-down approach.
"""


# === Sift Down (Max-Heapify) ===

def sift_down(arr, i, n):
    """Restore the max-heap property for the subtree rooted at index i.

    Assumes both subtrees of i are already valid max-heaps.
    Only considers elements in arr[0:n].
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        sift_down(arr, largest, n)


# === Build Max-Heap ===

def build_max_heap(arr):
    """Transform arr into a max-heap in O(n) time.

    Iterates from the last non-leaf node down to the root,
    calling sift_down at each position.
    """
    n = len(arr)
    # Last non-leaf node is at index n//2 - 1
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, i, n)


# === Build Min-Heap ===

def sift_down_min(arr, i, n):
    """Restore the min-heap property for the subtree rooted at index i."""
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] < arr[smallest]:
        smallest = left
    if right < n and arr[right] < arr[smallest]:
        smallest = right

    if smallest != i:
        arr[i], arr[smallest] = arr[smallest], arr[i]
        sift_down_min(arr, smallest, n)


def build_min_heap(arr):
    """Transform arr into a min-heap in O(n) time."""
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down_min(arr, i, n)


# === Demonstration ===

if __name__ == "__main__":
    # Build a max-heap
    data = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    print(f"Original array: {data}")

    build_max_heap(data)
    print(f"Max-heap:       {data}")

    # Verify the max-heap property
    for i in range(1, len(data)):
        parent = (i - 1) // 2
        assert data[parent] >= data[i], f"Heap violation at index {i}"
    print("Max-heap property verified.\n")

    # Build a min-heap
    data2 = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    build_min_heap(data2)
    print(f"Min-heap:       {data2}")

    # Compare with Python's heapq
    import heapq
    data3 = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    heapq.heapify(data3)
    print(f"heapq result:   {data3}")
```

**Output:**
```
Original array: [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
Max-heap:       [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Max-heap property verified.

Min-heap:       [1, 2, 3, 4, 7, 9, 10, 14, 8, 16]
heapq result:   [1, 2, 3, 4, 7, 9, 10, 14, 8, 16]
```

## Comparison of Build Strategies

| Strategy | Approach | Time | Space |
|----------|----------|------|-------|
| Repeated insertion | Insert one at a time, sift up each | $O(n \log n)$ | $O(1)$ |
| Bottom-up Build-Heap | Sift down from last non-leaf to root | $O(n)$ | $O(1)$ |

Both strategies produce a valid heap, but the bottom-up approach is strictly faster. Python's `heapq.heapify` uses the bottom-up $O(n)$ algorithm internally.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.3: Building a heap. MIT Press.
