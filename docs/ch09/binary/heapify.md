# Heapify (Sift Down)

The **sift-down** operation (also called **heapify** or **max-heapify**) is the fundamental repair procedure for binary heaps. When a single node violates the heap property while both its subtrees are valid heaps, sift-down restores the invariant by moving the offending node downward through the tree. This operation is the workhorse behind extract, build-heap, and heapsort.

## Precondition

Sift-down at node $i$ requires a specific precondition:

> The subtrees rooted at $\text{left}(i)$ and $\text{right}(i)$ are both valid heaps. Only node $i$ itself may violate the heap property.

This precondition is always satisfied in the two main use cases:

1. **Extract**: after moving the last element to the root, both subtrees are untouched and remain valid heaps.
2. **Build-Heap**: processing nodes bottom-up ensures that both children's subtrees are already heapified before the parent is processed.

## Algorithm

For a max-heap, sift-down compares node $i$ with its children and swaps it with the larger child if a violation exists. The process repeats at the child's position until the heap property is restored or a leaf is reached.

### Pseudocode

```
MAX-HEAPIFY(A, i, n):
    largest = i
    left = 2*i + 1
    right = 2*i + 2

    if left < n and A[left] > A[largest]:
        largest = left
    if right < n and A[right] > A[largest]:
        largest = right

    if largest != i:
        swap A[i] and A[largest]
        MAX-HEAPIFY(A, largest, n)
```

Each iteration makes two comparisons (node vs left child, node vs right child) and at most one swap.

## Step-by-Step Example

Consider a max-heap where the root (index 0) violates the property but both subtrees are valid:

```
Initial state (only root violates):
          2
        /    \
      14      10
     /  \    /  \
    8    7  9    3

Step 1: Compare 2 with children 14 and 10.
        largest = 14 (index 1). Swap 2 and 14.
          14
        /    \
      2       10
     /  \    /  \
    8    7  9    3

Step 2: Compare 2 with children 8 and 7.
        largest = 8 (index 3). Swap 2 and 8.
          14
        /    \
      8       10
     /  \    /  \
    2    7  9    3

Step 3: Node 2 is at index 3 with no children below it in this example.
        2 is a leaf — stop.

Result: Heap property restored in 2 swaps.
```

## Complexity Analysis

Sift-down traverses at most one root-to-leaf path. The height of a complete binary tree with $n$ nodes is $\lfloor \log_2 n \rfloor$, so:

$$
T(n) = O(\log n)
$$

More precisely, at each level the algorithm performs exactly 2 comparisons (with left and right children) and at most 1 swap. The total number of comparisons is at most $2 \lfloor \log_2 n \rfloor$ and the total number of swaps is at most $\lfloor \log_2 n \rfloor$.

!!! tip "Why Swap with the Larger Child?"
    In a max-heap, swapping with the **larger** child (not just any violating child) is essential. If we swapped with the smaller child, that child would become the parent of its former sibling, potentially creating a new violation. Choosing the larger child guarantees that the new parent is at least as large as both children.

## Recursive vs Iterative

The recursive version is cleaner but uses $O(\log n)$ stack space. The iterative version uses $O(1)$ space:

```python
"""
Sift-down (heapify) operation for binary heaps.

Provides both recursive and iterative implementations of the
fundamental heap repair operation.
"""


# === Recursive Sift-Down (Max-Heap) ===

def sift_down_recursive(arr, i, n):
    """Recursively restore the max-heap property at index i.

    Precondition: subtrees rooted at left(i) and right(i)
    are valid max-heaps.
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
        sift_down_recursive(arr, largest, n)


# === Iterative Sift-Down (Max-Heap) ===

def sift_down_iterative(arr, i, n):
    """Iteratively restore the max-heap property at index i.

    Uses O(1) auxiliary space instead of O(log n) stack space.
    """
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest


# === Sift-Down for Min-Heap ===

def sift_down_min(arr, i, n):
    """Restore the min-heap property at index i (iterative)."""
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] < arr[smallest]:
            smallest = left
        if right < n and arr[right] < arr[smallest]:
            smallest = right

        if smallest == i:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest


# === Demonstration ===

if __name__ == "__main__":
    # Max-heap with violation at root
    arr1 = [2, 14, 10, 8, 7, 9, 3]
    print(f"Before sift-down (recursive): {arr1}")
    sift_down_recursive(arr1, 0, len(arr1))
    print(f"After sift-down (recursive):  {arr1}")

    # Same example with iterative version
    arr2 = [2, 14, 10, 8, 7, 9, 3]
    print(f"\nBefore sift-down (iterative): {arr2}")
    sift_down_iterative(arr2, 0, len(arr2))
    print(f"After sift-down (iterative):  {arr2}")

    # Verify both produce the same result
    assert arr1 == arr2, "Mismatch between recursive and iterative"
    print("\nBoth versions produce identical results.")

    # Min-heap example
    arr3 = [10, 2, 3, 8, 7, 9, 4]
    print(f"\nMin-heap before sift-down: {arr3}")
    sift_down_min(arr3, 0, len(arr3))
    print(f"Min-heap after sift-down:  {arr3}")
```

**Output:**
```
Before sift-down (recursive): [2, 14, 10, 8, 7, 9, 3]
After sift-down (recursive):  [14, 8, 10, 2, 7, 9, 3]

Before sift-down (iterative): [2, 14, 10, 8, 7, 9, 3]
After sift-down (iterative):  [14, 8, 10, 2, 7, 9, 3]

Both versions produce identical results.

Min-heap before sift-down: [10, 2, 3, 8, 7, 9, 4]
Min-heap after sift-down:  [2, 7, 3, 8, 10, 9, 4]
```

## Correctness

**Loop invariant**: at the start of each iteration, the subtree rooted at index $i$ satisfies the max-heap property everywhere except possibly at node $i$ itself.

- **Initialization**: the precondition guarantees this.
- **Maintenance**: if $A[i]$ is smaller than one of its children, we swap with the larger child. After the swap, the former child position now holds a value at least as large as both of its children. The only potential new violation is at the position where the swapped element landed.
- **Termination**: the loop terminates when $i$ has no children ($i$ is a leaf) or when $A[i]$ is at least as large as both children. In both cases the heap property holds throughout the subtree.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.2: Maintaining the heap property. MIT Press.
