# In-Place Algorithms

When working with large datasets -- a multi-gigabyte training corpus or a tensor that barely fits in GPU memory -- allocating a second copy of the data is often not an option. An in-place algorithm transforms its input using only a small, constant amount of extra memory, overwriting the original data rather than producing a separate output. This property makes in-place algorithms essential in memory-constrained environments, from embedded systems to GPU-accelerated deep learning pipelines.

## Definition

An algorithm is **in-place** if it uses $O(1)$ [auxiliary space](auxiliary.md) -- that is, the extra memory beyond the input is bounded by a constant, independent of the input size $n$.

A slightly relaxed definition allows $O(\log n)$ auxiliary space to account for:

- The recursion stack of divide-and-conquer algorithms
- Loop indices and pointer variables that require $\lceil \log n \rceil$ bits

Under either definition, the key requirement is that the algorithm does not allocate data structures whose size grows with $n$.

## The In-Place Constraint in Practice

### What Counts as Auxiliary Space

| Counts | Does not count |
|--------|---------------|
| Temporary arrays | The input array |
| Hash tables, trees | Read-only input that is not modified |
| Recursion stack frames | Output that the caller provided space for |
| Buffers for merging | |

### Strict vs Relaxed In-Place

| Definition | Auxiliary space | Examples |
|------------|----------------|---------|
| Strict | $O(1)$ | Insertion sort, selection sort, heapsort |
| Relaxed | $O(\log n)$ | Quicksort (expected), in-place merge sort |
| Not in-place | $\omega(\log n)$ | Standard merge sort ($O(n)$), counting sort ($O(k)$) |

## Classic In-Place Algorithms

### In-Place Sorting

**Insertion sort** maintains a sorted prefix and inserts each new element by shifting. It uses one temporary variable for the element being inserted.

```python
def insertion_sort(arr):
    """Sort arr in place using insertion sort. Auxiliary space: O(1)."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
```

**Heapsort** builds a max-heap in the input array, then repeatedly extracts the maximum. It uses no extra array -- the heap structure is maintained within the original array.

**Quicksort** partitions the array around a pivot and recurses on the two halves. The partition step is in-place (swapping elements), and the expected recursion depth is $O(\log n)$.

### In-Place Array Reversal

Reversing an array in place requires only two index variables and one temporary:

```python
def reverse_in_place(arr):
    """Reverse arr in place. Auxiliary space: O(1)."""
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
```

### In-Place Matrix Transpose

Transposing an $n \times n$ matrix in place swaps $A[i][j]$ with $A[j][i]$ for all $i < j$:

```python
def transpose_in_place(matrix):
    """Transpose a square matrix in place. Auxiliary space: O(1)."""
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

For non-square matrices, in-place transposition is significantly more complex and involves following permutation cycles.

## Techniques for Making Algorithms In-Place

### Swapping

The most common in-place technique. Two elements exchange positions using a temporary variable (or XOR trick), requiring $O(1)$ auxiliary space.

### Partitioning

Quicksort's partition rearranges elements around a pivot without allocating a new array. The Lomuto and Hoare partition schemes both work in-place.

### Overwriting Input

When the output can replace the input element by element, no extra array is needed. For example, applying a function $f$ to each element: `arr[i] = f(arr[i])`.

### Bit Manipulation

Encoding extra information in unused bits of the input (e.g., sign bits, high-order bits) can eliminate auxiliary data structures. This is a specialized technique used in some in-place graph algorithms.

## When In-Place Is Not Possible (or Not Worth It)

Some algorithms fundamentally require extra space:

- **Merge sort**: The standard merge step requires an $O(n)$ temporary array. In-place merge algorithms exist but are complex and have larger constant factors.
- **Counting sort**: Requires a count array of size $k$ (the range of input values).
- **BFS**: Requires a queue that can grow to $O(V)$.
- **Hash tables**: Require $O(n)$ space for the table itself.

!!! warning "The Cost of Going In-Place"
    Forcing an algorithm to be in-place can increase time complexity or constant factors. In-place merge sort achieves $O(n \log^2 n)$ with a straightforward approach or $O(n \log n)$ with a complex block-merge strategy. Standard merge sort with $O(n)$ auxiliary space is simpler and often faster in practice.

## In-Place Operations in PyTorch

PyTorch provides in-place variants of many tensor operations, denoted by a trailing underscore:

| Standard | In-place | Effect |
|----------|----------|--------|
| `torch.add(x, y)` | `x.add_(y)` | Adds $y$ to $x$ in place |
| `torch.relu(x)` | `x.relu_()` | Applies ReLU in place |
| `torch.zero_like(x)` | `x.zero_()` | Zeros out $x$ in place |
| `x.clone()` | (no in-place) | Always allocates new memory |

!!! warning "In-Place Operations and Autograd"
    In-place operations can cause errors during backpropagation if they modify a tensor that is needed for gradient computation. PyTorch's autograd raises a `RuntimeError` when an in-place operation invalidates the computation graph. Use in-place operations only on leaf tensors or tensors that are not required for gradient computation.

## Stability and In-Place

An important interaction exists between the in-place property and sorting stability. Some in-place sorting algorithms are stable (insertion sort), while others are not (heapsort, standard quicksort). Achieving both in-place and stable sorting simultaneously with $O(n \log n)$ time is possible but requires sophisticated algorithms (e.g., block merge sort).

| Algorithm | In-place | Stable | Time |
|-----------|----------|--------|------|
| Insertion sort | Yes | Yes | $O(n^2)$ |
| Heapsort | Yes | No | $O(n \log n)$ |
| Quicksort | Yes (relaxed) | No | $O(n \log n)$ expected |
| Merge sort | No ($O(n)$ aux) | Yes | $O(n \log n)$ |
| Block merge sort | Yes | Yes | $O(n \log n)$ |

## Connections to Other Topics

- **[Auxiliary Space](auxiliary.md)**: The formal definition of the extra space that in-place algorithms minimize
- **[Memory Usage](memory.md)**: The broader perspective on algorithm memory consumption
- **[Space-Time Tradeoffs](tradeoffs.md)**: The cost of reducing space, often measured in increased time

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 2, 6-8. MIT Press.
- Knuth, D. E. (1997). *The Art of Computer Programming*, Vol. 3: Sorting and Searching. Addison-Wesley.
