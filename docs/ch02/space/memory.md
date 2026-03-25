# Memory Usage

Algorithm analysis typically focuses on time complexity, but memory is often the binding constraint in practice. A GPU has limited VRAM, a mobile device has limited RAM, and even a data center server cannot hold an arbitrarily large dataset in memory at once. Space complexity analysis quantifies how an algorithm's memory consumption grows with input size, enabling informed choices between algorithms when memory is scarce.

## Space Complexity

The **space complexity** $S(n)$ of an algorithm is the total amount of memory it requires as a function of the input size $n$. This includes:

1. **Input space**: Memory to store the input itself
2. **Auxiliary space**: Extra memory allocated during execution (temporary variables, data structures, stack frames)
3. **Output space**: Memory for the result, if separate from the input

$$
S(n) = S_{\text{input}}(n) + S_{\text{aux}}(n) + S_{\text{output}}(n)
$$

In most analyses, we focus on [auxiliary space](auxiliary.md) because the input and output sizes are determined by the problem specification, not the algorithm.

## Measuring Memory

### Units of Space

Space complexity can be measured at different levels of abstraction:

| Level | Unit | Example |
|-------|------|---------|
| Abstract | Words | Each variable, pointer, or array element counts as one word |
| Machine | Bytes | Each `int32` takes 4 bytes, each `float64` takes 8 bytes |
| System | Pages | Memory is allocated in pages (typically 4 KB) |

Algorithm analysis typically uses the **word model**: we count the number of words (machine-sized integers or pointers) used, where each word can hold a value up to $O(n)$ in $O(\log n)$ bits.

### Peak vs Cumulative

Space complexity measures **peak usage** -- the maximum amount of memory in use at any single point during execution. If an algorithm allocates an $O(n)$ array, frees it, then allocates another $O(n)$ array, the space complexity is $O(n)$, not $O(2n)$.

This differs from time complexity, which is cumulative: every operation counts, whether or not it overlaps with other operations.

## Common Space Complexity Classes

| Class | Description | Examples |
|-------|-------------|---------|
| $O(1)$ | Constant space | In-place sorting (insertion sort, heapsort) |
| $O(\log n)$ | Logarithmic | Balanced recursion (quicksort expected), binary search recursive |
| $O(n)$ | Linear | Merge sort auxiliary array, hash table, BFS queue |
| $O(n \log n)$ | Log-linear | Some divide-and-conquer with full recursion tree stored |
| $O(n^2)$ | Quadratic | Adjacency matrix, dynamic programming table for LCS |
| $O(2^n)$ | Exponential | Memoization table for subset problems |

## Sources of Memory Usage

### Stack Space

Every function call pushes a frame onto the call stack containing local variables, parameters, and the return address. Recursive algorithms accumulate stack frames proportional to the recursion depth.

$$
S_{\text{stack}}(n) = O(d) \cdot O(f)
$$

where $d$ is the maximum recursion depth and $f$ is the space per frame. For most algorithms, $f = O(1)$, so stack space is $O(d)$.

| Algorithm | Recursion depth | Stack space |
|-----------|----------------|-------------|
| Binary search | $O(\log n)$ | $O(\log n)$ |
| Merge sort | $O(\log n)$ | $O(\log n)$ |
| Quicksort (expected) | $O(\log n)$ | $O(\log n)$ |
| Quicksort (worst case) | $O(n)$ | $O(n)$ |
| Tree traversal | $O(h)$ | $O(h)$ |
| DFS on a graph | $O(V)$ | $O(V)$ |

### Heap Allocations

Explicit data structures allocated during execution consume heap memory. The size depends on the algorithm:

- A temporary array of size $n$ for merge sort: $O(n)$
- A hash table with $n$ entries: $O(n)$
- A priority queue for Dijkstra's algorithm: $O(V)$
- A memoization table for dynamic programming: depends on state space

### Hidden Memory Costs

Some operations allocate memory that is not immediately obvious:

- **String concatenation** in languages like Python creates a new string object each time
- **List comprehensions** allocate a new list
- **Tensor operations** in PyTorch allocate new tensors for the result (unless in-place variants are used)
- **Autograd graph**: PyTorch stores the computational graph during the forward pass, consuming memory proportional to the number of operations

## Analyzing Space Complexity: Examples

### Example 1: Iterative Sum

```python
def array_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
```

- Input space: $O(n)$ for the array
- Auxiliary space: $O(1)$ -- only the variable `total` and loop iterator
- Total: $O(n)$

### Example 2: Merge Sort

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

- Input space: $O(n)$
- Auxiliary space: $O(n)$ for the temporary arrays created during slicing, plus $O(\log n)$ for the recursion stack
- Total: $O(n)$

### Example 3: Dynamic Programming (Knapsack)

A 2D DP table for the 0/1 knapsack problem with $n$ items and capacity $W$:

- Table size: $n \times W$
- Auxiliary space: $O(nW)$

With space optimization (keeping only two rows), auxiliary space reduces to $O(W)$.

## Space Optimization Techniques

### Rolling Arrays

When a dynamic programming algorithm only depends on the previous row (or a constant number of previous rows), keep only those rows in memory instead of the full table.

### Streaming Algorithms

Process the input one element at a time without storing the entire input. Examples include computing the mean, finding the minimum, or maintaining a running histogram.

### Generators and Iterators

In Python, generators yield values one at a time instead of materializing the entire sequence in memory. This reduces space from $O(n)$ to $O(1)$ for many iteration patterns.

### Memory-Mapped Files

For datasets too large to fit in RAM, memory-mapped files provide the illusion of in-memory access while the OS pages data in and out of disk.

## Memory Usage in Deep Learning

Deep learning introduces unique memory challenges:

| Component | Memory | Scale |
|-----------|--------|-------|
| Model parameters | $O(P)$ | $P =$ number of parameters |
| Gradients | $O(P)$ | One gradient per parameter |
| Optimizer state | $O(P)$ to $O(3P)$ | Adam stores $m$ and $v$ per parameter |
| Activations | $O(B \cdot L \cdot d)$ | $B =$ batch size, $L =$ layers, $d =$ width |
| Input batch | $O(B \cdot D)$ | $D =$ input dimension |

For a model with 1 billion parameters in float32, the parameters alone consume approximately 4 GB. Gradients add another 4 GB, and Adam's optimizer state adds 8 GB more, totaling 16 GB before any activations are stored.

## Connections to Other Topics

- **[Auxiliary Space](auxiliary.md)**: The extra memory beyond the input
- **[In-Place Algorithms](in_place.md)**: Algorithms designed to minimize auxiliary space to $O(1)$
- **[Space-Time Tradeoffs](tradeoffs.md)**: How increasing space can reduce time and vice versa

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 2-4. MIT Press.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 8. MIT Press.
