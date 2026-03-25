# Auxiliary Space

When we say an algorithm uses $O(n)$ space, we might mean two very different things: the total memory footprint including the input, or only the *extra* memory the algorithm allocates beyond the input. The distinction matters because an algorithm that sorts an array of $n$ elements in place uses $O(n)$ total space but only $O(1)$ auxiliary space. Auxiliary space isolates the algorithm's own memory overhead from the size of the input it was given, making it the more informative measure when comparing algorithms that operate on the same data.

## Definition

The **auxiliary space** of an algorithm is the extra space (memory) used by the algorithm beyond the space occupied by the input itself. It includes:

- Temporary variables and counters
- Data structures created during execution (temporary arrays, hash tables, etc.)
- Stack frames from recursive calls
- Output space, if the output is separate from the input

Formally, if $S_{\text{total}}(n)$ is the total space and $S_{\text{input}}(n)$ is the space for the input, then:

$$
S_{\text{aux}}(n) = S_{\text{total}}(n) - S_{\text{input}}(n)
$$

## Auxiliary Space vs Total Space

| Measure | Includes input? | Typical use |
|---------|----------------|-------------|
| Total space | Yes | Analyzing memory-constrained environments |
| Auxiliary space | No | Comparing algorithm implementations on the same input |

An [in-place algorithm](in_place.md) is defined as one that uses $O(1)$ auxiliary space (or sometimes $O(\log n)$ to account for recursion stack depth). The term "in-place" refers to auxiliary space, not total space.

## Examples by Algorithm

### Sorting Algorithms

| Algorithm | Auxiliary space | Why |
|-----------|----------------|-----|
| Insertion sort | $O(1)$ | Uses only a constant number of temporary variables |
| Selection sort | $O(1)$ | Swaps elements in place using one temporary |
| Merge sort (standard) | $O(n)$ | Allocates a temporary array for merging |
| Merge sort (in-place) | $O(\log n)$ | Recursion stack only, but complex merge logic |
| Quicksort (Lomuto/Hoare) | $O(\log n)$ expected | Recursion stack depth is $O(\log n)$ on average |
| Quicksort (worst case) | $O(n)$ | Recursion stack depth is $O(n)$ for sorted input |
| Heapsort | $O(1)$ | Builds heap in the input array itself |
| Counting sort | $O(k)$ | Allocates count array of size $k$ (range of values) |
| Radix sort | $O(n + k)$ | Uses counting sort as a subroutine |

### Search and Graph Algorithms

| Algorithm | Auxiliary space | Why |
|-----------|----------------|-----|
| Binary search (iterative) | $O(1)$ | Two pointers and a midpoint |
| Binary search (recursive) | $O(\log n)$ | Recursion stack depth |
| BFS | $O(V)$ | Queue and visited array |
| DFS (iterative) | $O(V)$ | Explicit stack and visited array |
| DFS (recursive) | $O(V)$ | Recursion stack and visited array |

## Recursive Algorithms and Stack Space

A common source of auxiliary space that beginners overlook is the **call stack**. Every recursive call adds a stack frame containing local variables, parameters, and the return address. For a recursive algorithm with depth $d$, the stack contributes $O(d)$ auxiliary space.

??? example "Stack Space in Quicksort"
    Standard quicksort on a balanced partition has recursion depth $O(\log n)$, contributing $O(\log n)$ auxiliary space. In the worst case (already sorted input with naive pivot), the recursion depth is $O(n)$, and the auxiliary space becomes $O(n)$.

    Tail-call optimization or explicit iteration on the larger partition reduces the worst-case stack depth to $O(\log n)$.

### Converting Recursion to Iteration

Replacing recursion with an explicit stack does not eliminate auxiliary space -- it merely moves it from the call stack to the heap. The asymptotic auxiliary space remains the same. The benefit is practical: heap-allocated stacks avoid system stack overflow limits.

## Analyzing Auxiliary Space

To determine the auxiliary space of an algorithm:

1. **Identify all allocated data structures**: Arrays, lists, hash tables, trees, etc.
2. **Measure their sizes**: Express each as a function of the input size $n$
3. **Account for recursion depth**: Add $O(d)$ for a recursion of maximum depth $d$, where each frame uses $O(1)$ space
4. **Take the maximum**: Auxiliary space is the maximum over all points in execution (not the sum, unless the allocations overlap in time)

!!! warning "Simultaneous vs Sequential Allocations"
    If an algorithm allocates an $O(n)$ array, frees it, and then allocates another $O(n)$ array, the auxiliary space is $O(n)$, not $O(2n)$. Space analysis considers the peak memory usage at any single point in time.

## Auxiliary Space in Deep Learning

In the context of deep learning and PyTorch:

- **Forward pass**: Intermediate activations are stored for the backward pass. For a network with $L$ layers and batch size $B$, the auxiliary space for activations can be $O(B \cdot L \cdot d)$ where $d$ is the layer width.
- **Gradient checkpointing**: Trades auxiliary space for time by recomputing activations during the backward pass instead of storing them, reducing memory from $O(L)$ to $O(\sqrt{L})$ layers stored.
- **In-place operations**: PyTorch's in-place operators (e.g., `relu_()`, `add_()`) reduce auxiliary space but can interfere with autograd.

## Connections to Other Topics

- **[Memory Usage](memory.md)**: The broader view of how algorithms use memory
- **[In-Place Algorithms](in_place.md)**: Algorithms designed to minimize auxiliary space
- **[Space-Time Tradeoffs](tradeoffs.md)**: When reducing auxiliary space increases running time

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 2, 7-8. MIT Press.
- Knuth, D. E. (1997). *The Art of Computer Programming*, Vol. 3: Sorting and Searching. Addison-Wesley.
