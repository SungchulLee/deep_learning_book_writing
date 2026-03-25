# Space-Time Tradeoffs

Algorithm design rarely offers a free lunch: improving an algorithm's running time often requires using more memory, and reducing memory usage often means doing more computation. This fundamental tension -- the **space-time tradeoff** -- is one of the central principles in algorithm design. Understanding where an algorithm sits on the space-time spectrum helps practitioners choose the right implementation for their hardware constraints, whether that means fitting a model into GPU memory or meeting a latency target.

## The Core Principle

Given a computational problem, there is typically a family of algorithms that solve it, each making a different tradeoff between time $T(n)$ and space $S(n)$. The product $T(n) \cdot S(n)$ often satisfies a lower bound:

$$
T(n) \cdot S(n) \geq f(n)
$$

for some function $f(n)$ that depends on the problem. Algorithms on the **Pareto frontier** -- where no other algorithm is better in both time and space -- represent the best achievable tradeoffs.

In practice, the tradeoff manifests in two directions:

- **Use more space to save time**: Precompute and cache results (lookup tables, memoization, hash tables)
- **Use more time to save space**: Recompute values on the fly instead of storing them

## Classic Examples

### Memoization vs Recomputation

**Time-optimized**: Store all previously computed values in a table.

The Fibonacci recurrence $F(n) = F(n-1) + F(n-2)$ with naive recursion takes $O(2^n)$ time and $O(n)$ stack space. Memoization stores all $n$ values, reducing time to $O(n)$ at a cost of $O(n)$ space for the table.

**Space-optimized**: Since $F(n)$ depends only on $F(n-1)$ and $F(n-2)$, keep only two variables:

$$
T(n) = O(n), \quad S(n) = O(1)
$$

This eliminates the table entirely while retaining $O(n)$ time.

### Lookup Tables vs Computation

**Time-optimized**: Precompute $\sin(x)$ for all values in a fixed set and store them in a table. Each query is $O(1)$.

**Space-optimized**: Compute $\sin(x)$ on demand using a Taylor series or CORDIC algorithm. Each query takes $O(k)$ time (for $k$ terms of precision) but requires $O(1)$ space.

| Approach | Time per query | Space |
|----------|---------------|-------|
| Full table ($N$ entries) | $O(1)$ | $O(N)$ |
| Compute on demand | $O(k)$ | $O(1)$ |
| Sparse table + interpolation | $O(1)$ | $O(\sqrt{N})$ |

The sparse table approach offers an intermediate tradeoff.

### Sorting: Merge Sort vs Heapsort

Both achieve $O(n \log n)$ time, but they differ in space:

| Algorithm | Time | Auxiliary space |
|-----------|------|----------------|
| Merge sort | $O(n \log n)$ | $O(n)$ |
| Heapsort | $O(n \log n)$ | $O(1)$ |

Merge sort uses extra space for the merge step and typically has better cache behavior and smaller constant factors. Heapsort is in-place but has worse cache locality. The tradeoff here is between space and practical speed (constant factors).

### Hash Table vs Binary Search Tree

| Structure | Lookup time | Space | Ordered iteration |
|-----------|-------------|-------|-------------------|
| Hash table | $O(1)$ expected | $O(n)$ with overhead | No |
| Balanced BST | $O(\log n)$ | $O(n)$ | Yes |
| Sorted array | $O(\log n)$ | $O(n)$, minimal overhead | Yes |

Hash tables trade ordered access for $O(1)$ lookup. They also use more space per element due to load factor requirements (typically keeping the table at most 75% full).

## Dynamic Programming Tradeoffs

Dynamic programming offers rich space-time tradeoffs because the table structure can often be compressed.

### Full Table vs Rolling Array

For the longest common subsequence (LCS) of sequences of length $m$ and $n$:

- **Full table**: $O(mn)$ time, $O(mn)$ space. Supports solution reconstruction.
- **Two-row rolling**: $O(mn)$ time, $O(\min(m,n))$ space. Gives the optimal *value* but not the solution itself.
- **Hirschberg's algorithm**: $O(mn)$ time, $O(\min(m,n))$ space, with solution reconstruction using a divide-and-conquer approach.

### Knapsack

For the 0/1 knapsack with $n$ items and capacity $W$:

- **Full table**: $O(nW)$ time, $O(nW)$ space
- **One-row optimization**: $O(nW)$ time, $O(W)$ space (iterating the weight dimension in reverse)

## Space-Time Tradeoffs in Deep Learning

### Gradient Checkpointing

During training, the forward pass stores all intermediate activations for the backward pass. For a network with $L$ layers:

- **Standard**: $O(L)$ memory for activations, one forward + one backward pass
- **Gradient checkpointing**: $O(\sqrt{L})$ memory, but requires recomputing some activations during the backward pass, roughly doubling the forward computation time

This is the most prominent space-time tradeoff in deep learning training.

### Model Precision

Reducing numerical precision saves memory at the cost of potential accuracy loss:

| Precision | Bytes per parameter | Memory for 1B params |
|-----------|--------------------|-----------------------|
| float32 | 4 | 4 GB |
| float16 / bfloat16 | 2 | 2 GB |
| int8 | 1 | 1 GB |
| int4 | 0.5 | 0.5 GB |

Mixed-precision training uses float16 for most operations and float32 for critical accumulations, achieving a 2x memory reduction with minimal accuracy impact.

### KV-Cache vs Recomputation

In autoregressive transformer inference, the **KV-cache** stores key and value tensors from all previous tokens to avoid recomputing attention:

- **With KV-cache**: $O(L \cdot n \cdot d)$ memory, $O(d^2)$ time per new token
- **Without KV-cache**: $O(L \cdot d)$ memory, $O(n \cdot d^2)$ time per new token (recompute attention over all $n$ previous tokens)

For long sequences, the KV-cache can consume gigabytes of memory, motivating techniques like sliding-window attention and paged attention.

### Batch Size Tradeoffs

Larger batch sizes improve GPU utilization (time per sample decreases) but require more memory for activations:

$$
S_{\text{activations}} = O(B \cdot L \cdot d)
$$

where $B$ is the batch size. Gradient accumulation simulates large batches without the memory cost by summing gradients over multiple small-batch forward passes.

## Quantifying the Tradeoff

For some problems, the tradeoff has been characterized precisely:

- **Element distinctness**: Any algorithm deciding whether $n$ elements are distinct satisfies $T \cdot S = \Omega(n^{3/2})$ in the branching-program model.
- **Matrix multiplication**: Faster algorithms (Strassen, etc.) use more auxiliary space than the naive $O(n^3)$ algorithm.
- **Sorting**: Comparison-based sorting requires $\Omega(n \log n)$ comparisons regardless of space. With $O(1)$ extra space (heapsort), the constant factor is larger than with $O(n)$ extra space (merge sort).

## Decision Framework

When choosing an algorithm, consider:

| Question | Favor time | Favor space |
|----------|-----------|-------------|
| Is memory the bottleneck? | No | Yes |
| Is latency critical? | Yes | No |
| Is the computation repeated? | Cache results | Recompute each time |
| Is the data size fixed? | Precompute tables | Keep tables small |
| Training or inference? | Use checkpointing | Use full activation storage |

## Connections to Other Topics

- **[Memory Usage](memory.md)**: How to measure the space side of the tradeoff
- **[Auxiliary Space](auxiliary.md)**: The extra memory that tradeoffs typically affect
- **[In-Place Algorithms](in_place.md)**: Algorithms at the extreme space-efficient end of the spectrum

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training deep nets with sublinear memory cost. arXiv:1604.06174.
- Knuth, D. E. (1997). *The Art of Computer Programming*, Vol. 3: Sorting and Searching. Addison-Wesley.
