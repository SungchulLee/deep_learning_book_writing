# Parallel Prefix Sum

The **prefix sum** (also called *scan*) computes all partial sums of an array: given $[a_0, a_1, \ldots, a_{n-1}]$, produce $[a_0, a_0 + a_1, \ldots, a_0 + a_1 + \cdots + a_{n-1}]$. This operation appears everywhere in parallel computing -- from load balancing and stream compaction to sorting and graph algorithms. While a sequential scan runs in $O(n)$ time, the parallel prefix sum achieves $O(\log n)$ span with $O(n)$ work, making it one of the most important parallel primitives.

## Problem Definition

Given an array $A = [a_0, a_1, \ldots, a_{n-1}]$ and an associative binary operator $\oplus$, the **inclusive prefix sum** produces:

$$
S[i] = \bigoplus_{k=0}^{i} a_k = a_0 \oplus a_1 \oplus \cdots \oplus a_i
$$

The **exclusive prefix sum** shifts the result by one position:

$$
S[i] = \bigoplus_{k=0}^{i-1} a_k
$$

with $S[0]$ set to the identity element of $\oplus$.

## Blelloch's Algorithm (Work-Efficient Scan)

Blelloch's algorithm computes the prefix sum in two phases on a balanced binary tree structure built over the array.

### Phase 1: Up-Sweep (Reduce)

Traverse the tree from leaves to root, computing partial sums at each internal node. At level $d$ (counting from leaves), each node stores the sum of its subtree:

$$
\text{tree}[i] \leftarrow \text{tree}[i] \oplus \text{tree}[i - 2^d]
$$

After the up-sweep, the root holds the total sum.

### Phase 2: Down-Sweep (Distribute)

Set the root to the identity element, then traverse from root to leaves. At each node, propagate the prefix sum downward:

$$
\text{left} \leftarrow \text{parent}, \quad
\text{right} \leftarrow \text{parent} \oplus \text{old\_left}
$$

After the down-sweep, each position holds its exclusive prefix sum.

### Work-Span Analysis

- **Work**: Each phase performs $O(n)$ operations. Total: $T_1 = O(n)$.
- **Span**: Each phase has $O(\log n)$ levels. Total: $T_\infty = O(\log n)$.
- **Parallelism**: $P = O(n / \log n)$.

## Implementation

```python
"""
Parallel prefix sum (Blelloch's work-efficient scan).

Simulates the up-sweep (reduce) and down-sweep (distribute)
phases of the parallel prefix sum algorithm.
"""

import math

# ===================================================================
# Blelloch's Work-Efficient Scan
# ===================================================================

def parallel_prefix_sum(arr):
    """Compute exclusive prefix sum using Blelloch's algorithm.

    Args:
        arr: input array of numbers

    Returns:
        Exclusive prefix sum array
    """
    n = len(arr)
    # Pad to next power of 2
    size = 1 << math.ceil(math.log2(max(n, 2)))
    tree = list(arr) + [0] * (size - n)

    # Up-sweep (reduce phase)
    stride = 1
    while stride < size:
        for i in range(2 * stride - 1, size, 2 * stride):
            tree[i] += tree[i - stride]
        stride *= 2

    # Set root to zero for exclusive scan
    tree[size - 1] = 0

    # Down-sweep (distribute phase)
    stride = size // 2
    while stride >= 1:
        for i in range(2 * stride - 1, size, 2 * stride):
            temp = tree[i - stride]
            tree[i - stride] = tree[i]
            tree[i] += temp
        stride //= 2

    return tree[:n]


def inclusive_prefix_sum(arr):
    """Compute inclusive prefix sum.

    Args:
        arr: input array of numbers

    Returns:
        Inclusive prefix sum array
    """
    exclusive = parallel_prefix_sum(arr)
    return [exclusive[i] + arr[i] for i in range(len(arr))]

# ===================================================================
# Sequential Prefix Sum (for comparison)
# ===================================================================

def sequential_prefix_sum(arr):
    """Compute inclusive prefix sum sequentially.

    Args:
        arr: input array of numbers

    Returns:
        Inclusive prefix sum array
    """
    result = []
    running = 0
    for x in arr:
        running += x
        result.append(running)
    return result

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    arr = [3, 1, 7, 0, 4, 1, 6, 3]

    exclusive = parallel_prefix_sum(arr)
    inclusive = inclusive_prefix_sum(arr)
    sequential = sequential_prefix_sum(arr)

    print(f"Input:            {arr}")
    print(f"Exclusive prefix: {exclusive}")
    print(f"Inclusive prefix:  {inclusive}")
    print(f"Sequential check: {sequential}")
    print(f"Match: {inclusive == sequential}")

    # Work-span analysis
    n = len(arr)
    work = 2 * n
    span = 2 * math.ceil(math.log2(n))
    print(f"\nn = {n}")
    print(f"Work O(n) ~ {work}")
    print(f"Span O(log n) ~ {span}")
    print(f"Parallelism ~ {work / span:.1f}")
```

**Output:**
```
Input:            [3, 1, 7, 0, 4, 1, 6, 3]
Exclusive prefix: [0, 3, 4, 11, 11, 15, 16, 22]
Inclusive prefix:  [3, 4, 11, 11, 15, 16, 22, 25]
Sequential check: [3, 4, 11, 11, 15, 16, 22, 25]
Match: True

n = 8
Work O(n) ~ 16
Span O(log n) ~ 6
Parallelism ~ 2.7
```

## Hillis-Steele Algorithm

An alternative approach with simpler logic but more work:

At each step $d = 0, 1, \ldots, \lceil \log_2 n \rceil - 1$, for every index $i \ge 2^d$:

$$
a_i^{(d+1)} = a_i^{(d)} + a_{i - 2^d}^{(d)}
$$

- **Work**: $T_1 = O(n \log n)$ (not work-efficient).
- **Span**: $T_\infty = O(\log n)$.
- **Parallelism**: $O(n)$.

!!! tip "Choosing between algorithms"
    Blelloch's algorithm is work-efficient ($O(n)$ work) and preferred when processor count is limited. Hillis-Steele performs more total work but has simpler synchronization, making it suitable for GPU architectures with abundant parallelism.

## Applications

- **Stream compaction**: Filter elements satisfying a predicate. Prefix sum computes output positions for each kept element.
- **Radix sort**: Counting sort at each digit uses prefix sum to compute output positions in parallel.
- **Load balancing**: Distribute work evenly by computing prefix sums of task sizes.
- **Parallel BFS**: Compute frontier offsets for the next level.

## Complexity Summary

| Algorithm | Work $T_1$ | Span $T_\infty$ | Parallelism |
|---|---|---|---|
| Sequential scan | $O(n)$ | $O(n)$ | $O(1)$ |
| Blelloch (work-efficient) | $O(n)$ | $O(\log n)$ | $O(n / \log n)$ |
| Hillis-Steele | $O(n \log n)$ | $O(\log n)$ | $O(n)$ |

## Reference

- Blelloch, G. E. (1990). "Prefix sums and their applications." *Tech Report CMU-CS-90-190*.
- Hillis, W. D. and Steele, G. L. (1986). "Data parallel algorithms." *Communications of the ACM*, 29(12), 1170--1183.
