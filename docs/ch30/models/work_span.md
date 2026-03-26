# Work-Span Model

When analyzing parallel algorithms, we need a model that captures both the total amount of computation and the inherent sequential bottleneck. The **work-span model** (also called the work-depth model) provides exactly this: it measures the total number of operations (work) and the length of the longest chain of dependencies (span), yielding a clean upper bound on parallel running time through Brent's theorem.

## Definitions

Let $T_p$ denote the running time of a parallel algorithm on $p$ processors.

**Work** $T_1$ is the total number of operations executed when the algorithm runs on a single processor. It equals the sequential running time.

**Span** $T_\infty$ (also called *depth* or *critical-path length*) is the running time on an unbounded number of processors. It measures the longest chain of sequentially dependent operations in the computation DAG.

**Parallelism** is the ratio of work to span:

$$
P = \frac{T_1}{T_\infty}
$$

Parallelism represents the maximum useful number of processors: adding processors beyond $P$ yields no further speedup.

## Computation DAG

A parallel computation can be modeled as a directed acyclic graph (DAG) where each node represents a constant-time operation and each edge represents a dependency. In this model:

- **Work** $T_1$ equals the total number of nodes in the DAG.
- **Span** $T_\infty$ equals the length of the longest path (the critical path) in the DAG.

## Brent's Theorem

Brent's theorem provides an upper bound on the parallel execution time given a fixed number of processors.

**Theorem (Brent, 1974).** For any computation with work $T_1$ and span $T_\infty$, the running time on $p$ processors satisfies:

$$
T_p \le \frac{T_1}{p} + T_\infty
$$

??? note "Proof sketch"
    Partition the computation DAG into $T_\infty$ levels, where level $i$ contains all nodes whose longest incoming path has length $i$. Let $m_i$ be the number of nodes at level $i$. With $p$ processors, level $i$ takes $\lceil m_i / p \rceil$ time steps. Summing over all levels:

    $$
    T_p \le \sum_{i=1}^{T_\infty} \left\lceil \frac{m_i}{p} \right\rceil \le \sum_{i=1}^{T_\infty} \left( \frac{m_i}{p} + 1 \right) = \frac{T_1}{p} + T_\infty
    $$

    since $\sum_i m_i = T_1$. $\square$

Two important corollaries follow from Brent's theorem:

1. **Linear speedup region**: When $p \le T_1 / T_\infty$, the term $T_1 / p$ dominates, and $T_p \approx T_1 / p$, giving near-linear speedup.
2. **Diminishing returns**: When $p > T_1 / T_\infty$, the span $T_\infty$ dominates, and adding more processors provides little benefit.

## Speedup and Efficiency

**Speedup** on $p$ processors is the ratio of sequential to parallel running time:

$$
S_p = \frac{T_1}{T_p}
$$

**Efficiency** measures how well the processors are utilized:

$$
E_p = \frac{S_p}{p} = \frac{T_1}{p \cdot T_p}
$$

An algorithm achieves **perfect linear speedup** when $S_p = p$ and $E_p = 1$.

## Example: Parallel Reduction

Consider summing an array of $n$ elements. Sequential summation performs $n - 1$ additions. A parallel tree-based reduction pairs elements at each level, halving the problem size.

- **Work**: $T_1 = n - 1$ (every element must be added).
- **Span**: $T_\infty = \lceil \log_2 n \rceil$ (the depth of the reduction tree).
- **Parallelism**: $P = (n - 1) / \lceil \log_2 n \rceil \in \Theta(n / \log n)$.

```python
"""
Work-span analysis: parallel tree reduction for array summation.

Demonstrates computing work and span for a parallel reduction,
then applies Brent's theorem to estimate parallel running time.
"""

import math

# ===================================================================
# Work and Span Computation
# ===================================================================

def compute_work_span_reduction(n):
    """Compute work and span for parallel reduction on n elements.

    Args:
        n: number of elements to sum

    Returns:
        Tuple of (work, span, parallelism)
    """
    work = n - 1
    span = math.ceil(math.log2(n))
    parallelism = work / span if span > 0 else float("inf")
    return work, span, parallelism


def brent_bound(work, span, p):
    """Compute Brent's upper bound on T_p.

    Args:
        work: total operations T_1
        span: critical path length T_∞
        p: number of processors

    Returns:
        Upper bound on parallel running time
    """
    return work / p + span


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    n = 1024
    work, span, parallelism = compute_work_span_reduction(n)

    print(f"Array size:   n = {n}")
    print(f"Work T_1:     {work}")
    print(f"Span T_inf:   {span}")
    print(f"Parallelism:  {parallelism:.1f}")
    print()

    for p in [1, 4, 16, 64, 256]:
        tp = brent_bound(work, span, p)
        speedup = work / tp
        efficiency = speedup / p
        print(f"p={p:>3}: T_p <= {tp:>8.1f}, "
              f"speedup = {speedup:>6.2f}, "
              f"efficiency = {efficiency:.3f}")
```

**Output:**
```
Array size:   n = 1024
Work T_1:     1023
Span T_inf:   10
Parallelism:  102.3

p=  1: T_p <=   1033.0, speedup =   0.99, efficiency = 0.990
p=  4: T_p <=    265.8, speedup =   3.85, efficiency = 0.962
p= 16: T_p <=     73.9, speedup =  13.83, efficiency = 0.864
p= 64: T_p <=     26.0, speedup =  39.37, efficiency = 0.615
p=256: T_p <=     14.0, speedup =  73.07, efficiency = 0.285
```

!!! tip "Reading the output"
    As the number of processors increases, speedup grows but efficiency drops. Beyond the parallelism threshold ($p \approx 102$), adding processors yields diminishing returns because the span dominates.

## Common Work-Span Results

| Algorithm | Work $T_1$ | Span $T_\infty$ | Parallelism |
|---|---|---|---|
| Parallel reduction | $O(n)$ | $O(\log n)$ | $O(n / \log n)$ |
| Parallel prefix sum | $O(n)$ | $O(\log n)$ | $O(n / \log n)$ |
| Parallel merge sort | $O(n \log n)$ | $O(\log^2 n)$ | $O(n / \log n)$ |
| Matrix multiply (naive) | $O(n^3)$ | $O(\log n)$ | $O(n^3 / \log n)$ |

## Reference

- Brent, R. P. (1974). "The parallel evaluation of general arithmetic expressions." *Journal of the ACM*, 21(2), 201--206.
- Cormen, T. H. et al. *Introduction to Algorithms*, Chapter 27 (Multithreaded Algorithms).
