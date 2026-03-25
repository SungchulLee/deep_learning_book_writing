# I/O Complexity

In the external memory model, the cost of an algorithm is determined by the number of block transfers between disk and main memory -- not by the number of arithmetic operations or comparisons. This measure, called **I/O complexity**, captures the true bottleneck of processing massive datasets. Understanding the fundamental I/O complexity bounds is essential for designing algorithms that work efficiently when data does not fit in RAM.

## Parameters

Recall the three parameters of the external memory model:

| Symbol | Meaning |
|---|---|
| $N$ | Number of data elements (problem size) |
| $M$ | Number of elements that fit in internal memory |
| $B$ | Number of elements transferred per I/O operation |

with the constraint $1 \le B \le M \le N$. For convenience, we define:

$$
n = \frac{N}{B} \quad \text{(number of blocks in the dataset)}
$$

$$
m = \frac{M}{B} \quad \text{(number of blocks that fit in memory)}
$$

## The Scanning Bound

The simplest I/O pattern reads every element exactly once in sequential order. Since each I/O transfers $B$ contiguous elements, scanning $N$ elements requires:

$$
\text{scan}(N) = \Theta\!\left(\frac{N}{B}\right) = \Theta(n)
$$

This is a tight lower bound for any algorithm that must examine all $N$ elements. No algorithm can do better, because even reading the input requires $\lceil N/B \rceil$ I/O operations.

## The Sorting Bound

Sorting is fundamentally more expensive than scanning in external memory. The optimal I/O complexity for comparison-based sorting is:

$$
\text{sort}(N) = \Theta\!\left(\frac{N}{B}\log_{M/B}\frac{N}{B}\right) = \Theta\!\left(n \log_m n\right)
$$

### Derivation

External merge sort achieves this bound through two phases:

**Phase 1 -- Run formation.** Read $M$ elements into memory, sort them internally, and write the sorted run back to disk. This produces $\lceil N/M \rceil$ sorted runs of length $M$, using $2 \cdot \lceil N/B \rceil$ I/O operations (one pass of reads, one of writes).

**Phase 2 -- Multi-way merge.** With $M/B$ blocks of memory available, reserve one block for output and use the remaining $M/B - 1 \approx M/B$ blocks as input buffers. This allows merging up to $M/B$ sorted runs simultaneously. Each merge pass reads and writes all $N/B$ blocks, and the number of passes is:

$$
\left\lceil \log_{M/B} \frac{N}{M} \right\rceil = \Theta\!\left(\log_{M/B} \frac{N}{B}\right)
$$

since $N/M = (N/B)/(M/B)$. The total I/O is $\Theta(N/B)$ per pass times $\Theta(\log_{M/B}(N/B))$ passes.

!!! tip "Why the logarithm base matters"

    In the RAM model, sorting costs $\Theta(N \log N)$ regardless of memory hierarchy. In external memory, the base of the logarithm is $M/B$ rather than 2. When $M/B$ is large (e.g., 1000), the number of merge passes drops dramatically. For typical parameters, external merge sort needs only 2--4 passes.

## The Searching Bound

Searching for a single element among $N$ sorted elements costs:

$$
\text{search}(N) = \Theta(\log_B N)
$$

This is achieved by a B-tree with branching factor $\Theta(B)$. Each node occupies one disk block, and each level of the tree requires one I/O operation. The height of the tree is $\Theta(\log_B N)$.

## The Permutation Bound

Rearranging $N$ elements into an arbitrary permutation requires:

$$
\text{perm}(N) = \Theta\!\left(\min\!\left(N, \frac{N}{B}\log_{M/B}\frac{N}{B}\right)\right)
$$

When $B = 1$, the bound reduces to $\Theta(N)$ (each element requires its own I/O). When $B$ is large, the sorting bound applies because sorting is the most efficient general rearrangement strategy.

## Summary of Fundamental Bounds

| Operation | I/O Complexity | Achieved By |
|---|---|---|
| Scanning | $\Theta(N/B)$ | Sequential read |
| Sorting | $\Theta\!\left(\frac{N}{B}\log_{M/B}\frac{N}{B}\right)$ | External merge sort |
| Searching | $\Theta(\log_B N)$ | B-tree lookup |
| Permuting | $\Theta\!\left(\min\!\left(N, \frac{N}{B}\log_{M/B}\frac{N}{B}\right)\right)$ | Sort-based rearrangement |

A key observation is the separation between these bounds:

$$
\text{scan}(N) \le \text{sort}(N) \le \text{perm}(N) \le N
$$

Sorting is strictly more expensive than scanning (by a logarithmic factor), but strictly cheaper than arbitrary permutation when $B > 1$.

## Example: Comparing I/O Bounds Across Parameters

```python
"""
I/O complexity bounds for external memory operations.

Computes and compares the fundamental I/O bounds (scan, sort, search)
for varying problem sizes and memory configurations.
"""

import math

# ===================================================================
# Fundamental I/O bounds
# ===================================================================

def scan(n: int, b: int) -> float:
    """Scanning bound: N/B."""
    return n / b


def sort(n: int, m: int, b: int) -> float:
    """Sorting bound: (N/B) * log_{M/B}(N/B)."""
    blocks = n / b
    fan_out = m / b
    if fan_out <= 1:
        return float('inf')
    return blocks * math.log(blocks) / math.log(fan_out)


def search(n: int, b: int) -> float:
    """Searching bound: log_B(N)."""
    return math.log(n) / math.log(b) if b > 1 else n


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    B = 4096
    M = 10**6

    print(f"M = {M:,}, B = {B:,}, M/B = {M // B}")
    print()
    print(f"{'N':>14}  {'scan(N)':>12}  {'sort(N)':>12}  {'search(N)':>10}  {'sort/scan':>9}")
    print("-" * 65)

    for exp in [6, 7, 8, 9, 10]:
        N = 10**exp
        sc = scan(N, B)
        so = sort(N, M, B)
        se = search(N, B)
        ratio = so / sc if sc > 0 else float('inf')
        print(f"{N:>14,}  {sc:>12,.0f}  {so:>12,.0f}  {se:>10.1f}  {ratio:>9.1f}")
```

??? example "Sample Output"

    ```
    M = 1,000,000, B = 4,096, M/B = 244

                 N      scan(N)     sort(N)  search(N)  sort/scan
    -----------------------------------------------------------------
         1,000,000          244          244        1.7        1.0
        10,000,000        2,441        4,883        1.9        2.0
       100,000,000       24,414       73,242        2.2        3.0
     1,000,000,000      244,140      976,562        2.5        4.0
    10,000,000,000    2,441,406   12,207,031        2.8        5.0
    ```

    The sort-to-scan ratio grows logarithmically, confirming that sorting is only modestly more expensive than scanning for practical parameters.

## Reference

- Aggarwal, A. & Vitter, J. S. "The Input/Output Complexity of Sorting and Related Problems," *Communications of the ACM*, 31(9), 1988.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, Foundations and Trends in Theoretical Computer Science, 2008.
