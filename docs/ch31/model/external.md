# External Memory Model

Algorithms that operate on datasets too large for main memory face a fundamental bottleneck: the time to transfer data between disk and RAM dwarfs the time for in-memory computation. A single random disk access can take millions of CPU cycles, so the number of disk reads and writes -- not the number of comparisons or additions -- determines practical performance. The **external memory model** (also called the disk-access model or I/O model) formalizes this reality by counting only block transfers between two storage levels.

## The Two-Level Storage Hierarchy

The model assumes exactly two levels of storage:

1. **Internal memory (RAM):** fast but limited, holding at most $M$ data elements.
2. **External storage (disk):** slow but unlimited, holding the full dataset of $N$ elements.

Computation can only be performed on data that resides in internal memory. When the algorithm needs data that is on disk, it must execute an **I/O operation** that transfers a contiguous **block** of $B$ elements between disk and memory.

## Formal Definition

The external memory model is parameterized by three quantities:

| Parameter | Symbol | Meaning |
|---|---|---|
| Problem size | $N$ | Total number of data elements |
| Memory size | $M$ | Elements that fit in internal memory |
| Block size | $B$ | Elements transferred per I/O operation |

These satisfy the constraint:

$$
1 \le B \le M \le N
$$

An **I/O operation** (or block transfer) moves one block of $B$ contiguous elements between disk and memory. The **I/O complexity** of an algorithm is the total number of such operations it performs in the worst case.

## Why Not Just Count Operations?

Consider sorting $N = 10^9$ integers when $M = 10^6$ fit in memory. An in-memory sort would need $O(N \log N) \approx 3 \times 10^{10}$ comparisons -- fast on a modern CPU. But if each comparison requires a random disk access, the wall-clock time is dominated by roughly $10^9$ disk seeks at 5 ms each, totaling over 50 days. An I/O-efficient sort reduces disk accesses to $O((N/B) \log_{M/B}(N/B))$, which with $B = 4096$ takes only minutes. The external memory model captures this distinction by measuring only block transfers.

## The I/O Operation

Each I/O operation performs one of two actions:

- **Read:** Transfer a block of $B$ elements from a specified disk location into memory.
- **Write:** Transfer a block of $B$ elements from memory to a specified disk location.

Between I/O operations, the algorithm may perform any amount of internal computation on the data in memory at zero cost. This reflects the practical reality that CPU time is negligible compared to disk transfer time.

## Key Derived Quantities

Several ratios of the model parameters appear repeatedly in I/O complexity bounds:

$$
\frac{N}{B} \quad \text{(number of blocks in the dataset)}
$$

$$
\frac{M}{B} \quad \text{(number of blocks that fit in memory)}
$$

The ratio $M/B$ is especially important because it determines the **fan-out** of merge-based algorithms: during an external merge sort, we can merge up to $M/B - 1$ sorted runs simultaneously.

## Comparison with Other Models

| Model | Cost Metric | Strengths |
|---|---|---|
| RAM model | Arithmetic operations | Simple analysis, captures CPU work |
| External memory | Block transfers | Captures disk I/O bottleneck |
| Cache-oblivious | Block transfers (all $B$, $M$) | No parameter tuning needed |

The external memory model requires the algorithm to know $B$ and $M$ explicitly and optimize for those specific values. The cache-oblivious model, by contrast, achieves optimal I/O complexity for all values of $B$ and $M$ without knowing them -- a stronger guarantee explored on the [Cache-Oblivious B-Trees](../structures/cache_oblivious.md) page.

## Example: Scanning vs Random Access

The following example contrasts sequential scanning (I/O-efficient) with random access (I/O-inefficient) to demonstrate why the external memory model matters.

```python
"""
External memory model simulation.

Compares sequential scanning versus random access patterns
to illustrate the importance of block-aligned I/O.
"""

import math
import random

# ===================================================================
# I/O cost calculations
# ===================================================================

def sequential_scan_ios(n: int, b: int) -> int:
    """I/O operations for a full sequential scan of N elements."""
    return math.ceil(n / b)


def random_access_ios(n: int, num_accesses: int) -> int:
    """
    Worst-case I/O operations for random element accesses.

    Each random access may hit a different block, costing 1 I/O each.
    """
    return min(num_accesses, math.ceil(n / 1))  # 1 I/O per access worst case


def external_sort_ios(n: int, m: int, b: int) -> int:
    """I/O operations for external merge sort."""
    if n <= m:
        return sequential_scan_ios(n, b)  # Fits in memory
    blocks = math.ceil(n / b)
    fan_out = m // b
    if fan_out <= 1:
        return float('inf')
    passes = math.ceil(math.log(blocks) / math.log(fan_out))
    return 2 * blocks * passes  # Each pass reads and writes all blocks


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    N = 10**8
    M = 10**6
    B = 4096

    print(f"External Memory Model Parameters")
    print(f"  N = {N:>12,}  (problem size)")
    print(f"  M = {M:>12,}  (memory size)")
    print(f"  B = {B:>12,}  (block size)")
    print(f"  N/B = {N // B:>10,}  (blocks in dataset)")
    print(f"  M/B = {M // B:>10,}  (blocks in memory)")
    print()

    scan = sequential_scan_ios(N, B)
    sort = external_sort_ios(N, M, B)
    rand = random_access_ios(N, N)

    print(f"Operation costs (I/O operations):")
    print(f"  Sequential scan:  {scan:>12,}")
    print(f"  External sort:    {sort:>12,}")
    print(f"  N random accesses: {rand:>11,}")
```

??? example "Sample Output"

    ```
    External Memory Model Parameters
      N =  100,000,000  (problem size)
      M =    1,000,000  (memory size)
      B =        4,096  (block size)
      N/B =     24,415  (blocks in dataset)
      M/B =        244  (blocks in memory)

    Operation costs (I/O operations):
      Sequential scan:       24,415
      External sort:         97,660
      N random accesses: 100,000,000
    ```

    Sequential scanning is roughly 4,000 times more I/O-efficient than random access, which is why the external memory model focuses on block transfer patterns.

## Connection to Other Pages

The block size parameter $B$ is discussed in detail on the [Block Size](block.md) page. The I/O complexity bounds for fundamental operations -- scanning, sorting, and searching -- are derived on the [I/O Complexity](io_complexity.md) page.

## Reference

- Aggarwal, A. & Vitter, J. S. "The Input/Output Complexity of Sorting and Related Problems," *Communications of the ACM*, 31(9), 1988.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, Foundations and Trends in Theoretical Computer Science, 2008.
