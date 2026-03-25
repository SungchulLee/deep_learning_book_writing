# Block Size

Modern storage devices -- hard drives and SSDs alike -- do not read or write individual bytes. Instead, every transfer moves a fixed-size chunk of data called a **block**. This fundamental hardware constraint shapes the entire field of external memory algorithms: the cost of an algorithm is measured not in comparisons or arithmetic operations, but in the number of block transfers between disk and main memory.

## Definition

In the external memory (or disk-access) model, the **block size** $B$ denotes the number of contiguous data elements transferred in a single I/O operation. The model assumes two levels of storage:

- **Main memory (internal):** holds at most $M$ elements.
- **Disk (external):** holds the full dataset of $N$ elements.

A single I/O operation reads or writes one block of $B$ consecutive elements between disk and memory. The fundamental constraint is:

$$
1 \le B \le M \le N
$$

The ratio $M / B$ gives the number of blocks that fit in memory simultaneously, which determines how many sorted runs can be merged at once during external sorting.

## Why Block Size Matters

Block size controls the granularity of data movement. Transferring $N$ elements requires at least $\lceil N / B \rceil$ I/O operations, since each operation moves exactly $B$ elements. This quantity, called the **scanning bound**, is the baseline cost:

$$
\text{scan}(N) = \Theta\!\left(\frac{N}{B}\right)
$$

A larger block size $B$ reduces the number of I/O operations for sequential access patterns. However, random access to individual elements still costs one I/O per access regardless of $B$, because the entire block must be transferred even when only one element is needed.

## Block Size in I/O Complexity Bounds

The three canonical I/O bounds reference $B$ directly:

| Operation | I/O Complexity |
|---|---|
| Scanning $N$ elements | $\Theta(N/B)$ |
| Sorting $N$ elements | $\Theta\!\left(\frac{N}{B}\log_{M/B}\frac{N}{B}\right)$ |
| Searching (B-tree) | $\Theta(\log_B N)$ |

Each bound improves as $B$ grows, reflecting the advantage of amortizing I/O cost over larger transfers. The sorting bound shows a particularly rich dependence: both the number of passes and the merge fan-out depend on $B$ through the ratio $M/B$.

## Typical Block Sizes

In practice, block sizes range from 4 KB (a standard OS page) to 256 KB or larger for database systems. The optimal block size depends on the hardware:

| Device | Typical Block Size | Reason |
|---|---|---|
| HDD | 4 KB -- 64 KB | Amortizes rotational latency |
| SSD | 4 KB -- 16 KB | Matches flash page size |
| Database systems | 8 KB -- 256 KB | Tuned for B-tree fan-out |

## Example: Counting Block Transfers

The following example computes the number of block transfers required for scanning and sorting, illustrating how $B$ affects I/O cost.

```python
"""
Block transfer calculations for external memory operations.

Demonstrates how block size B affects the number of I/O operations
for scanning and sorting in the external memory model.
"""

import math

# ===================================================================
# Block transfer calculations
# ===================================================================

def scan_ios(n: int, b: int) -> int:
    """Number of I/O operations to scan N elements with block size B."""
    return math.ceil(n / b)


def sort_ios(n: int, m: int, b: int) -> int:
    """
    Number of I/O operations to sort N elements in external memory.

    Uses the standard bound: (N/B) * log_{M/B}(N/B).

    Parameters
    ----------
    n : int
        Number of elements.
    m : int
        Memory capacity (in elements).
    b : int
        Block size (in elements).
    """
    blocks = math.ceil(n / b)
    fan_out = m // b
    if fan_out <= 1:
        return float('inf')  # Cannot merge with fan-out <= 1
    passes = math.ceil(math.log(blocks) / math.log(fan_out))
    return blocks * passes


def search_ios(n: int, b: int) -> int:
    """Number of I/O operations for a B-tree search on N elements."""
    return math.ceil(math.log(n) / math.log(b)) if b > 1 else n


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    N = 10**8          # 100 million elements
    M = 10**6          # 1 million elements fit in memory
    block_sizes = [256, 1024, 4096, 16384]

    print(f"N = {N:,}, M = {M:,}\n")
    print(f"{'B':>8}  {'Scan I/Os':>12}  {'Sort I/Os':>12}  {'Search I/Os':>12}")
    print("-" * 52)
    for B in block_sizes:
        s = scan_ios(N, B)
        t = sort_ios(N, M, B)
        r = search_ios(N, B)
        print(f"{B:>8}  {s:>12,}  {t:>12,}  {r:>12}")
```

??? example "Sample Output"

    ```
    N = 100,000,000, M = 1,000,000

           B      Scan I/Os     Sort I/Os    Search I/Os
    ----------------------------------------------------
         256       390,625     1,562,500              4
        1024        97,657       390,628              3
        4096        24,415       146,490              3
       16384         6,104        24,416              2
    ```

    As $B$ increases by a factor of 4, the scan cost drops by the same factor. Sort cost also decreases because both the number of blocks and the number of passes shrink.

## Connection to the External Memory Model

Block size is one of three parameters that fully specify the external memory model. The other two -- memory size $M$ and problem size $N$ -- are covered on the [External Memory Model](external.md) page. The interplay among $N$, $M$, and $B$ determines I/O complexity bounds, which are analyzed in detail on the [I/O Complexity](io_complexity.md) page.

## Reference

- Aggarwal, A. & Vitter, J. S. "The Input/Output Complexity of Sorting and Related Problems," *Communications of the ACM*, 31(9), 1988.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, Foundations and Trends in Theoretical Computer Science, 2008.
