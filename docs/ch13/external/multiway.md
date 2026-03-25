# Multi-Way Merge

Two-way external merge sort merges pairs of runs, requiring $O(\log_2(N/M))$ passes
over the data.  Each pass reads and writes every element, so reducing the number of
passes directly reduces I/O cost.  **Multi-way merge** (or $k$-way merge) merges $k$
runs simultaneously using a min-heap (priority queue), increasing the logarithm's base
from 2 to $k$ and dramatically cutting the number of passes for large datasets.

## Why Multi-Way Merge Matters

With memory $M$ and block size $B$, we can afford $k = \lfloor M/B \rfloor - 1$ input
buffers (reserving one buffer for output).  Each input buffer holds one block from a
different run.  By merging $k$ runs at once, the number of merge passes drops from
$\lceil \log_2 (N/M) \rceil$ to $\lceil \log_k (N/M) \rceil$.

**Example:** Sorting 1 TB with 1 GB memory and 4 KB blocks gives
$k \approx 250{,}000$.  Two-way merge needs about 20 passes; multi-way merge needs
just 1 or 2 passes.

## Algorithm Overview

1. **Run formation** (same as two-way): create $\lceil N/M \rceil$ sorted runs.
2. **Multi-way merge pass:**
    - Open $k$ runs for reading, loading one block from each into memory.
    - Insert the first element of each block into a min-heap of size $k$.
    - Repeatedly extract the minimum from the heap, write it to the output buffer,
      and insert the next element from the same run.
    - When an input buffer is exhausted, read the next block from that run.
    - When the output buffer is full, write it to disk.
3. Repeat until $\lceil N/M \rceil / k$ runs remain, then merge again (if needed).

## I/O Complexity

Each merge pass reads and writes all $N$ elements: $2 \lceil N/B \rceil$ I/Os.
With merge factor $k = \lfloor M/B \rfloor - 1$, the number of passes is
$\lceil \log_k (N/M) \rceil$.

$$
\text{I/O}(N, M, B) = O\!\left(\frac{N}{B} \log_{M/B} \frac{N}{M}\right)
$$

This is known to be **asymptotically optimal** in the external memory model -- no
comparison-based external sorting algorithm can do better.

The in-memory cost per merge step is $O(\log k)$ for the heap operation, giving total
comparison complexity:

$$
T(N, k) = O\!\left(N \log_k \frac{N}{M} \cdot \log k\right) = O(N \log N)
$$

The comparison count matches internal sorting, confirming that the bottleneck is I/O,
not computation.

## Worked Example

Sort $N = 12{,}000$ elements with $M = 3{,}000$ and $B = 1{,}000$.

**Run formation:** 4 sorted runs of 3,000 elements each. Cost: $2 \times 12 = 24$ I/Os.

**Merge:** $k = \lfloor 3{,}000/1{,}000 \rfloor - 1 = 2$ input buffers.

- With $k = 2$: 2 passes, costing $2 \times 24 = 48$ I/Os. Total: 72 I/Os.

Now suppose $B = 500$, so $k = \lfloor 3{,}000/500 \rfloor - 1 = 5$:

- All 4 runs merge in a single pass: $2 \times 24 = 48$ I/Os. Total: 72 I/Os.

With larger memory ($M = 6{,}000$): only 2 runs, one merge pass suffices regardless.

## Min-Heap for k-Way Merge

The min-heap stores entries of the form $(value, run\_index)$.  At each step:

1. Extract-min gives the globally smallest unprocessed element and identifies which
   run it came from.
2. Insert the next element from that run (cost $O(\log k)$).

After $N$ total extractions, the heap has performed $O(N \log k)$ work.

## Implementation

```python
"""
Multi-way merge -- merges k sorted runs using a min-heap.

Reduces the number of merge passes in external sorting from
log_2(N/M) to log_k(N/M) where k = M/B - 1.
Time:  O(N log k) comparisons per pass
I/O:   O((N/B) * log_{M/B}(N/M)) block transfers
"""

import heapq
from typing import Iterator


# === k-way merge using a min-heap ==========================================

def k_way_merge(sorted_runs: list[list[int]]) -> list[int]:
    """Merge k sorted lists into a single sorted list.

    Parameters
    ----------
    sorted_runs : list[list[int]]
        A list of k sorted lists (runs).

    Returns
    -------
    list[int]
        Merged sorted list.
    """
    heap: list[tuple[int, int, int]] = []

    # Initialize heap with first element of each run
    for run_idx, run in enumerate(sorted_runs):
        if run:
            heapq.heappush(heap, (run[0], run_idx, 0))

    result: list[int] = []
    while heap:
        val, run_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # Push next element from the same run
        next_idx = elem_idx + 1
        if next_idx < len(sorted_runs[run_idx]):
            heapq.heappush(
                heap,
                (sorted_runs[run_idx][next_idx], run_idx, next_idx),
            )

    return result


# === External sort with multi-way merge =====================================

def external_sort_multiway(
    data: list[int], memory_size: int, merge_factor: int
) -> list[int]:
    """Sort data using multi-way external merge sort.

    Parameters
    ----------
    data : list[int]
        Input data.
    memory_size : int
        Max elements in memory at once.
    merge_factor : int
        Number of runs to merge simultaneously (k).

    Returns
    -------
    list[int]
        Sorted data.
    """
    # Phase 1: create sorted runs
    runs: list[list[int]] = []
    for start in range(0, len(data), memory_size):
        runs.append(sorted(data[start : start + memory_size]))

    # Phase 2: multi-way merge passes
    while len(runs) > 1:
        next_runs: list[list[int]] = []
        for i in range(0, len(runs), merge_factor):
            batch = runs[i : i + merge_factor]
            merged = k_way_merge(batch)
            next_runs.append(merged)
        runs = next_runs

    return runs[0] if runs else []


# === Demo ===================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    data = random.sample(range(10000), 100)

    # 2-way merge
    sorted_2way = external_sort_multiway(data, memory_size=20, merge_factor=2)
    print(f"2-way merge correct: {sorted_2way == sorted(data)}")

    # 5-way merge
    sorted_5way = external_sort_multiway(data, memory_size=20, merge_factor=5)
    print(f"5-way merge correct: {sorted_5way == sorted(data)}")

    # Show merge passes needed
    import math
    num_runs = math.ceil(len(data) / 20)
    for k in [2, 5, 10]:
        passes = math.ceil(math.log(num_runs) / math.log(k)) if num_runs > 1 else 0
        print(f"  k={k:2d}: {passes} merge pass(es) for {num_runs} runs")
```

**Output:**
```
2-way merge correct: True
5-way merge correct: True
  k= 2: 3 merge pass(es) for 5 runs
  k= 5: 1 merge pass(es) for 5 runs
  k=10: 1 merge pass(es) for 5 runs
```

## Practical Considerations

| Factor | Impact |
|--------|--------|
| Increasing $k$ | Fewer passes but more heap overhead per step |
| Very large $k$ | Diminishing returns; disk seek time dominates |
| SSD vs HDD | SSD tolerates higher $k$ due to fast random access |
| Double buffering | Overlap I/O with computation for each of the $k$ streams |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and
  Searching* (2nd ed.), Section 5.4. Addison-Wesley.
