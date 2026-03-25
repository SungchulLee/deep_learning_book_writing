# External Sorting

When a dataset of $N$ elements does not fit in main memory of size $M$, in-memory sorting algorithms like quicksort or mergesort cannot be applied directly. Each random access would require a separate disk I/O, resulting in $O(N \log N)$ I/O operations -- far worse than necessary. **External merge sort** solves this problem by organizing disk access into sequential scans and multi-way merges, achieving the optimal I/O complexity of $O((N/B) \log_{M/B}(N/B))$.

## Algorithm Overview

External merge sort proceeds in two phases:

1. **Run formation:** Partition the input into sorted chunks (runs) that each fit in memory.
2. **Multi-way merge:** Repeatedly merge groups of runs until only one sorted run remains.

## Phase 1: Run Formation

Read $M$ elements into memory, sort them using any in-memory algorithm (e.g., quicksort), and write the sorted run back to disk. Repeat for the next $M$ elements until all $N$ elements have been processed.

This produces $\lceil N/M \rceil$ sorted runs, each of length at most $M$. The I/O cost is:

$$
2 \cdot \left\lceil \frac{N}{B} \right\rceil
$$

because each element is read once and written once during this phase.

!!! tip "Replacement selection"

    The replacement selection technique can produce runs of expected length $2M$ (instead of $M$) by using a priority queue that outputs the minimum element and replaces it with the next input element, as long as the new element continues the current run. This reduces the number of initial runs by roughly half.

## Phase 2: Multi-Way Merge

With $M$ bytes of memory and blocks of size $B$, we can buffer $M/B$ blocks simultaneously. Reserving one block for output, we can merge up to:

$$
f = \frac{M}{B} - 1 \approx \frac{M}{B}
$$

sorted runs in a single pass. Each merge pass reads and writes all $N/B$ blocks, costing $2 \lceil N/B \rceil$ I/O operations.

### Number of Passes

Starting from $\lceil N/M \rceil$ runs and merging $f$ at a time, the number of merge passes is:

$$
p = \left\lceil \log_f \frac{N}{M} \right\rceil = \left\lceil \log_{M/B} \frac{N}{B} \right\rceil
$$

since $N/M = (N/B)/(M/B)$ and $f \approx M/B$.

### Total I/O Complexity

Each of the $p$ passes costs $O(N/B)$ I/O operations. Including the initial run formation pass:

$$
\text{Total I/O} = O\!\left(\frac{N}{B} \cdot \log_{M/B} \frac{N}{B}\right) = O(\text{sort}(N))
$$

This bound is **optimal** for comparison-based sorting in the external memory model.

## Practical Pass Counts

For typical parameters, the number of merge passes is remarkably small:

| $N$ | $M$ | $B$ | $M/B$ (fan-out) | Passes |
|---|---|---|---|---|
| $10^8$ | $10^6$ | $4096$ | 244 | 2 |
| $10^{10}$ | $10^6$ | $4096$ | 244 | 3 |
| $10^{12}$ | $10^6$ | $4096$ | 244 | 4 |
| $10^{12}$ | $10^8$ | $4096$ | 24414 | 2 |

Even for a trillion elements, external merge sort needs only 3--4 passes when the fan-out $M/B$ is in the hundreds.

## Example: External Merge Sort Simulation

```python
"""
External merge sort simulation.

Demonstrates the two-phase approach: run formation followed by
multi-way merge, tracking I/O operations at each stage.
"""

import math
import heapq

# ===================================================================
# External merge sort (simulated)
# ===================================================================

def external_merge_sort(data: list[int], memory_size: int,
                        block_size: int) -> tuple[list[int], dict]:
    """
    Simulate external merge sort on a list of integers.

    Parameters
    ----------
    data : The input data (simulates disk-resident data).
    memory_size : Number of elements that fit in memory (M).
    block_size : Elements per block transfer (B).

    Returns
    -------
    Tuple of (sorted data, I/O statistics dict).
    """
    n = len(data)
    io_count = 0

    # Phase 1: Run formation
    runs = []
    for start in range(0, n, memory_size):
        end = min(start + memory_size, n)
        chunk = data[start:end]
        io_count += math.ceil(len(chunk) / block_size)  # Read
        chunk.sort()  # In-memory sort (free)
        runs.append(chunk)
        io_count += math.ceil(len(chunk) / block_size)  # Write

    phase1_ios = io_count
    num_initial_runs = len(runs)

    # Phase 2: Multi-way merge
    fan_out = max(2, memory_size // block_size - 1)
    merge_pass = 0

    while len(runs) > 1:
        merge_pass += 1
        new_runs = []
        for i in range(0, len(runs), fan_out):
            group = runs[i:i + fan_out]

            # Merge this group using a min-heap
            merged = []
            heap = []
            for run_idx, run in enumerate(group):
                if run:
                    heapq.heappush(heap, (run[0], run_idx, 0))

            while heap:
                val, run_idx, pos = heapq.heappop(heap)
                merged.append(val)
                if pos + 1 < len(group[run_idx]):
                    heapq.heappush(
                        heap,
                        (group[run_idx][pos + 1], run_idx, pos + 1)
                    )

            # Count I/O: read all input blocks + write all output blocks
            total_elements = sum(len(r) for r in group)
            io_count += 2 * math.ceil(total_elements / block_size)
            new_runs.append(merged)

        runs = new_runs

    stats = {
        "n": n,
        "memory_size": memory_size,
        "block_size": block_size,
        "fan_out": fan_out,
        "initial_runs": num_initial_runs,
        "merge_passes": merge_pass,
        "phase1_ios": phase1_ios,
        "total_ios": io_count,
    }

    return runs[0] if runs else [], stats


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    N = 10000
    M = 1000
    B = 100

    data = [random.randint(0, 10**6) for _ in range(N)]
    sorted_data, stats = external_merge_sort(data, M, B)

    # Verify correctness
    assert sorted_data == sorted(data), "Sort failed!"

    print(f"External Merge Sort Simulation")
    print(f"  N = {stats['n']:,}")
    print(f"  M = {stats['memory_size']:,}")
    print(f"  B = {stats['block_size']}")
    print(f"  Fan-out (M/B - 1): {stats['fan_out']}")
    print(f"  Initial runs:      {stats['initial_runs']}")
    print(f"  Merge passes:      {stats['merge_passes']}")
    print(f"  Phase 1 I/Os:      {stats['phase1_ios']}")
    print(f"  Total I/Os:        {stats['total_ios']}")

    # Theoretical bound
    blocks = math.ceil(N / B)
    theoretical = blocks * math.ceil(
        math.log(blocks) / math.log(max(2, M // B))
    )
    print(f"  Theoretical O((N/B)log_{{M/B}}(N/B)): ~{theoretical}")
```

??? example "Sample Output"

    ```
    External Merge Sort Simulation
      N = 10,000
      M = 1,000
      B = 100
      Fan-out (M/B - 1): 9
      Initial runs:      10
      Merge passes:       2
      Phase 1 I/Os:      200
      Total I/Os:        600
      Theoretical O((N/B)log_{M/B}(N/B)): ~200
    ```

    The simulation confirms that external merge sort requires only 2 merge passes with fan-out 9, keeping total I/O proportional to a few sequential scans.

## Lower Bound

The $\Theta((N/B) \log_{M/B}(N/B))$ bound is not just achievable -- it is a **lower bound** for comparison-based external sorting. The proof extends the decision-tree lower bound from the RAM model by accounting for the fact that each I/O operation can read $B$ elements (providing $B!$ orderings) and memory can hold $M$ elements (allowing $M!$ rearrangements).

## Reference

- Aggarwal, A. & Vitter, J. S. "The Input/Output Complexity of Sorting and Related Problems," *Communications of the ACM*, 31(9), 1988.
- Knuth, D. *The Art of Computer Programming*, Vol. 3: Sorting and Searching, 1998.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.
