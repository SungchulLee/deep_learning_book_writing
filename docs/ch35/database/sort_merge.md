# External Sort-Merge

When a dataset is too large to fit in memory, standard in-memory sorting algorithms cannot be applied directly. **External sort-merge** solves this by breaking the data into memory-sized chunks, sorting each chunk internally, and then merging the sorted chunks using disk I/O. It is the workhorse behind database `ORDER BY`, sort-merge joins, and any operation that requires sorted access to large files.

## Intuition

Imagine sorting a deck of 10,000 cards when your table can only hold 100 cards at a time. You would:

1. Pick up 100 cards, sort them, and place the sorted pile on the floor.
2. Repeat until all 100 piles are sorted.
3. Merge the piles: look at the top card of each pile, pick the smallest, and place it in the output.

External sort-merge follows exactly this strategy, replacing "cards" with disk pages and "table" with memory buffer.

## Algorithm

### Phase 1 -- Run Generation

Read $M$ pages of data into memory at a time, sort them using an efficient in-memory algorithm (e.g., quicksort), and write the sorted **run** back to disk.

Given a file of $b$ pages and $M$ pages of memory, this produces:

$$
\left\lceil \frac{b}{M} \right\rceil
$$

initial sorted runs, each of length $M$ pages (except possibly the last).

### Phase 2 -- Multi-way Merge

Merge the sorted runs using $(M - 1)$-way merge, reserving one page for output:

1. Open all runs (up to $M - 1$ at a time), reading the first page of each into memory.
2. Repeatedly select the smallest record across all input pages and write it to the output page.
3. When an input page is exhausted, read the next page from that run.
4. When the output page is full, flush it to disk.

If the number of runs exceeds $M - 1$, multiple merge passes are required.

### Number of Passes

The total number of passes (including run generation) is:

$$
1 + \left\lceil \log_{M-1}\!\left\lceil \frac{b}{M} \right\rceil \right\rceil
$$

- Pass 0 (run generation) creates $\lceil b/M \rceil$ runs.
- Each subsequent merge pass reduces the number of runs by a factor of $M - 1$.
- Merging stops when a single sorted run remains.

### I/O Cost

Each pass reads and writes every page once, so the total I/O cost is:

$$
2 \, b \cdot \left(1 + \left\lceil \log_{M-1}\!\left\lceil \frac{b}{M} \right\rceil \right\rceil\right)
$$

The factor of 2 accounts for one read and one write per page per pass. The final write of the last pass can sometimes be piped directly to the consuming operator (e.g., a merge join), saving $b$ I/Os.

??? example "Concrete example"
    Suppose $b = 10{,}000$ pages and $M = 100$ pages:

    - **Pass 0**: Creates $\lceil 10{,}000 / 100 \rceil = 100$ sorted runs. Cost: $2 \times 10{,}000 = 20{,}000$ I/Os.
    - **Pass 1**: Merge 100 runs using 99-way merge into $\lceil 100/99 \rceil = 2$ runs. Cost: $20{,}000$ I/Os.
    - **Pass 2**: Merge 2 runs into 1. Cost: $20{,}000$ I/Os.
    - **Total**: $3 \times 20{,}000 = 60{,}000$ I/Os.

    Using the formula: $2 \times 10{,}000 \times (1 + \lceil \log_{99}(100) \rceil) = 2 \times 10{,}000 \times 3 = 60{,}000$ I/Os.

## Replacement Sort (Run Generation Optimization)

A **priority queue** (min-heap) can generate longer initial runs:

1. Fill memory with $M$ pages and build a min-heap.
2. Extract the minimum record and write it to the current run.
3. Read the next record from input. If it is greater than or equal to the last output, insert it into the heap; otherwise, mark it for the next run.
4. When the heap is exhausted of records for the current run, start a new run.

On average, this produces runs of length $2M$ (assuming random input), cutting the number of initial runs in half and potentially eliminating one merge pass.

!!! tip "When replacement sort helps"
    If the input is nearly sorted, replacement sort can produce runs much longer than $2M$, sometimes fitting the entire file in a single run and eliminating the merge phase entirely.

## Blocked I/O Optimization

Instead of reading one page at a time from each run during merging, read $B$ consecutive pages at a time from each run. This reduces random I/O by issuing larger sequential reads:

- Available merge fans: $\lfloor (M - B) / B \rfloor = \lfloor M/B \rfloor - 1$
- Larger $B$ means fewer fans per pass but faster per-fan I/O

The optimal block size balances merge fan-out against sequential read speed.

## Implementation

```python
"""
External Sort-Merge -- simulation of disk-based sorting.

Demonstrates the two-phase external sort-merge algorithm using
in-memory lists to simulate disk pages.
"""

import heapq


# === External Sort-Merge ======================================================

def external_sort_merge(data: list[int], memory_pages: int,
                        page_size: int) -> list[int]:
    """Simulate external sort-merge on a list of integers.

    Args:
        data: The unsorted input data.
        memory_pages: Number of pages that fit in memory (M).
        page_size: Number of records per page (B).

    Returns:
        Sorted list of integers.
    """
    memory_size = memory_pages * page_size

    # --- Phase 1: Run generation ---
    runs: list[list[int]] = []
    for start in range(0, len(data), memory_size):
        chunk = data[start:start + memory_size]
        chunk.sort()
        runs.append(chunk)

    # --- Phase 2: Multi-way merge ---
    fan_in = memory_pages - 1  # reserve 1 page for output
    while len(runs) > 1:
        new_runs: list[list[int]] = []
        for i in range(0, len(runs), fan_in):
            group = runs[i:i + fan_in]
            merged = list(heapq.merge(*group))
            new_runs.append(merged)
        runs = new_runs

    return runs[0] if runs else []


# === Statistics ===============================================================

def compute_passes(num_pages: int, memory_pages: int) -> int:
    """Compute the total number of passes for external sort-merge."""
    import math
    if num_pages <= memory_pages:
        return 1
    initial_runs = math.ceil(num_pages / memory_pages)
    merge_passes = math.ceil(math.log(initial_runs) / math.log(memory_pages - 1))
    return 1 + merge_passes


def compute_io_cost(num_pages: int, memory_pages: int) -> int:
    """Compute total I/O cost (reads + writes) for external sort-merge."""
    passes = compute_passes(num_pages, memory_pages)
    return 2 * num_pages * passes


# === Main =====================================================================

if __name__ == "__main__":
    import random

    # Simulation parameters
    random.seed(42)
    n_records = 10000
    page_size = 100
    memory_pages = 10
    data = random.sample(range(n_records * 10), n_records)

    print(f"Records: {n_records}")
    print(f"Page size: {page_size} records")
    print(f"Memory: {memory_pages} pages ({memory_pages * page_size} records)")

    num_pages = (n_records + page_size - 1) // page_size
    passes = compute_passes(num_pages, memory_pages)
    io_cost = compute_io_cost(num_pages, memory_pages)
    print(f"Pages: {num_pages}")
    print(f"Passes: {passes}")
    print(f"I/O cost: {io_cost} page transfers")

    sorted_data = external_sort_merge(data, memory_pages, page_size)
    assert sorted_data == sorted(data), "Sort verification failed"
    print(f"Sort verified: first 10 = {sorted_data[:10]}")
```

## Reference

- [Database System Concepts (Silberschatz, Korth, Sudarshan)](https://www.db-book.com/), Chapter 13
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
- Knuth, D. E. *The Art of Computer Programming*, Volume 3: Sorting and Searching
