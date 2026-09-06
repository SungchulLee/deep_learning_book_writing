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

## Exercises

**Exercise 1.**
A file has 1000 pages and the buffer pool has 5 pages. How many sorted runs are created in the first pass, and how many merge passes are needed?

??? success "Solution to Exercise 1"
    First pass: read 5 pages at a time, sort in memory, write back. Number of sorted runs: $\lceil 1000 / 5 \rceil = 200$. Each merge pass uses $B - 1 = 4$ buffer pages for input (one page per run) and 1 for output, merging 4 runs at a time. Number of merge passes: $\lceil \log_4(200) \rceil = \lceil 3.82 \rceil = 4$. Total passes: $1 + 4 = 5$. Total I/O: $2 \times 1000 \times 5 = 10{,}000$ page reads and writes. $\square$

---

**Exercise 2.**
Explain the replacement-sort optimization for generating longer initial runs. When does it help?

??? success "Solution to Exercise 2"
    Replacement sort uses a priority queue (min-heap) of size $M$ (buffer pages). As data is read, elements are inserted into the heap. The minimum is output to the current run. When a new element is read, if it is $\ge$ the last output, it joins the current run. If it is smaller, it is marked for the next run and stays in the heap. On average, this produces initial runs of size $2M$ (twice the buffer size) for random input. For nearly-sorted input, runs can be much longer (up to the entire file). This reduces the number of initial runs, saving merge passes. It helps most when data has partial order or when $M$ is small relative to the file size, since longer runs reduce $\lceil \log_{B-1}(\text{runs}) \rceil$. $\square$

---

**Exercise 3.**
Prove that external sort-merge with $B$ buffer pages requires $O(N \log_{B-1}(N/B))$ I/O operations for a file of $N$ pages.

??? success "Solution to Exercise 3"
    The sort phase produces $\lceil N/B \rceil$ sorted runs, each requiring $N$ page reads and $N$ page writes. Each merge phase reads all $N$ pages and writes $N$ pages, reducing the number of runs by a factor of $B - 1$. The number of merge passes is $\lceil \log_{B-1}(N/B) \rceil$. Total I/O: $2N \times (1 + \lceil \log_{B-1}(N/B) \rceil) = O(N \log_{B-1}(N/B))$. The $\log_{B-1}$ factor reflects the merge tree's fan-out: a wider merge (more buffer pages) reduces the number of passes. With $B = 101$ (100-way merge) and $N = 10^6$ pages, passes $= \lceil \log_{100}(10^4) \rceil = 2$, so total I/O $\approx 6 \times 10^6$. $\square$

---

**Exercise 4.**
Describe how double buffering improves the I/O efficiency of external sort-merge. What hardware feature does it exploit?

??? success "Solution to Exercise 4"
    Without double buffering, the CPU is idle while waiting for a page to be read from disk, and the disk is idle while the CPU processes a page. Double buffering allocates two buffers per input stream: while the CPU processes data from buffer A, the disk prefetches the next page into buffer B. When the CPU finishes A, it switches to B (instantly), and the disk starts filling A. This overlaps I/O and computation, exploiting the fact that disk and CPU are independent hardware units that can operate concurrently. The cost is doubling the buffer space per stream: with $B$ buffer pages and double buffering, only $(B-1)/2$ merge streams can run simultaneously. The tradeoff is worth it when I/O latency dominates, which is typical for HDD-based systems. On SSDs with low latency, the benefit is smaller. $\square$

---

**Exercise 5.**
Compare external sort-merge with external hash-based grouping for a `GROUP BY` query. When is each approach preferred?

??? success "Solution to Exercise 5"
    **External sort-merge**: sorts the data by the grouping key, then scans the sorted result to aggregate consecutive groups. Cost: $O(N \log_{B-1}(N/B))$ I/O. Produces sorted output (useful if `ORDER BY` is also needed). **External hash-based grouping**: partitions the data by hashing the grouping key, then aggregates each partition independently. Cost: $O(N)$ if partitions fit in memory; $O(3N)$ with one partitioning pass. Does not produce sorted output. Sorting is preferred when: (1) the query also requires sorted output; (2) the grouping key has few distinct values (sort + sequential scan is simple and cache-friendly). Hashing is preferred when: (1) no sorted output is needed; (2) the data is large and reducing I/O passes matters; (3) the number of groups is small enough that each partition's hash table fits in memory. Most modern databases use hashing for `GROUP BY` and sorting only when `ORDER BY` is also specified. $\square$
