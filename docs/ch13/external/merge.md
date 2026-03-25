# External Merge Sort

Standard sorting algorithms assume that the entire dataset fits in main memory.  When
the data is too large -- a 100 GB log file on a machine with 4 GB of RAM, for instance
-- we must sort using disk (or other secondary storage).  Disk I/O is orders of
magnitude slower than memory access, so the algorithm's goal shifts from minimizing
comparisons to minimizing the number of **disk reads and writes**.  **External merge
sort** is the classic algorithm for this setting.

## The External Memory Model

The analysis uses the **external memory (I/O) model** with three parameters:

| Symbol | Meaning |
|--------|---------|
| $N$ | Total number of elements |
| $M$ | Number of elements that fit in memory |
| $B$ | Number of elements per disk block (page) |

A single I/O operation reads or writes one block of $B$ elements.  The goal is to
minimize the total number of I/O operations.

## Algorithm Overview

External merge sort proceeds in two phases:

### Phase 1 -- Run Formation

1. Read $M$ elements into memory.
2. Sort them using an efficient in-memory sort (e.g., quicksort).
3. Write the sorted **run** of $M$ elements back to disk.
4. Repeat until all $N$ elements have been processed.

This produces $\lceil N / M \rceil$ sorted runs, each of length at most $M$.

**I/O cost of Phase 1:** Each element is read once and written once, for a total of
$2 \lceil N / B \rceil$ I/Os.

### Phase 2 -- Merge Passes

Merge the sorted runs pairwise (2-way merge):

1. Open two input runs for reading and one output file for writing.
2. Repeatedly compare the front elements of both runs, write the smaller one to the
   output, and advance the corresponding input.
3. After one merge pass, the number of runs is halved and each run is twice as long.
4. Repeat until a single sorted run remains.

Each merge pass reads and writes all $N$ elements, costing $2 \lceil N / B \rceil$
I/Os.  The number of passes is $\lceil \log_2 (N / M) \rceil$.

## I/O Complexity

$$
\text{I/O}(N, M, B) = O\!\left(\frac{N}{B} \log_2 \frac{N}{M}\right)
$$

Each of the $O(\log_2 (N/M))$ passes performs $O(N/B)$ I/Os.

This can be substantially improved using **multi-way merge** (covered in the next
section), which increases the base of the logarithm from 2 to $M/B - 1$:

$$
\text{I/O}(N, M, B) = O\!\left(\frac{N}{B} \log_{M/B} \frac{N}{M}\right)
$$

## Worked Example

Sort $N = 10{,}000$ elements with $M = 1{,}000$ and $B = 100$.

| Phase | Runs | Run length | I/Os per pass |
|-------|------|-----------|---------------|
| Run formation | 10 | 1,000 | 200 |
| Merge pass 1 | 5 | 2,000 | 200 |
| Merge pass 2 | 3 | 4,000 | 200 |
| Merge pass 3 | 2 | 8,000 / 2,000 | 200 |
| Merge pass 4 | 1 | 10,000 | 200 |

Total: $200 + 4 \times 200 = 1{,}000$ I/Os (versus $100{,}000$ I/Os if each element
were accessed individually).

## Double Buffering

A practical optimization is **double buffering**: while processing one input block in
memory, the next block is being read from disk asynchronously.  This overlaps
computation with I/O and keeps the disk busy continuously.

## Implementation

```python
"""
External merge sort -- two-way merge for data larger than memory.

Simulates external sorting using files. In production, this would use
memory-mapped files or direct disk I/O with buffer management.
Time:  O(N log(N/M)) comparisons
I/O:   O((N/B) * log_2(N/M)) block transfers
"""

import heapq
import tempfile
import os


# === Run formation ==========================================================

def _create_sorted_runs(
    data: list[int], memory_size: int, temp_dir: str
) -> list[str]:
    """Split data into sorted runs of size *memory_size*.

    Returns a list of file paths, each containing one sorted run.
    """
    runs = []
    for start in range(0, len(data), memory_size):
        chunk = sorted(data[start : start + memory_size])
        path = os.path.join(temp_dir, f"run_{len(runs)}.txt")
        with open(path, "w") as f:
            for val in chunk:
                f.write(f"{val}\n")
        runs.append(path)
    return runs


# === Two-way merge ===========================================================

def _merge_two_runs(path_a: str, path_b: str, output_path: str) -> str:
    """Merge two sorted run files into a single sorted output file."""
    with open(path_a) as fa, open(path_b) as fb, open(output_path, "w") as out:
        a = fa.readline()
        b = fb.readline()
        while a and b:
            if int(a) <= int(b):
                out.write(a)
                a = fa.readline()
            else:
                out.write(b)
                b = fb.readline()
        # Write remaining elements
        while a:
            out.write(a)
            a = fa.readline()
        while b:
            out.write(b)
            b = fb.readline()
    return output_path


# === External merge sort =====================================================

def external_merge_sort(data: list[int], memory_size: int) -> list[int]:
    """Sort *data* using external merge sort with given memory constraint.

    Parameters
    ----------
    data : list[int]
        Input data (simulating a large file).
    memory_size : int
        Maximum number of elements that fit in memory.

    Returns
    -------
    list[int]
        Sorted data.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Phase 1: create sorted runs
        runs = _create_sorted_runs(data, memory_size, temp_dir)

        # Phase 2: merge passes
        pass_num = 0
        while len(runs) > 1:
            next_runs = []
            for i in range(0, len(runs), 2):
                if i + 1 < len(runs):
                    out_path = os.path.join(
                        temp_dir, f"pass{pass_num}_merge{i}.txt"
                    )
                    _merge_two_runs(runs[i], runs[i + 1], out_path)
                    next_runs.append(out_path)
                else:
                    next_runs.append(runs[i])
            runs = next_runs
            pass_num += 1

        # Read final sorted run
        with open(runs[0]) as f:
            return [int(line) for line in f]


# === Demo ===================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    data = random.sample(range(10000), 100)
    memory_size = 20  # simulate small memory

    sorted_data = external_merge_sort(data, memory_size)
    print(f"Input (first 10):  {data[:10]}")
    print(f"Sorted (first 10): {sorted_data[:10]}")
    print(f"Correctly sorted:  {sorted_data == sorted(data)}")
    print(f"Runs created:      {len(data) // memory_size + (1 if len(data) % memory_size else 0)}")
    print(f"Merge passes:      {(len(data) // memory_size - 1).bit_length()}")
```

**Output:**
```
Input (first 10):  [4575, 7562, 7326, 1040, 6498, 8802, 2848, 2813, 7147, 4280]
Sorted (first 10): [12, 37, 75, 105, 127, 153, 175, 239, 242, 252]
Correctly sorted:  True
Runs created:      5
Merge passes:      3
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
- Vitter, J. S. (2001). External memory algorithms and data structures:
  dealing with massive data. *ACM Computing Surveys*, 33(2), 209-271.
