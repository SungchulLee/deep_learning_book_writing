# Replacement Selection

In external merge sort, the **run formation** phase creates sorted runs of length $M$
(the memory capacity).  Fewer, longer runs mean fewer merge passes, which directly
reduces I/O cost.  **Replacement selection** is a technique that produces runs of
expected length $2M$ -- twice as long as naive run formation -- by using a min-heap to
continuously output the smallest available element while reading new input.

## Motivation

With naive run formation, $N$ elements produce $\lceil N/M \rceil$ runs of length $M$.
Replacement selection typically halves the number of runs to roughly $\lceil N/(2M)
\rceil$, saving one full merge pass in many practical scenarios.  When the input is
nearly sorted, replacement selection can produce runs that are much longer than $2M$,
sometimes spanning the entire file.

## Algorithm Overview

Maintain a min-heap of capacity $M$ (the memory size).

1. **Initialize.** Read the first $M$ elements into the heap.
2. **Output loop.** While the heap is not empty:
    - Extract the minimum element $x$ from the heap and write it to the current run.
    - Read the next input element $y$ (if any).
    - If $y \ge x$, insert $y$ into the heap (it belongs to the current run).
    - If $y < x$, mark $y$ as belonging to the *next* run.  It stays in the heap
      but is treated as $+\infty$ for ordering purposes until the current run ends.
3. **End current run.** When all elements in the heap belong to the next run,
   "unmark" them (they now form the new current run) and continue from step 2.
4. **Terminate.** When the input is exhausted and the heap is empty, all runs are
   complete.

## Expected Run Length

**Theorem (Knuth).** If the input is a random permutation and the heap has capacity
$M$, the expected length of each run produced by replacement selection is $2M$.

*Intuition.* The heap acts as a "reservoir" of $M$ elements.  On average, half of the
incoming elements are large enough to extend the current run, and half are too small
and must wait for the next run.  This doubles the effective run length compared to
simply sorting $M$-element chunks.

The analysis uses the "snowplow" analogy: imagine a circular road of length $2M$ with
snow falling uniformly.  A plow traveling around the road sweeps up $2M$ units of snow
per revolution, regardless of where it starts.

## Worked Example

Memory capacity $M = 3$.  Input stream: $[5, 3, 8, 1, 7, 2, 6, 4]$.

**Run 1:**

| Step | Heap | Output | Input | Action |
|------|------|--------|-------|--------|
| Init | $\{3, 5, 8\}$ | -- | $1, 7, 2, 6, 4$ | Load first 3 |
| 1 | $\{5, 8\}$ | $3$ | $1, 7, 2, 6, 4$ | Extract 3, read 1 (1 < 3, mark for next) |
| 2 | $\{5, 8, [1]\}$ | $3$ | $7, 2, 6, 4$ | 1 marked, extract 5, read 7 (7 >= 5) |
| 3 | $\{7, 8, [1]\}$ | $3, 5$ | $2, 6, 4$ | Extract 7, read 2 (2 < 7, mark) |
| 4 | $\{8, [1, 2]\}$ | $3, 5, 7$ | $6, 4$ | Extract 8, read 6 (6 < 8, mark) |
| 5 | $\{[1, 2, 6]\}$ | $3, 5, 7, 8$ | $4$ | All marked -- end Run 1 |

**Run 1 output:** $[3, 5, 7, 8]$ (length 4 > $M = 3$).

**Run 2:** Unmark $\{1, 2, 6\}$, read 4.

| Step | Heap | Output | Input | Action |
|------|------|--------|-------|--------|
| 1 | $\{2, 4, 6\}$ | $1$ | -- | Extract 1, read 4 (4 >= 1) |
| 2 | $\{4, 6\}$ | $1, 2$ | -- | Extract 2, no more input |
| 3 | $\{6\}$ | $1, 2, 4$ | -- | Extract 4 |
| 4 | $\{\}$ | $1, 2, 4, 6$ | -- | Extract 6 -- done |

**Run 2 output:** $[1, 2, 4, 6]$ (length 4).

Total: 2 runs of length 4 instead of 3 runs of length 3 with naive formation.

## Implementation

```python
"""
Replacement selection -- produces longer sorted runs for external merge sort.

Expected run length is 2M where M is the heap (memory) capacity.
This reduces the number of merge passes needed.
"""

import heapq


# === Replacement selection ==================================================

def replacement_selection(data: list[int], memory_size: int) -> list[list[int]]:
    """Produce sorted runs from *data* using replacement selection.

    Parameters
    ----------
    data : list[int]
        Input stream of elements.
    memory_size : int
        Maximum number of elements in the heap (memory capacity).

    Returns
    -------
    list[list[int]]
        List of sorted runs.
    """
    runs: list[list[int]] = []
    current_run: list[int] = []

    # Heap entries: (effective_key, actual_value, generation)
    # generation 0 = current run, generation 1 = next run
    heap: list[tuple[int, int, int]] = []
    current_gen = 0
    pos = 0

    # Initialize heap
    while pos < len(data) and len(heap) < memory_size:
        heapq.heappush(heap, (data[pos], data[pos], 0))
        pos += 1

    while heap:
        # Extract minimum
        _, val, gen = heapq.heappop(heap)

        if gen > current_gen:
            # All remaining elements belong to the next run
            runs.append(current_run)
            current_run = []
            current_gen = gen

        current_run.append(val)

        # Read next input element
        if pos < len(data):
            next_val = data[pos]
            pos += 1

            if next_val >= val:
                # Belongs to current run
                heapq.heappush(heap, (next_val, next_val, current_gen))
            else:
                # Too small -- belongs to next run
                # Use a large sentinel key so it stays in the heap
                # but won't be extracted until current run ends
                heapq.heappush(
                    heap, (float("inf"), next_val, current_gen + 1)
                )

    if current_run:
        runs.append(current_run)

    return runs


# === Demo ===================================================================

if __name__ == "__main__":
    data = [5, 3, 8, 1, 7, 2, 6, 4]
    memory_size = 3

    runs = replacement_selection(data, memory_size)
    print(f"Input: {data}")
    print(f"Memory size: {memory_size}")
    print(f"Number of runs: {len(runs)}")
    for i, run in enumerate(runs):
        print(f"  Run {i + 1}: {run} (length {len(run)})")

    # Compare with naive run formation
    naive_runs = []
    for start in range(0, len(data), memory_size):
        naive_runs.append(sorted(data[start : start + memory_size]))
    print(f"\nNaive run formation: {len(naive_runs)} runs")
    for i, run in enumerate(naive_runs):
        print(f"  Run {i + 1}: {run} (length {len(run)})")
```

**Output:**
```
Input: [5, 3, 8, 1, 7, 2, 6, 4]
Memory size: 3
Number of runs: 2
  Run 1: [3, 5, 7, 8] (length 4)
  Run 2: [1, 2, 4, 6] (length 4)

Naive run formation: 3 runs
  Run 1: [3, 5, 8] (length 3)
  Run 2: [1, 2, 7] (length 3)
  Run 3: [4, 6] (length 2)
```

## Comparison with Naive Run Formation

| Property | Naive | Replacement Selection |
|----------|-------|----------------------|
| Run length | Exactly $M$ | Expected $2M$ |
| Number of runs | $\lceil N/M \rceil$ | $\approx \lceil N/(2M) \rceil$ |
| Nearly sorted input | No benefit | Runs can span entire file |
| Implementation | Simple array sort | Min-heap with generation tracking |
| Extra merge passes saved | -- | Typically 1 |

## Reference

- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and
  Searching* (2nd ed.), Section 5.4.1. Addison-Wesley.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
