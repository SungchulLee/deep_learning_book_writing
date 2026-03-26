# PRAM Model

Analyzing parallel algorithms requires a formal model of computation, just
as the RAM model underlies sequential algorithm analysis.  The **Parallel
Random Access Machine (PRAM)** extends the RAM model to multiple
processors sharing a common memory, providing a clean abstraction for
reasoning about parallel time and work complexity.

## Definition

A PRAM consists of:

- $p$ processors $P_1, P_2, \dots, P_p$, each with private local memory.
- A shared global memory of unbounded size.
- A common clock that synchronizes all processors.

In each time step, every processor can:

1. Read one cell from shared memory.
2. Perform one local computation.
3. Write one cell to shared memory.

All processors execute the same program but branch on their processor ID.

## PRAM Variants

When multiple processors access the same memory cell simultaneously,
the behavior depends on the PRAM variant:

| Variant | Concurrent Read | Concurrent Write | Power |
|---|---|---|---|
| **EREW** (Exclusive Read, Exclusive Write) | No | No | Weakest |
| **CREW** (Concurrent Read, Exclusive Write) | Yes | No | Medium |
| **CRCW** (Concurrent Read, Concurrent Write) | Yes | Yes | Strongest |

### CRCW Write Conflict Resolution

When multiple processors write to the same cell in CRCW:

- **Common:** All writing processors must write the same value.
- **Arbitrary:** One arbitrary writer succeeds.
- **Priority:** The processor with the smallest ID wins.

!!! note "Relative Power"
    Any algorithm running in $T$ time on a CRCW PRAM can be simulated on
    an EREW PRAM in $O(T \log p)$ time.  So the variants differ by at most
    a logarithmic factor.

## Complexity Measures

For a PRAM algorithm using $p$ processors:

- **Parallel time** $T_p$: the number of synchronous steps.
- **Work** $W = p \cdot T_p$: the total number of operations across all
  processors and all steps.
- **Cost** $C = p \cdot T_p$: same as work (used interchangeably).

An algorithm is **work-efficient** if $W = O(T_1)$, where $T_1$ is the
best sequential time for the problem.

### Brent's Theorem

**Theorem (Brent).**  Any PRAM algorithm with parallel time $T$ and work $W$
can be executed on $p$ processors in time

$$
T_p = O\!\left(\frac{W}{p} + T\right)
$$

This bound shows that a work-efficient algorithm ($W = T_1$) achieves
near-optimal speedup when $p \le T_1 / T$.

## Example: Parallel Sum

Computing the sum of $n$ numbers illustrates PRAM algorithm design.

### Binary Tree Reduction

Pair up elements and sum in parallel, halving the problem size each step:

$$
T = O(\log n), \quad p = n/2, \quad W = O(n)
$$

This is work-efficient since sequential sum takes $O(n)$.

```python
"""
PRAM parallel sum simulation using binary tree reduction.

Parallel time: O(log n)
Work: O(n) — work-efficient
"""

import math


# === Parallel Sum (simulated) ===
def parallel_sum(arr: list[int]) -> int:
    """Simulate binary tree reduction on a PRAM."""
    n = len(arr)
    if n == 0:
        return 0

    # Pad to power of 2
    size = 1 << math.ceil(math.log2(max(n, 1)))
    data = list(arr) + [0] * (size - n)

    steps = 0
    stride = 1
    while stride < size:
        # Each "processor" computes one addition
        new_data = list(data)
        for i in range(0, size, 2 * stride):
            new_data[i] = data[i] + data[i + stride]
        data = new_data
        stride *= 2
        steps += 1

    return data[0]


# === Parallel Maximum ===
def parallel_max(arr: list[int]) -> int:
    """Simulate PRAM parallel maximum via binary tree reduction."""
    n = len(arr)
    if n == 0:
        return float("-inf")

    size = 1 << math.ceil(math.log2(max(n, 1)))
    data = list(arr) + [float("-inf")] * (size - n)

    stride = 1
    while stride < size:
        new_data = list(data)
        for i in range(0, size, 2 * stride):
            new_data[i] = max(data[i], data[i + stride])
        data = new_data
        stride *= 2

    return data[0]


# === Example ===
if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Input: {data}")
    print(f"Parallel sum: {parallel_sum(data)}")
    print(f"Parallel max: {parallel_max(data)}")
    print(f"Steps: {math.ceil(math.log2(len(data)))}")
```

## PRAM vs. Other Models

| Model | Shared memory | Synchronous | Communication cost |
|---|---|---|---|
| PRAM | Yes | Yes | Free (unit cost) |
| BSP | Yes (via supersteps) | Barrier-synchronized | Explicit |
| LogP | No (message passing) | No | Latency + bandwidth |
| Work-Span | Implicit DAG | Yes (fork/join) | Free |

!!! warning "PRAM Limitations"
    The PRAM assumes unit-cost shared memory access regardless of the
    number of processors.  Real machines face memory contention, cache
    coherence overhead, and non-uniform access latencies (NUMA).  PRAM
    analysis gives a useful lower bound on parallel time but may
    overestimate practical speedup.

## Key Problems and Their PRAM Complexity

| Problem | Time | Processors | Work-efficient? |
|---|---|---|---|
| Sum / Reduction | $O(\log n)$ | $O(n)$ | Yes |
| Prefix sum | $O(\log n)$ | $O(n)$ | Yes |
| Sorting (Cole's) | $O(\log n)$ | $O(n)$ | Yes |
| List ranking | $O(\log n)$ | $O(n)$ | Yes |
| Matrix multiply | $O(\log n)$ | $O(n^3)$ | Yes |
| Connected components | $O(\log^2 n)$ | $O(n^2)$ | No |

## Reference

- JaJa, J. *An Introduction to Parallel Algorithms*. Addison-Wesley, 1992.
- Karp, R. M. & Ramachandran, V. "Parallel Algorithms for Shared-Memory
  Machines." *Handbook of Theoretical Computer Science*, Vol. A, 1990.
