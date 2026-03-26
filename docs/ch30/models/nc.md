# The NC Complexity Class

Some problems become dramatically faster with parallel processors: sorting
$n$ numbers takes $O(n \log n)$ sequentially but only $O(\log n)$ parallel
time with $n$ processors.  The complexity class **NC** (Nick's Class)
captures problems that are "efficiently parallelizable"---solvable in
polylogarithmic time using polynomially many processors.

## Formal Definition

The standard model for defining NC is the **PRAM** (Parallel Random Access
Machine), which consists of multiple synchronous processors sharing a common
memory.  Each processor can read from or write to any shared memory cell in
one step, and all processors execute instructions in lock-step.

A decision problem is in **NC** if it can be solved on a PRAM in

$$
O(\log^k n) \text{ time using } O(n^c) \text{ processors}
$$

for some constants $k$ and $c$, where $n$ is the input size.

### The NC Hierarchy

NC is stratified by the exponent on the logarithmic time bound:

$$
\text{NC}^1 \subseteq \text{NC}^2 \subseteq \cdots \subseteq \text{NC}^k \subseteq \cdots \subseteq \text{NC}
$$

where $\text{NC}^k$ consists of problems solvable in $O(\log^k n)$ time
with polynomially many processors.

!!! note "Circuit Characterization"
    Equivalently, $\text{NC}^k$ is the class of problems decidable by
    Boolean circuits of polynomial size and $O(\log^k n)$ depth, with
    bounded fan-in gates.  The depth corresponds to parallel time.

## Relationship to P

Every NC problem is in P, because the parallel computation can be simulated
sequentially: $O(n^c)$ processors running for $O(\log^k n)$ steps perform
at most $O(n^c \log^k n)$ total operations, which is polynomial.  Therefore:

$$
\text{NC} \subseteq \text{P}
$$

The converse---whether $\text{P} = \text{NC}$---is a major open question.
If $\text{P} \ne \text{NC}$, there exist problems solvable in polynomial
time that do not admit efficient parallelization.

## P-Complete Problems

Just as NP-complete problems represent the hardest problems in NP,
**P-complete** problems (under NC reductions) represent the hardest
problems in P to parallelize.  A problem is P-complete if every problem in
P reduces to it via a reduction computable in NC.  If any P-complete
problem is in NC, then $\text{P} = \text{NC}$.

P-complete problems are therefore considered **inherently sequential**: no
known polylogarithmic-time parallel algorithm exists for them, and their
parallelizability would collapse the entire hierarchy.

| P-Complete Problem | Description |
|---|---|
| Circuit Value Problem | Evaluate a Boolean circuit on given inputs |
| Horn-SAT | Satisfiability of Horn clauses |
| Linear Programming (feasibility) | Decide if a system of linear inequalities is feasible |
| Maximum Flow (general) | Find max flow in a network |
| Context-Free Grammar Membership | Decide if a string belongs to a CFL |

## Problems in NC

The following table lists well-known NC results.  The time and processor
counts refer to specific PRAM algorithms; circuit-depth classifications may
differ (for instance, sorting networks achieve $O(\log n)$ depth, placing
sorting in $\text{NC}^1$ in the circuit model).

| Problem | NC Level | PRAM Time | PRAM Processors |
|---|---|---|---|
| Parity | $\text{NC}^1$ | $O(\log n)$ | $O(n)$ |
| Integer addition | $\text{NC}^1$ | $O(\log n)$ | $O(n)$ |
| Integer multiplication | $\text{NC}^1$ | $O(\log n)$ | $O(n \log n \log \log n)$ |
| Sorting (Cole's merge sort) | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n)$ |
| Matrix multiplication | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n^3)$ |
| Connected components | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n^2)$ |

## Example: Parallel Prefix Sum

The prefix sum problem illustrates how a computation achieves $O(\log n)$
parallel time through a structured sweep pattern, placing it squarely in
$\text{NC}^1$.  The Blelloch algorithm performs an **up-sweep** (reduce
phase) followed by a **down-sweep** (distribute phase), each taking
$O(\log n)$ parallel steps with $O(n)$ total work.

```python
"""
Parallel prefix sum (scan) via the Blelloch algorithm (simulated).

Parallel time : O(log n)
Work (total ops): O(n)

Note: this implementation pads the input to the next power of 2
so that the binary-tree indexing works correctly for all input sizes.
"""

# === Helpers ===

def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# === Parallel Prefix Sum (simulated) ===

def parallel_prefix_sum(arr: list[int]) -> list[int]:
    """Compute inclusive prefix sums using the Blelloch algorithm.

    The input is padded to a power-of-2 length internally so that
    the up-sweep and down-sweep indexing covers every element.

    Args:
        arr: List of integers.

    Returns:
        List of inclusive prefix sums with the same length as arr.
    """
    n = len(arr)
    if n == 0:
        return []

    # Pad to power of 2
    m = next_power_of_two(n)
    x = list(arr) + [0] * (m - n)

    # Up-sweep (reduce): build partial sums bottom-up
    step = 1
    while step < m:
        for i in range(2 * step - 1, m, 2 * step):
            x[i] += x[i - step]
        step *= 2

    # Down-sweep: distribute partial sums top-down
    x[m - 1] = 0
    step = m // 2
    while step >= 1:
        for i in range(2 * step - 1, m, 2 * step):
            temp = x[i - step]
            x[i - step] = x[i]
            x[i] += temp
        step //= 2

    # Convert exclusive prefix sums to inclusive
    result = [x[i] + arr[i] for i in range(n)]
    return result


# === Demonstration ===

if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    prefix = parallel_prefix_sum(data)
    print(f"Input:      {data}")
    print(f"Prefix sum: {prefix}")

    # Verify against a simple sequential scan
    expected = []
    s = 0
    for v in data:
        s += v
        expected.append(s)
    print(f"Expected:   {expected}")
    assert prefix == expected

    # Test non-power-of-2 length
    data2 = [1, 2, 3, 4, 5]
    prefix2 = parallel_prefix_sum(data2)
    print(f"\nInput:      {data2}")
    print(f"Prefix sum: {prefix2}")
    assert prefix2 == [1, 3, 6, 10, 15]
    print("All tests passed.")
```

**Output:**

```
Input:      [3, 1, 4, 1, 5, 9, 2, 6]
Prefix sum: [3, 4, 8, 9, 14, 23, 25, 31]
Expected:   [3, 4, 8, 9, 14, 23, 25, 31]

Input:      [1, 2, 3, 4, 5]
Prefix sum: [1, 3, 6, 10, 15]
All tests passed.
```

## Complexity Landscape

The position of NC within the broader hierarchy of complexity classes clarifies its significance:

$$
\text{NC}^1 \subseteq \text{L} \subseteq \text{NL} \subseteq \text{NC}^2 \subseteq \text{P} \subseteq \text{NP}
$$

where L and NL are the logarithmic-space classes.  Whether any of these
inclusions are strict remains open, but all are believed to be.

!!! warning "NC vs. Practical Parallelism"
    NC measures theoretical parallelizability with an unlimited number of
    processors.  In practice, the number of processors is fixed, and
    communication costs, memory bandwidth, and synchronization overhead
    dominate.  A problem in NC may still be difficult to parallelize
    efficiently on real hardware.

## Reference

- Greenlaw, R., Hoover, H. J., & Ruzzo, W. L. *Limits to Parallel
  Computation: P-Completeness Theory*. Oxford University Press, 1995.
- JaJa, J. *An Introduction to Parallel Algorithms*. Addison-Wesley, 1992.
