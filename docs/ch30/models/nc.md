# The NC Complexity Class

Some problems become dramatically faster with parallel processors: sorting
$n$ numbers takes $O(n \log n)$ sequentially but only $O(\log n)$ parallel
time with $n$ processors.  The complexity class **NC** (Nick's Class)
captures problems that are "efficiently parallelizable"---solvable in
polylogarithmic time using polynomially many processors.

## Formal Definition

A decision problem is in **NC** if it can be solved on a PRAM (Parallel
Random Access Machine) in

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

Every NC problem is in P (simulate the parallel computation sequentially):

$$
\text{NC} \subseteq \text{P}
$$

The converse---whether $\text{P} = \text{NC}$---is a major open question.
If $\text{P} \ne \text{NC}$, there exist problems solvable in polynomial
time that do not admit efficient parallelization.

## P-Complete Problems

A problem is **P-complete** (under NC reductions) if every problem in P
reduces to it in NC.  If any P-complete problem is in NC, then
$\text{P} = \text{NC}$.

P-complete problems are considered **inherently sequential**---they are
the hardest problems in P to parallelize.

| P-Complete Problem | Description |
|---|---|
| Circuit Value Problem | Evaluate a Boolean circuit on given inputs |
| Linear Programming | Solve a system of linear inequalities |
| Maximum Flow (general) | Find max flow in a network |
| Context-Free Grammar Membership | Decide if a string belongs to a CFL |

## Problems in NC

| Problem | NC Level | Time | Processors |
|---|---|---|---|
| Parity | $\text{NC}^1$ | $O(\log n)$ | $O(n)$ |
| Sorting | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n)$ |
| Matrix multiplication | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n^3)$ |
| Connected components | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n^2)$ |
| Integer addition | $\text{NC}^1$ | $O(\log n)$ | $O(n)$ |
| Integer multiplication | $\text{NC}^1$ | $O(\log n)$ | $O(n^2)$ |

## Example: Parallel Prefix Sum

Computing the prefix sums of an array is a canonical NC problem.

```python
"""
Parallel prefix sum (scan) algorithm.

Parallel time : O(log n)
Work (total ops): O(n)
"""


# === Parallel Prefix Sum (simulated) ===
def parallel_prefix_sum(arr: list[int]) -> list[int]:
    """Compute prefix sums using the Blelloch algorithm (simulated)."""
    n = len(arr)
    if n == 0:
        return []

    # Work array
    x = list(arr)

    # Up-sweep (reduce)
    step = 1
    while step < n:
        for i in range(step - 1, n, 2 * step):
            right = i + step
            if right < n:
                x[right] += x[i]
        step *= 2

    # Down-sweep
    x[n - 1] = 0
    step = n // 2
    while step >= 1:
        for i in range(step - 1, n, 2 * step):
            right = i + step
            if right < n:
                temp = x[i]
                x[i] = x[right]
                x[right] += temp
        step //= 2

    # Exclusive to inclusive
    result = [x[i] + arr[i] for i in range(n)]
    return result


# === Example ===
if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    prefix = parallel_prefix_sum(data)
    print(f"Input:      {data}")
    print(f"Prefix sum: {prefix}")

    # Verify
    expected = []
    s = 0
    for v in data:
        s += v
        expected.append(s)
    print(f"Expected:   {expected}")
    assert prefix == expected
```

## The Bigger Picture

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
