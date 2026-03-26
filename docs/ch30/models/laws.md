# Amdahl's and Gustafson's Laws

Adding more processors does not always make a program faster.  The serial
fraction---the part that cannot be parallelized---limits the achievable
speedup.  Amdahl's Law quantifies this ceiling, while Gustafson's Law
reframes the analysis to show that parallel computing shines when we scale
the problem size with the machine.

## Amdahl's Law

Let $s$ be the fraction of a program's execution time that is inherently
serial, and $p$ be the number of processors.  The **speedup** from
parallelization is:

$$
S(p) = \frac{1}{s + \dfrac{1 - s}{p}}
$$

### Derivation

- Sequential time: $T_1 = T_s + T_p$ where $T_s$ is serial time and $T_p$
  is parallelizable time.
- With $p$ processors: $T_p = T_s + T_p / p$.
- Speedup: $S = T_1 / T_p = (T_s + T_p) / (T_s + T_p / p)$.
- Dividing numerator and denominator by $T_1$ and letting $s = T_s / T_1$
  gives the formula above.

### Key Implication

As $p \to \infty$:

$$
\lim_{p \to \infty} S(p) = \frac{1}{s}
$$

Even with infinitely many processors, speedup is bounded by $1/s$.  If
$10\%$ of the program is serial ($s = 0.1$), the maximum speedup is $10$.

!!! example "Numerical Example"
    With $s = 0.05$ (5% serial) and $p = 32$ processors:

    $$
    S(32) = \frac{1}{0.05 + \frac{0.95}{32}} = \frac{1}{0.05 + 0.0297} = \frac{1}{0.0797} \approx 12.55
    $$

    Despite using 32 processors, the speedup is only about 12.5x.

## Gustafson's Law

Amdahl's Law assumes a **fixed problem size**.  In practice, users often
increase the problem size as more processors become available.
Gustafson's Law accounts for this by fixing the parallel execution time
rather than the total work.

Let $s'$ be the fraction of the *parallel* execution time spent on serial
work.  The **scaled speedup** is:

$$
S_G(p) = p - s'(p - 1)
$$

### Derivation

- Parallel execution time: $T_p = T_s + T_{\text{par}}$.
- If the problem is scaled so that $T_p$ stays constant, the sequential
  time for the scaled problem is $T_1 = T_s + p \cdot T_{\text{par}}$.
- Speedup: $S_G = T_1 / T_p = (T_s + p \cdot T_{\text{par}}) / (T_s + T_{\text{par}})$.
- With $s' = T_s / T_p$: $S_G = s' + p(1 - s') = p - s'(p - 1)$.

### Key Implication

Gustafson's speedup grows **linearly** with $p$:

$$
S_G(p) \approx p \quad \text{when } s' \ll 1
$$

This is much more optimistic than Amdahl's bound because real workloads
often grow with available compute.

## Comparison

| Aspect | Amdahl's Law | Gustafson's Law |
|---|---|---|
| Assumption | Fixed problem size | Fixed parallel time |
| Speedup limit | $1/s$ (constant) | $\sim p$ (linear) |
| Perspective | Pessimistic for large $p$ | Optimistic for large $p$ |
| Best describes | Latency-sensitive tasks | Throughput-oriented tasks |

## Visualization

```python
"""
Visualization of Amdahl's and Gustafson's Laws.

Shows how speedup scales with the number of processors.
"""

import math


# === Amdahl's Law ===
def amdahl_speedup(s: float, p: int) -> float:
    """Compute Amdahl's speedup for serial fraction s and p processors."""
    return 1.0 / (s + (1.0 - s) / p)


# === Gustafson's Law ===
def gustafson_speedup(s_prime: float, p: int) -> float:
    """Compute Gustafson's scaled speedup."""
    return p - s_prime * (p - 1)


# === Efficiency ===
def efficiency(speedup: float, p: int) -> float:
    """Parallel efficiency = speedup / p."""
    return speedup / p


# === Example ===
if __name__ == "__main__":
    serial_fractions = [0.01, 0.05, 0.10, 0.25]
    processors = [1, 2, 4, 8, 16, 32, 64, 128]

    print("=== Amdahl's Law ===")
    print(f"{'p':>5}", end="")
    for s in serial_fractions:
        print(f"  s={s:.2f}", end="")
    print()
    for p in processors:
        print(f"{p:5d}", end="")
        for s in serial_fractions:
            sp = amdahl_speedup(s, p)
            print(f"  {sp:6.2f}", end="")
        print()

    print("\n=== Gustafson's Law ===")
    print(f"{'p':>5}", end="")
    for s in serial_fractions:
        print(f"  s={s:.2f}", end="")
    print()
    for p in processors:
        print(f"{p:5d}", end="")
        for s in serial_fractions:
            sp = gustafson_speedup(s, p)
            print(f"  {sp:6.2f}", end="")
        print()
```

## Practical Implications

- **Amdahl's Law** motivates optimizing the serial bottleneck: even a
  small reduction in $s$ can significantly raise the speedup ceiling.
- **Gustafson's Law** justifies building larger parallel systems: as long
  as the problem scales, near-linear speedup is achievable.
- **Efficiency** $E = S/p$ measures how well processors are utilized.
  Amdahl-limited workloads have $E \to 0$ as $p$ grows; Gustafson-scaled
  workloads maintain constant $E$.

## Reference

- Amdahl, G. M. "Validity of the Single Processor Approach to Achieving
  Large Scale Computing Capabilities." AFIPS 1967.
- Gustafson, J. L. "Reevaluating Amdahl's Law." *Communications of the
  ACM*, 31(5), 1988.
