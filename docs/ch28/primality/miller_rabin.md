# Miller-Rabin Primality Test

Fermat's little theorem states that if $p$ is prime, then $a^{p-1} \equiv 1
\pmod{p}$ for every $a$ not divisible by $p$.  Some composites also satisfy
this relation (Carmichael numbers), so Fermat's test alone is unreliable.
The Miller-Rabin test strengthens Fermat's test by exploiting the structure
of square roots of unity, achieving error probability at most $4^{-k}$ after
$k$ independent rounds.

## Mathematical Foundation

Write $n - 1 = 2^s \cdot d$ where $d$ is odd.  For a prime $n$ and any base
$a$ with $\gcd(a, n) = 1$, at least one of the following holds:

1. $a^d \equiv 1 \pmod{n}$, or
2. $a^{2^r d} \equiv -1 \pmod{n}$ for some $0 \le r < s$.

**Why this works.**  The sequence $a^d, a^{2d}, a^{4d}, \dots, a^{2^s d} = a^{n-1}$
consists of repeated squarings.  By Fermat's little theorem, the last term
is $1$.  In $\mathbb{Z}/p\mathbb{Z}$ (a field), the only square roots of $1$
are $\pm 1$.  So the first time $1$ appears in the sequence, the previous
term must be $-1$ (or the sequence starts at $1$).

If $n$ is composite, this property fails for at least $3/4$ of all bases
$a \in \{2, \dots, n - 2\}$.

## Algorithm

```python
"""
Miller-Rabin probabilistic primality test.

Error probability: at most 4^(-k) for k rounds.
Time per round: O(log^2 n) with fast modular exponentiation.
"""

import random


# === Miller-Rabin Test ===
def miller_rabin(n: int, k: int = 20) -> bool:
    """Return True if n is probably prime, False if definitely composite."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Write n - 1 = 2^s * d with d odd
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)

        if x in (1, n - 1):
            continue

        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False  # definitely composite

    return True  # probably prime


# === Example ===
if __name__ == "__main__":
    test_values = [2, 7, 15, 31, 49, 97, 561, 1009, 1729]
    for val in test_values:
        result = "probably prime" if miller_rabin(val) else "composite"
        print(f"miller_rabin({val}) = {result}")
```

## Error Analysis

For a single round with a random base $a$:

- If $n$ is prime, the test always returns "probably prime" (no false
  negatives).
- If $n$ is composite, the probability of a false positive (the test saying
  "probably prime") is at most $1/4$.

After $k$ independent rounds:

$$
\Pr[\text{false positive}] \le \left(\frac{1}{4}\right)^k = 4^{-k}
$$

With $k = 20$ rounds, the error probability is less than $10^{-12}$.

!!! note "Why 1/4, Not 1/2"
    The bound $1/4$ (not $1/2$) comes from Rabin's analysis showing that at
    most $(n - 1)/4$ bases are "strong liars" for any composite $n$.  This
    is tighter than the $1/2$ bound from Fermat witnesses alone.

## Deterministic Variant

For small $n$, specific sets of bases guarantee correctness:

| Range of $n$ | Bases to test |
|---|---|
| $n < 2{,}047$ | $\{2\}$ |
| $n < 1{,}373{,}653$ | $\{2, 3\}$ |
| $n < 3{,}215{,}031{,}751$ | $\{2, 3, 5, 7\}$ |
| $n < 3.3 \times 10^{24}$ | First 13 primes |

Under the Generalized Riemann Hypothesis (GRH), testing all bases
$a \le 2 (\ln n)^2$ suffices for any $n$---this is Miller's original
deterministic test.

## Complexity

- **Per round:** One modular exponentiation ($O(\log n)$ squarings, each
  costing $O(\log^2 n)$ with schoolbook multiplication) gives $O(\log^3 n)$
  bit operations.
- **Total for $k$ rounds:** $O(k \log^3 n)$.

This makes Miller-Rabin vastly faster than trial division ($O(\sqrt{n})$)
and AKS ($\widetilde{O}(\log^6 n)$) in practice.

## Carmichael Numbers

A Carmichael number $n$ satisfies $a^{n-1} \equiv 1 \pmod{n}$ for all
$a$ coprime to $n$, fooling the basic Fermat test.  The smallest is
$561 = 3 \times 11 \times 17$.  Miller-Rabin correctly identifies
Carmichael numbers as composite because it checks square roots of unity,
not just the final Fermat condition.

!!! warning "Fermat Test Is Insufficient"
    The Fermat test alone cannot distinguish Carmichael numbers from primes.
    Always use Miller-Rabin (or stronger) rather than the plain Fermat test
    in practice.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
- Rabin, M. O. "Probabilistic algorithm for testing primality." *Journal of
  Number Theory*, 12(1), 1980.
