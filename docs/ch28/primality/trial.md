# Trial Division Primality Test

The most natural way to check whether a number $n$ is prime is to try
dividing it by every integer from $2$ up to $\sqrt{n}$.  If no divisor is
found, $n$ must be prime.  Trial division is the simplest primality test,
ideal for small numbers and as a first filter before applying more
sophisticated methods.

## Mathematical Basis

**Theorem.**  If $n > 1$ is composite, then $n$ has a prime factor
$p \le \sqrt{n}$.

**Proof.**  Write $n = ab$ with $1 < a \le b < n$.  Then $a^2 \le ab = n$,
so $a \le \sqrt{n}$.  The smallest prime factor of $a$ (which divides $n$)
is at most $a \le \sqrt{n}$.  $\square$

This theorem means we need to test at most $\pi(\sqrt{n}) \approx
2\sqrt{n}/\ln n$ prime divisors, or at most $\sqrt{n}/2$ odd divisors if we
skip even numbers.

## Algorithm

```python
"""
Trial division primality test.

Time : O(sqrt(n))
Space: O(1)
"""

import math


# === Trial Division Primality ===
def is_prime(n: int) -> bool:
    """Return True if n is prime using trial division."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    # Check divisors of the form 6k +/- 1
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6

    return True


# === Example ===
if __name__ == "__main__":
    test_values = [1, 2, 3, 4, 17, 49, 97, 100, 997, 1000003]
    for val in test_values:
        result = "prime" if is_prime(val) else "composite"
        print(f"is_prime({val}) = {result}")
```

**Expected output:**

```
is_prime(1) = composite
is_prime(2) = prime
is_prime(3) = prime
is_prime(4) = composite
is_prime(17) = prime
is_prime(49) = composite
is_prime(97) = prime
is_prime(100) = composite
is_prime(997) = prime
is_prime(1000003) = prime
```

## Why 6k +/- 1?

Every integer falls into one of six classes modulo $6$:

$$
n \equiv 0, 1, 2, 3, 4, 5 \pmod{6}
$$

Numbers congruent to $0, 2, 4$ are divisible by $2$; numbers congruent to
$0, 3$ are divisible by $3$.  After checking divisibility by $2$ and $3$,
the only remaining primes satisfy $n \equiv 1$ or $5 \pmod{6}$, i.e.,
$n = 6k \pm 1$.  Testing only these candidates reduces the divisor count
by a factor of $3$ compared to testing all integers.

## Complexity

| Variant | Divisors tested | Time |
|---|---|---|
| Naive (all $d$ from 2 to $\sqrt{n}$) | $\sqrt{n}$ | $O(\sqrt{n})$ |
| Odd only | $\sqrt{n}/2$ | $O(\sqrt{n})$ |
| 6k +/- 1 | $\sqrt{n}/3$ | $O(\sqrt{n})$ |
| Precomputed primes | $\pi(\sqrt{n}) \approx 2\sqrt{n}/\ln n$ | $O(\sqrt{n}/\ln n)$ |

All variants have the same asymptotic complexity $O(\sqrt{n})$, but constant
factors differ significantly in practice.

!!! tip "When to Switch Methods"
    Trial division is practical for $n$ up to about $10^{14}$.  For larger
    numbers, probabilistic tests like Miller-Rabin ($O(k \log^3 n)$) are
    vastly more efficient.

## Comparison with Other Primality Tests

| Test | Type | Time | Error |
|---|---|---|---|
| Trial division | Deterministic | $O(\sqrt{n})$ | None |
| Miller-Rabin | Probabilistic | $O(k \log^3 n)$ | $\le 4^{-k}$ |
| AKS | Deterministic | $\widetilde{O}(\log^6 n)$ | None |

Trial division is the only method that simultaneously finds the smallest
factor when $n$ is composite, making it useful beyond pure primality testing.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
- Crandall, R. & Pomerance, C. *Prime Numbers: A Computational Perspective*.
  Springer, 2005.
