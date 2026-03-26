# Sieve of Eratosthenes

Finding all prime numbers up to a given limit is a foundational task in
number theory and algorithm design.  Rather than testing each number
individually, the Sieve of Eratosthenes systematically eliminates composites,
producing all primes up to $n$ in $O(n \log \log n)$ time --- nearly linear
in $n$.

## Core Idea

Every composite number $m$ has a prime factor $p \le \sqrt{m}$.  The sieve
works by iterating through small primes and marking their multiples as
composite.  Whatever remains unmarked is prime.

## Algorithm

1. Create a boolean array `is_prime[0..n]`, initialized to `True`.
2. Set `is_prime[0] = is_prime[1] = False`.
3. For each $i$ from $2$ to $\lfloor \sqrt{n} \rfloor$:
    - If `is_prime[i]` is `True`, mark all multiples $i^2, i^2 + i,
      i^2 + 2i, \dots \le n$ as `False`.
4. All indices still marked `True` are prime.

!!! note "Why Start at i-squared"
    When processing prime $p$, all multiples $2p, 3p, \dots, (p-1)p$ have
    already been marked by smaller primes.  Starting at $p^2$ avoids
    redundant work.

## Implementation

```python
"""
Sieve of Eratosthenes.

Time : O(n log log n)
Space: O(n)
"""

import math


# === Sieve of Eratosthenes ===
def sieve_of_eratosthenes(n: int) -> list[int]:
    """Return all primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            # Mark multiples starting from i*i
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(2, n + 1) if is_prime[i]]


# === Example ===
if __name__ == "__main__":
    n = 50
    primes = sieve_of_eratosthenes(n)
    print(f"Primes up to {n}: {primes}")
    print(f"Count: {len(primes)}")

    # Prime counting function comparison
    for limit in [100, 1000, 10000]:
        count = len(sieve_of_eratosthenes(limit))
        approx = int(limit / math.log(limit))
        print(f"pi({limit}) = {count}, n/ln(n) ~ {approx}")
```

## Correctness

**Claim.**  After the sieve completes, `is_prime[k]` is `True` if and only
if $k$ is prime.

**Proof.**  If $k$ is composite, write $k = p \cdot q$ where $p$ is the
smallest prime factor of $k$.  Then $p \le \sqrt{k} \le \sqrt{n}$, so the
outer loop processes $p$.  The inner loop marks $k = p \cdot q$ because
$k \ge p^2$ (since $q \ge p$).  Conversely, if $k$ is prime, no smaller
prime divides $k$, so $k$ is never marked.  $\square$

## Complexity Analysis

### Time Complexity

The total number of markings is

$$
\sum_{\substack{p \le \sqrt{n} \\ p \text{ prime}}} \frac{n}{p} = n \sum_{\substack{p \le \sqrt{n} \\ p \text{ prime}}} \frac{1}{p}
$$

By Mertens' theorem, $\sum_{p \le x} 1/p = \ln \ln x + M + O(1/\ln x)$
where $M \approx 0.2615$ is Mertens' constant.  Therefore the total work is

$$
O(n \log \log n)
$$

### Space Complexity

The boolean array uses $O(n)$ space.  This can be reduced to $O(\sqrt{n})$
using the segmented sieve variant.

## Optimizations

### Odd-Only Sieve

Since $2$ is the only even prime, skip even numbers entirely and halve the
memory:

```python
"""
Odd-only sieve variant (halves memory usage).
"""


# === Odd-Only Sieve ===
def sieve_odd_only(n: int) -> list[int]:
    """Return all primes up to n, sieving only odd numbers."""
    if n < 2:
        return []
    if n == 2:
        return [2]

    # is_prime[i] represents the number 2*i + 1
    size = (n - 1) // 2
    is_prime = [True] * (size + 1)

    for i in range(1, (int(math.isqrt(n)) - 1) // 2 + 1):
        if is_prime[i]:
            p = 2 * i + 1
            # Start marking from p*p; index of p*p is (p*p - 1) // 2
            start = (p * p - 1) // 2
            for j in range(start, size + 1, p):
                is_prime[j] = False

    primes = [2]
    for i in range(1, size + 1):
        if is_prime[i]:
            primes.append(2 * i + 1)
    return primes


# === Example ===
if __name__ == "__main__":
    print(sieve_odd_only(50))
```

### Bitwise Sieve

Replace the boolean array with a bit array to reduce memory by a factor
of 8.  Combined with odd-only sieving, this gives a 16x memory reduction
compared to the naive approach.

## Prime Number Theorem Connection

The number of primes up to $n$, denoted $\pi(n)$, satisfies

$$
\pi(n) \sim \frac{n}{\ln n}
$$

This means the sieve outputs approximately $n / \ln n$ primes, and the
density of primes near $n$ is roughly $1 / \ln n$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
- Hardy, G. H. & Wright, E. M. *An Introduction to the Theory of Numbers*.
  Oxford University Press.
