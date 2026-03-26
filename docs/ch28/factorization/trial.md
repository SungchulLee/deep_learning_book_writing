# Trial Division

The simplest approach to factoring an integer is to test each candidate
divisor one by one.  Trial division is the natural starting point for any
factoring routine: it is easy to implement, requires no advanced number
theory, and efficiently handles numbers with small prime factors.

## Core Idea

Every composite integer $n > 1$ has a prime factor $p \le \sqrt{n}$.

**Proof.**  If $n = ab$ with $1 < a \le b < n$, then $a \le \sqrt{n}$
(otherwise $ab > n$).  Therefore the smallest prime factor of $n$ is at
most $\sqrt{n}$.  $\square$

This means we only need to test divisors up to $\sqrt{n}$.

## Algorithm

To find the complete prime factorization of $n$:

1. For each candidate $d = 2, 3, 5, 7, 11, \dots$ up to $\sqrt{n}$:
      - While $d \mid n$, record $d$ as a factor and replace $n \leftarrow n / d$.
2. If $n > 1$ after the loop, then $n$ itself is prime---record it.

```python
"""
Trial division for integer factorization.

Time : O(sqrt(n))
Space: O(log n) for the list of factors
"""


# === Trial Division ===
def trial_division(n: int) -> list[int]:
    """Return the complete prime factorization of n as a sorted list."""
    if n <= 1:
        return []
    factors = []

    # Handle factor of 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2

    # Test odd divisors from 3 to sqrt(n)
    d = 3
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 2

    # Remaining n is prime
    if n > 1:
        factors.append(n)

    return factors


# === Example ===
if __name__ == "__main__":
    for num in [84, 97, 3600, 1000003]:
        factors = trial_division(num)
        print(f"{num} = {' x '.join(map(str, factors))}")
```

**Expected output:**

```
84 = 2 x 2 x 3 x 7
97 = 97
3600 = 2 x 2 x 2 x 2 x 3 x 3 x 5 x 5
1000003 = 1000003
```

## Complexity Analysis

- **Worst case.**  When $n$ is prime, the loop runs through all odd
  integers up to $\sqrt{n}$, giving $O(\sqrt{n})$ divisions.
- **Best case.**  When $n$ is a power of $2$, factoring takes $O(\log n)$
  divisions.
- **Space.**  The output list has at most $\lfloor \log_2 n \rfloor$ entries.

Each division costs $O(\log^2 n)$ bit operations, so the total bit
complexity is $O(\sqrt{n} \cdot \log^2 n)$.

## Optimizations

### Skip Even Numbers

After checking $d = 2$, increment by $2$ (test only odd candidates).
This halves the number of iterations.

### Wheel Factorization

Extend the idea beyond $2$: after removing factors of $2$, $3$, and $5$,
only test $d$ values coprime to $30$.  The increments follow a repeating
pattern of length 8:

$$
\Delta = [1, 7, 11, 13, 17, 19, 23, 29] + 30k
$$

This skips $73\%$ of candidates instead of $50\%$ with odd-only testing.

```python
"""
Trial division with 2-3-5 wheel factorization.
"""


# === Wheel Trial Division ===
def trial_division_wheel(n: int) -> list[int]:
    """Factor n using 2-3-5 wheel to skip non-coprime candidates."""
    factors = []
    for p in (2, 3, 5):
        while n % p == 0:
            factors.append(p)
            n //= p
    increments = [4, 2, 4, 2, 4, 6, 2, 6]
    d = 7
    i = 0
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += increments[i]
        i = (i + 1) % 8
    if n > 1:
        factors.append(n)
    return factors


# === Example ===
if __name__ == "__main__":
    print(trial_division_wheel(2 * 3 * 5 * 7 * 11 * 13))  # [2, 3, 5, 7, 11, 13]
```

### Precomputed Small Primes

Use a sieve to generate all primes up to $\sqrt{n}$ first, then only test
prime divisors.  This reduces the iteration count by a factor of
$\ln \sqrt{n}$ (by the prime number theorem).

## When to Use Trial Division

| Scenario | Recommendation |
|---|---|
| $n < 10^{12}$ | Trial division alone suffices |
| $n$ up to $10^{18}$ | Trial division for small factors, then Pollard's rho |
| $n > 10^{20}$ | Trial division only as a first pass |

!!! tip "Combining Methods"
    In practice, factoring routines start with trial division up to a small
    bound (e.g., $10^6$), then switch to Pollard's rho or the Quadratic
    Sieve for the remaining cofactor.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
- Crandall, R. & Pomerance, C. *Prime Numbers: A Computational Perspective*.
  Springer, 2005.
