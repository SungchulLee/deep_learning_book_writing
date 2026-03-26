# Miller-Rabin Primality Test

Testing whether a large number is prime is fundamental to cryptography,
where RSA key generation requires finding primes with hundreds of digits.
Trial division takes $O(\sqrt{n})$ time — infeasible for 1024-bit numbers.
The **Miller-Rabin test** is a Monte Carlo algorithm that determines
primality in $O(k \log^2 n)$ time with error probability at most $4^{-k}$,
where $k$ is the number of rounds.

## Background: Fermat's Little Theorem

If $p$ is prime and $\gcd(a, p) = 1$, then:

$$
a^{p-1} \equiv 1 \pmod{p}
$$

A **Fermat witness** for compositeness of $n$ is a value $a$ such that
$a^{n-1} \not\equiv 1 \pmod{n}$. However, Carmichael numbers pass the
Fermat test for all bases coprime to $n$ despite being composite.

## The Miller-Rabin Improvement

Miller-Rabin strengthens the Fermat test by exploiting a property of primes
related to square roots of unity.

Write $n - 1 = 2^s \cdot d$ where $d$ is odd. If $n$ is prime and
$\gcd(a, n) = 1$, then either:

1. $a^d \equiv 1 \pmod{n}$, or
2. $a^{2^r d} \equiv -1 \pmod{n}$ for some $0 \le r < s$.

!!! note "Why This Works"
    If $n$ is prime, the sequence $a^d, a^{2d}, a^{4d}, \ldots, a^{2^s d} = a^{n-1}$
    must end at $1$ (by Fermat). Working backwards, the value just before
    the first $1$ must be $-1$ (since $\pm 1$ are the only square roots
    of $1$ modulo a prime).

## Algorithm

**Input:** Odd integer $n > 3$ and number of rounds $k$.

1. Write $n - 1 = 2^s \cdot d$ with $d$ odd.
2. Repeat $k$ times:
    - Choose random $a \in \{2, 3, \ldots, n - 2\}$.
    - Compute $x = a^d \bmod n$.
    - If $x = 1$ or $x = n - 1$: this round passes (continue).
    - For $r = 1, 2, \ldots, s - 1$:
        - $x = x^2 \bmod n$
        - If $x = n - 1$: this round passes (break).
    - If we never found $x = n - 1$: **composite** (return).
3. Return **probably prime**.

## Error Analysis

**Theorem.** If $n$ is composite, the probability that Miller-Rabin declares
it "probably prime" after $k$ rounds is at most $4^{-k}$.

More precisely, at least $3/4$ of the values $a \in \{1, \ldots, n-1\}$ are
Miller-Rabin witnesses for any composite $n$. This is stronger than the
Fermat test, where Carmichael numbers have *no* witnesses.

| Rounds $k$ | Error probability |
|---|---|
| 1 | $\le 1/4$ |
| 10 | $\le 10^{-6}$ |
| 20 | $\le 10^{-12}$ |
| 40 | $\le 10^{-24}$ |

## Deterministic Variants

For small $n$, specific sets of bases guarantee correctness:

| Range of $n$ | Sufficient bases |
|---|---|
| $< 2{,}047$ | $\{2\}$ |
| $< 1{,}373{,}653$ | $\{2, 3\}$ |
| $< 3{,}215{,}031{,}751$ | $\{2, 3, 5, 7\}$ |
| $< 3.3 \times 10^{24}$ | First 13 primes |

## Implementation

```python
"""
Miller-Rabin primality test.

A Monte Carlo algorithm that tests whether a number is prime with
error probability at most 4^{-k} for k rounds.
"""

import random


# === Modular Exponentiation ===

def power_mod(base, exp, mod):
    """Compute base^exp mod mod using repeated squaring.

    Time complexity: O(log exp * log^2 mod) with Python's big integers.
    """
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp //= 2
        base = (base * base) % mod
    return result


# === Miller-Rabin Test ===

def miller_rabin(n, k=20):
    """Test if n is probably prime using k rounds of Miller-Rabin.

    Args:
        n: integer to test (must be > 1).
        k: number of rounds (error <= 4^{-k}).

    Returns:
        True if probably prime, False if definitely composite.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 = 2^s * d with d odd
    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    # k rounds of testing
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = power_mod(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False  # Composite

    return True  # Probably prime


# === Deterministic Miller-Rabin ===

def is_prime_deterministic(n):
    """Deterministic primality test for n < 3.3 * 10^24.

    Uses specific bases that guarantee correctness.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    # Sufficient witnesses for n < 3,317,044,064,679,887,385,961,981
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

    for a in witnesses:
        if a >= n:
            continue
        x = power_mod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False

    return True


# === Main ===

if __name__ == "__main__":
    random.seed(42)

    # Test small numbers
    primes = [n for n in range(2, 50) if miller_rabin(n)]
    print(f"Primes < 50: {primes}")

    # Test large numbers
    large_prime = 104729  # Known prime
    large_composite = 104723 * 3  # Known composite
    print(f"\n{large_prime} is prime: {miller_rabin(large_prime)}")
    print(f"{large_composite} is prime: {miller_rabin(large_composite)}")

    # Carmichael number (561 = 3 * 11 * 17)
    print(f"\n561 (Carmichael): {miller_rabin(561)}")

    # Deterministic test
    print(f"\nDeterministic test on 104729: {is_prime_deterministic(104729)}")
    print(f"Deterministic test on 561: {is_prime_deterministic(561)}")
```

**Output:**
```
Primes < 50: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

104729 is prime: True
314169 is prime: False

561 (Carmichael): False

Deterministic test on 104729: True
Deterministic test on 561: False
```

## Complexity

| Operation | Time |
|---|---|
| One round | $O(\log^2 n)$ (dominated by modular exponentiation) |
| $k$ rounds | $O(k \log^2 n)$ |
| Space | $O(\log n)$ |

## Classification

Miller-Rabin is a **Monte Carlo** algorithm:

- It always runs in polynomial time.
- It may err: a composite number may be declared "probably prime."
- It never errs on primes: if it says "composite," the number is definitely composite.

This is a one-sided error algorithm (false positives but no false negatives).

## Reference

- Rabin, M. O. "Probabilistic Algorithm for Testing Primality." *J. Number Theory*, 1980.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms*. MIT Press, Chapter 31.
