# Segmented Sieve

The classic Sieve of Eratosthenes needs $O(n)$ memory to find all primes up
to $n$.  For large $n$ (e.g., $10^{12}$), this array does not fit in memory.
The segmented sieve solves this by processing the range in fixed-size blocks,
reducing memory to $O(\sqrt{n})$ while maintaining the same time complexity.

## Prerequisites

The key observation is that any composite number $m \le n$ has a prime factor
$p \le \sqrt{n}$.  Therefore, to sieve any block of numbers, we only need the
primes up to $\sqrt{n}$.

## Algorithm

### Step 1 --- Base Primes

Use the standard Sieve of Eratosthenes to find all primes up to
$\sqrt{n}$.  This requires $O(\sqrt{n})$ memory and
$O(\sqrt{n} \log \log \sqrt{n})$ time.

### Step 2 --- Block Processing

Divide the range $[\sqrt{n} + 1, n]$ into blocks of size $\Delta$ (typically
$\Delta = \sqrt{n}$ or a cache-friendly value like $2^{18}$).  For each
block $[L, L + \Delta)$:

1. Initialize a boolean array `is_prime[0..\Delta-1]` to all `True`.
2. For each base prime $p$:
    - Find the smallest multiple of $p$ in $[L, L + \Delta)$:

    $$
    \text{start} = \left\lceil \frac{L}{p} \right\rceil \cdot p
    $$

    If $\text{start} = p$, advance to $2p$ (the prime itself is not composite).
    - Mark all multiples of $p$ in the block as composite.
3. Collect unmarked positions as primes.

## Implementation

```python
"""
Segmented Sieve of Eratosthenes.

Time : O(n log log n)  — same as basic sieve
Space: O(sqrt(n))      — only base primes + one block
"""

import math


# === Basic Sieve (for base primes) ===
def simple_sieve(limit: int) -> list[int]:
    """Return all primes up to limit using standard sieve."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]


# === Segmented Sieve ===
def segmented_sieve(n: int) -> list[int]:
    """Return all primes up to n using segmented sieve."""
    if n < 2:
        return []

    limit = int(math.isqrt(n)) + 1
    base_primes = simple_sieve(limit)

    # Primes from base sieve
    primes = [p for p in base_primes if p <= n]

    # Process segments
    delta = max(limit, 1)
    low = limit + 1

    while low <= n:
        high = min(low + delta - 1, n)
        block_size = high - low + 1
        is_prime = [True] * block_size

        for p in base_primes:
            # First multiple of p >= low
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, high + 1, p):
                is_prime[j - low] = False

        for i in range(block_size):
            if is_prime[i]:
                primes.append(low + i)

        low = high + 1

    return primes


# === Example ===
if __name__ == "__main__":
    n = 100
    result = segmented_sieve(n)
    print(f"Primes up to {n}: {result}")
    print(f"Count: {len(result)}")

    # Larger example
    n_large = 10_000
    count = len(segmented_sieve(n_large))
    print(f"Number of primes up to {n_large}: {count}")
```

## Complexity Analysis

| Aspect | Basic Sieve | Segmented Sieve |
|---|---|---|
| Time | $O(n \log \log n)$ | $O(n \log \log n)$ |
| Space | $O(n)$ | $O(\sqrt{n})$ |
| Cache behavior | Poor for large $n$ | Excellent (block fits in L1/L2) |

The time complexity is identical because every composite number is still
crossed out the same number of times.  The memory improvement comes from
reusing a single block-sized array.

## Block Size Selection

The block size $\Delta$ affects cache performance:

- **$\Delta = \sqrt{n}$** minimizes the number of blocks and is the
  theoretical default.
- **$\Delta = $ L1 cache size / 8** (in bytes) yields the best practical
  performance because each block fits entirely in the fastest cache level.

!!! tip "Cache-Friendly Sieving"
    On modern CPUs with 32 KB L1 cache, setting $\Delta = 2^{15} = 32{,}768$
    is a good practical choice.  This ensures the boolean array and loop
    variables all reside in L1 cache during sieving.

## Sieving a Range

The segmented sieve generalizes naturally to find primes in an arbitrary
range $[a, b]$ without sieving from $2$:

1. Compute base primes up to $\sqrt{b}$.
2. Sieve the single block $[a, b]$.

This is useful for problems like "count primes between $10^{12}$ and
$10^{12} + 10^6$" where the full sieve from $2$ is unnecessary.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
- Crandall, R. & Pomerance, C. *Prime Numbers: A Computational Perspective*.
  Springer, 2005.
