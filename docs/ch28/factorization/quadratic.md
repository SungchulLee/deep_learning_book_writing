# Quadratic Sieve

For numbers beyond the reach of Pollard's rho (roughly 30-digit factors),
the Quadratic Sieve (QS) is the method of choice for factoring integers up
to about 100 digits.  It was the fastest known general-purpose factoring
algorithm until the Number Field Sieve superseded it in the 1990s.

## Mathematical Foundation

The Quadratic Sieve exploits a classical observation due to Fermat:

> If we find integers $x$ and $y$ such that $x^2 \equiv y^2 \pmod{n}$ but
> $x \not\equiv \pm y \pmod{n}$, then $\gcd(x - y, n)$ is a non-trivial
> factor of $n$.

The challenge is finding such a congruence of squares.

## Overview of the Algorithm

The algorithm proceeds in three phases:

### Phase 1 --- Sieving

Choose a **factor base** $\mathcal{B} = \{p_1, p_2, \dots, p_k\}$
consisting of small primes $p$ for which $n$ is a quadratic residue modulo
$p$ (i.e., the Legendre symbol $(n/p) = 1$).

For values of $x$ near $\lceil \sqrt{n} \rceil$, compute

$$
Q(x) = x^2 - n
$$

and attempt to factor $Q(x)$ over the factor base.  If $Q(x)$ factors
completely over $\mathcal{B}$, it is called **$\mathcal{B}$-smooth**:

$$
Q(x) = \prod_{i=1}^{k} p_i^{e_i}
$$

The sieving step identifies many such smooth values efficiently by
subtracting $\ln p$ for each prime $p$ that divides $Q(x)$.

### Phase 2 --- Linear Algebra

Collect at least $k + 1$ smooth relations (by the pigeonhole principle,
this guarantees a linear dependency).  Write each relation as a vector of
exponents modulo 2:

$$
\mathbf{v}_x = (e_1 \bmod 2, \, e_2 \bmod 2, \, \dots, \, e_k \bmod 2)
$$

Find a subset of these vectors that sums to the zero vector over
$\mathbb{F}_2$ using Gaussian elimination.  This subset gives a
congruence of squares.

### Phase 3 --- Factor Extraction

The selected subset yields

$$
\left(\prod_{x \in S} x\right)^2 \equiv \left(\prod_{x \in S} Q(x)\right) \equiv y^2 \pmod{n}
$$

Compute $\gcd\!\left(\prod_{x \in S} x - y, \, n\right)$.  With probability
at least $1/2$, this is a non-trivial factor.  If not, try another linear
dependency.

## Complexity

The running time of the Quadratic Sieve is sub-exponential:

$$
L_n\!\left[\frac{1}{2}, 1\right] = \exp\!\left(\sqrt{\ln n \cdot \ln \ln n}\right)
$$

This notation $L_n[u, v] = \exp\!\bigl(v (\ln n)^u (\ln \ln n)^{1-u}\bigr)$
interpolates between polynomial ($u = 0$) and exponential ($u = 1$).

| Algorithm | Complexity | Best for |
|---|---|---|
| Trial Division | $O(\sqrt{n})$ | Small numbers |
| Pollard's Rho | $O(n^{1/4})$ | Numbers up to ~60 digits |
| Quadratic Sieve | $L_n[1/2, 1]$ | Numbers up to ~100 digits |
| Number Field Sieve | $L_n[1/3, c]$ | Numbers beyond ~100 digits |

## Simplified Example

```python
"""
Simplified demonstration of the Quadratic Sieve concept.

This illustrates the congruence-of-squares approach on small numbers.
Full QS implementations require optimized sieving and sparse linear algebra.
"""

import math
from itertools import combinations


# === Factor Base Selection ===
def build_factor_base(n: int, bound: int) -> list[int]:
    """Return primes up to bound where n is a quadratic residue."""
    primes = []
    for p in range(2, bound + 1):
        if all(p % i != 0 for i in range(2, int(p**0.5) + 1)) or p == 2:
            if p == 2 or pow(n, (p - 1) // 2, p) == 1:
                primes.append(p)
    return primes


# === Smooth Check ===
def try_factor_over_base(value: int, base: list[int]) -> list[int] | None:
    """Factor value over the factor base. Return exponents or None."""
    if value == 0:
        return None
    exponents = []
    v = abs(value)
    for p in base:
        e = 0
        while v % p == 0:
            v //= p
            e += 1
        exponents.append(e)
    return exponents if v == 1 else None


# === Congruence of Squares ===
def quadratic_sieve_demo(n: int) -> int | None:
    """Demonstrate QS on a small composite n."""
    base = build_factor_base(n, 30)
    root = math.isqrt(n)

    # Collect smooth relations
    relations = []
    for x in range(root + 1, root + 1000):
        q = x * x - n
        exps = try_factor_over_base(q, base)
        if exps is not None:
            relations.append((x, q, exps))
        if len(relations) > len(base) + 5:
            break

    # Try subsets for congruence of squares
    for size in range(2, min(len(relations), 6) + 1):
        for combo in combinations(range(len(relations)), size):
            combined = [0] * len(base)
            for idx in combo:
                for j, e in enumerate(relations[idx][2]):
                    combined[j] += e
            if all(e % 2 == 0 for e in combined):
                x_prod = 1
                y_sq = 1
                for idx in combo:
                    x_prod = (x_prod * relations[idx][0]) % n
                    y_sq *= relations[idx][1]
                y = math.isqrt(y_sq)
                if y * y == y_sq:
                    g = math.gcd(abs(x_prod - y), n)
                    if 1 < g < n:
                        return g
    return None


# === Example ===
if __name__ == "__main__":
    n = 15347  # = 103 * 149
    factor = quadratic_sieve_demo(n)
    if factor:
        print(f"n = {n}")
        print(f"Factor found: {factor}")
        print(f"Other factor: {n // factor}")
    else:
        print("No factor found (try larger sieve interval)")
```

!!! warning "Production vs. Demonstration"
    Real Quadratic Sieve implementations use polynomial sieving, large prime
    variations, and block Lanczos or structured Gaussian elimination over
    $\mathbb{F}_2$ for the linear algebra phase.  The code above is purely
    educational.

## Key Optimizations

- **Multiple Polynomial QS (MPQS).**  Use several polynomials instead of a
  single $Q(x) = x^2 - n$ to spread the sieving over a wider range and
  find more smooth values.
- **Large prime variation.**  Allow relations with one factor slightly larger
  than the factor base bound; combine two such partial relations into a full
  relation.
- **Block Lanczos.**  Solve the $\mathbb{F}_2$ linear system using block
  methods that process 64 bits at once using word-level operations.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
- Pomerance, C. "The Quadratic Sieve Factoring Algorithm." *Advances in
  Cryptology*, EUROCRYPT 1984.
