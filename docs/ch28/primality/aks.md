# AKS Primality Test

Before 2002, all known deterministic primality tests were either exponential
or relied on unproven conjectures.  The AKS test, due to Agrawal, Kayal, and
Saxena, settled a long-standing open question by providing the first
**deterministic polynomial-time** algorithm for primality.

## Key Result

The AKS algorithm proves that PRIMES $\in$ P: given an integer $n$, it
decides whether $n$ is prime in time polynomial in $\log n$ (the number of
digits).

## Mathematical Foundation

The test builds on a generalization of Fermat's little theorem to
polynomial rings.

!!! note "Core Identity"
    An integer $n \ge 2$ is prime if and only if the polynomial congruence

    $$
    (x + a)^n \equiv x^n + a \pmod{n}
    $$

    holds for every integer $a$ coprime to $n$.

Checking this identity directly requires expanding $(x + a)^n$, which has
$n + 1$ terms and is thus exponential.  The AKS insight is to check the
identity modulo $x^r - 1$ for a carefully chosen small $r$:

$$
(x + a)^n \equiv x^n + a \pmod{x^r - 1, \, n}
$$

## Algorithm

**Input:** Integer $n \ge 2$.

1. **Perfect power check.**  If $n = b^k$ for integers $b \ge 2$ and
   $k \ge 2$, output COMPOSITE.
2. **Find suitable $r$.**  Find the smallest $r$ such that $\text{ord}_r(n) > (\log_2 n)^2$,
   where $\text{ord}_r(n)$ is the multiplicative order of $n$ modulo $r$.
3. **GCD checks.**  For all $a \le r$, if $1 < \gcd(a, n) < n$, output
   COMPOSITE.
4. **Small $n$ check.**  If $n \le r$, output PRIME.
5. **Polynomial checks.**  For $a = 1, 2, \dots, \lfloor \sqrt{\phi(r)} \log_2 n \rfloor$,
   verify

    $$
    (x + a)^n \equiv x^n + a \pmod{x^r - 1, \, n}
    $$

    If any check fails, output COMPOSITE.

6. Output PRIME.

## Correctness Sketch

The algorithm relies on two directions:

- **Completeness.**  If $n$ is prime, the polynomial identity holds for all
  $a$ (a consequence of the binomial theorem modulo $p$), so the algorithm
  always outputs PRIME.
- **Soundness.**  If $n$ is composite and passes all polynomial checks, one
  can derive a contradiction using properties of the multiplicative group in
  $\mathbb{F}_p[x]/(x^r - 1)$ where $p$ is a prime factor of $n$.  The
  bound on $r$ and the number of $a$-values ensure enough constraints to
  force $n$ to be a prime power, which step 1 already ruled out.

## Complexity

- **Finding $r$:** $r = O((\log n)^5)$ suffices (later improved to
  $O((\log n)^3)$ in some variants).
- **Polynomial checks:** Each check multiplies polynomials of degree $< r$
  modulo $n$, costing $\widetilde{O}(r \log n)$ per check.
- **Number of checks:** $O(\sqrt{r} \log n)$.
- **Total:** $\widetilde{O}(r^{3/2} (\log n)^2)$, which is
  $\widetilde{O}((\log n)^{21/2})$ with the original bound on $r$.

Subsequent improvements by Lenstra and Pomerance reduced the complexity to
$\widetilde{O}((\log n)^6)$.

| Primality Test | Type | Complexity |
|---|---|---|
| Trial division | Deterministic | $O(\sqrt{n})$ |
| Miller-Rabin | Probabilistic | $O(k \log^2 n)$ per round |
| AKS | Deterministic | $\widetilde{O}((\log n)^6)$ |

## Simplified Implementation

```python
"""
Simplified AKS primality test (educational version).

The full AKS test requires polynomial arithmetic modulo (x^r - 1, n).
This implementation demonstrates the structure of the algorithm.
"""

import math


# === Perfect Power Check ===
def is_perfect_power(n: int) -> bool:
    """Check if n = b^k for some b >= 2, k >= 2."""
    if n <= 3:
        return False
    for k in range(2, n.bit_length() + 1):
        b = round(n ** (1.0 / k))
        for candidate in (b - 1, b, b + 1):
            if candidate >= 2 and candidate**k == n:
                return True
    return False


# === Multiplicative Order ===
def multiplicative_order(n: int, r: int) -> int:
    """Return the smallest k such that n^k = 1 (mod r)."""
    if math.gcd(n, r) > 1:
        return 0
    result = 1
    power = n % r
    while power != 1:
        power = (power * n) % r
        result += 1
    return result


# === Polynomial Multiplication mod (x^r - 1, n) ===
def poly_mult_mod(a: list[int], b: list[int], r: int, n: int) -> list[int]:
    """Multiply polynomials a and b modulo x^r - 1 and n."""
    result = [0] * r
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            result[(i + j) % r] = (result[(i + j) % r] + ai * bj) % n
    return result


# === Polynomial Power mod (x^r - 1, n) ===
def poly_pow_mod(base: list[int], exp: int, r: int, n: int) -> list[int]:
    """Compute base^exp modulo x^r - 1 and n."""
    result = [0] * r
    result[0] = 1
    b = base[:]
    while exp > 0:
        if exp % 2 == 1:
            result = poly_mult_mod(result, b, r, n)
        b = poly_mult_mod(b, b, r, n)
        exp //= 2
    return result


# === AKS Test ===
def aks(n: int) -> bool:
    """Return True if n is prime (AKS algorithm)."""
    if n <= 1:
        return False
    if n <= 3:
        return True

    # Step 1: perfect power check
    if is_perfect_power(n):
        return False

    # Step 2: find r
    log2n = math.log2(n)
    threshold = log2n * log2n
    r = 2
    while r < n:
        if math.gcd(r, n) > 1 and r < n:
            if math.gcd(r, n) == n:
                r += 1
                continue
            return False
        if multiplicative_order(n, r) > threshold:
            break
        r += 1

    # Step 3: GCD checks
    for a in range(2, min(r + 1, n)):
        g = math.gcd(a, n)
        if 1 < g < n:
            return False

    # Step 4: small n
    if n <= r:
        return True

    # Step 5: polynomial checks
    limit = int(math.sqrt(r) * log2n) + 1
    for a in range(1, limit + 1):
        # Compute (x + a)^n mod (x^r - 1, n)
        poly = [0] * r
        poly[0] = a % n
        poly[1 % r] = (poly[1 % r] + 1) % n
        lhs = poly_pow_mod(poly, n, r, n)

        # Expected: x^(n mod r) + a
        rhs = [0] * r
        rhs[n % r] = 1
        rhs[0] = (rhs[0] + a) % n

        if lhs != rhs:
            return False

    return True


# === Example ===
if __name__ == "__main__":
    test_values = [2, 7, 10, 13, 15, 31, 37, 49, 97]
    for val in test_values:
        result = "prime" if aks(val) else "composite"
        print(f"AKS({val}) = {result}")
```

!!! warning "Performance Note"
    AKS is primarily of theoretical importance.  In practice, Miller-Rabin
    with enough rounds (or a deterministic variant for small $n$) is far
    faster.  AKS proves that a polynomial-time deterministic test *exists*,
    but its constants make it impractical for large inputs.

## Historical Significance

The AKS result resolved the complexity of primality testing:

- **Before AKS:** PRIMES was known to be in co-NP and in BPP (via
  Miller-Rabin), but not known to be in P.
- **After AKS (2002):** PRIMES $\in$ P, unconditionally.

## Reference

- Agrawal, M., Kayal, N., & Saxena, N. "PRIMES is in P." *Annals of
  Mathematics*, 160(2), 2004.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
