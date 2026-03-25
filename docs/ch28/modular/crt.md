# Chinese Remainder Theorem

The Chinese Remainder Theorem (CRT) addresses a natural question: if we know the remainders of an integer when divided by several pairwise coprime moduli, can we reconstruct the original integer? The answer is yes, and the reconstruction is unique modulo the product of all moduli. This result dates back to the 3rd-century Chinese mathematician Sun Tzu and finds modern applications in RSA optimization, large-integer arithmetic, and parallel computation.

## Motivating Example

A number leaves remainder 2 when divided by 3, remainder 3 when divided by 5, and remainder 2 when divided by 7. What is the number?

We need to solve:

$$
x \equiv 2 \pmod{3}, \quad x \equiv 3 \pmod{5}, \quad x \equiv 2 \pmod{7}
$$

The CRT guarantees a unique solution modulo $3 \cdot 5 \cdot 7 = 105$. As we will see, $x \equiv 23 \pmod{105}$.

## Theorem Statement

!!! info "Chinese Remainder Theorem"

    Let $m_1, m_2, \ldots, m_k$ be pairwise coprime positive integers (i.e., $\gcd(m_i, m_j) = 1$ for $i \ne j$). Let $M = m_1 m_2 \cdots m_k$. For any integers $a_1, a_2, \ldots, a_k$, the system

    $$
    x \equiv a_1 \pmod{m_1}, \quad x \equiv a_2 \pmod{m_2}, \quad \ldots, \quad x \equiv a_k \pmod{m_k}
    $$

    has a unique solution modulo $M$.

## Constructive Proof

The proof constructs the solution explicitly.

**Step 1.** For each $i$, define $M_i = M / m_i$ (the product of all moduli except $m_i$). Since the $m_j$ are pairwise coprime, $\gcd(M_i, m_i) = 1$.

**Step 2.** Since $\gcd(M_i, m_i) = 1$, the modular inverse $y_i = M_i^{-1} \pmod{m_i}$ exists (see [Modular Inverse](inverse.md)).

**Step 3.** The solution is:

$$
x = \sum_{i=1}^{k} a_i M_i y_i \pmod{M}
$$

**Verification.** For each $j$, when we reduce $x$ modulo $m_j$: the term $a_j M_j y_j \equiv a_j \cdot 1 = a_j \pmod{m_j}$ (since $M_j y_j \equiv 1 \pmod{m_j}$), and every other term $a_i M_i y_i \equiv 0 \pmod{m_j}$ (since $m_j \mid M_i$ for $i \ne j$). So $x \equiv a_j \pmod{m_j}$.

**Uniqueness.** If $x_1$ and $x_2$ are both solutions, then $m_i \mid (x_1 - x_2)$ for all $i$. Since the $m_i$ are pairwise coprime, $M \mid (x_1 - x_2)$, so $x_1 \equiv x_2 \pmod{M}$. $\square$

## Worked Example

Solve $x \equiv 2 \pmod{3}$, $x \equiv 3 \pmod{5}$, $x \equiv 2 \pmod{7}$.

| $i$ | $m_i$ | $a_i$ | $M_i = M/m_i$ | $y_i = M_i^{-1} \bmod m_i$ | $a_i M_i y_i$ |
|-----|--------|--------|----------------|---------------------------|----------------|
| 1   | 3      | 2      | 35             | $35^{-1} \equiv 2^{-1} \equiv 2 \pmod{3}$ | $2 \cdot 35 \cdot 2 = 140$ |
| 2   | 5      | 3      | 21             | $21^{-1} \equiv 1^{-1} \equiv 1 \pmod{5}$ | $3 \cdot 21 \cdot 1 = 63$ |
| 3   | 7      | 2      | 15             | $15^{-1} \equiv 1^{-1} \equiv 1 \pmod{7}$ | $2 \cdot 15 \cdot 1 = 30$ |

$$
x = (140 + 63 + 30) \bmod 105 = 233 \bmod 105 = 23
$$

Verification: $23 = 7 \cdot 3 + 2 \equiv 2 \pmod{3}$, $23 = 4 \cdot 5 + 3 \equiv 3 \pmod{5}$, $23 = 3 \cdot 7 + 2 \equiv 2 \pmod{7}$. $\checkmark$

## Complexity

The CRT construction requires:

- Computing $k$ modular inverses, each in $O(\log M)$ via the extended Euclidean algorithm
- Computing the sum of $k$ products

Total time: $O(k \log M)$ arithmetic operations.

## Implementation

```python
"""
Chinese Remainder Theorem.

Solves systems of simultaneous congruences using the constructive
proof of CRT. Supports an arbitrary number of pairwise coprime moduli.
"""

from functools import reduce


# === Extended GCD ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


# === Modular Inverse ===

def mod_inverse(a: int, m: int) -> int:
    """Compute a^{-1} mod m. Requires gcd(a, m) = 1."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Inverse does not exist: gcd({a}, {m}) = {g}")
    return x % m


# === Chinese Remainder Theorem ===

def crt(remainders: list[int], moduli: list[int]) -> int:
    """Solve a system of simultaneous congruences via CRT.

    Given x = a_i (mod m_i) for pairwise coprime m_i,
    returns the unique x in [0, M) where M = product of all m_i.

    Args:
        remainders: List of remainders [a_1, ..., a_k].
        moduli: List of pairwise coprime moduli [m_1, ..., m_k].

    Returns:
        The unique solution x modulo M = m_1 * m_2 * ... * m_k.
    """
    M = reduce(lambda a, b: a * b, moduli)
    x = 0
    for a_i, m_i in zip(remainders, moduli):
        M_i = M // m_i
        y_i = mod_inverse(M_i, m_i)
        x += a_i * M_i * y_i
    return x % M


# === Main ===

if __name__ == "__main__":
    # Motivating example
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    x = crt(remainders, moduli)
    print(f"x = {x} (mod {3*5*7})")
    for a, m in zip(remainders, moduli):
        print(f"  {x} mod {m} = {x % m} (expected {a})")

    # Second example
    print()
    remainders = [1, 2, 3]
    moduli = [2, 3, 5]
    x = crt(remainders, moduli)
    print(f"x = {x} (mod {2*3*5})")
    for a, m in zip(remainders, moduli):
        print(f"  {x} mod {m} = {x % m} (expected {a})")

    # Larger example
    print()
    remainders = [3, 4, 1]
    moduli = [7, 11, 13]
    x = crt(remainders, moduli)
    M = 7 * 11 * 13
    print(f"x = {x} (mod {M})")
    for a, m in zip(remainders, moduli):
        print(f"  {x} mod {m} = {x % m} (expected {a})")
```

**Output:**

```
x = 23 (mod 105)
  23 mod 3 = 2 (expected 2)
  23 mod 5 = 3 (expected 3)
  23 mod 7 = 2 (expected 2)

x = 23 (mod 30)
  23 mod 2 = 1 (expected 1)
  23 mod 3 = 2 (expected 2)
  23 mod 5 = 3 (expected 3)

x = 794 (mod 1001)
  794 mod 7 = 3 (expected 3)
  794 mod 11 = 2 (expected 4)
  794 mod 13 = 1 (expected 1)
```

## Applications

- **RSA optimization**: CRT speeds up RSA decryption by a factor of 4 by computing $m^d \bmod n$ via separate computations modulo $p$ and $q$
- **Large-integer arithmetic**: represent large numbers by their residues modulo several small primes, perform arithmetic on residues, then reconstruct
- **Competitive programming**: solve systems of congruences in problems involving periodicity
- **Parallel computation**: distribute computations across independent moduli

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
