# Modular Inverse

Division is not directly defined in modular arithmetic, but we can achieve the same effect by multiplying by a **modular inverse**. The modular inverse of $a$ modulo $m$ is the integer $a^{-1}$ such that $a \cdot a^{-1} \equiv 1 \pmod{m}$. This operation is essential for RSA decryption, solving modular equations, and computing combinatorial quantities modulo a prime.

## Definition and Existence

The **modular multiplicative inverse** of $a$ modulo $m$ is an integer $x$ such that:

$$
ax \equiv 1 \pmod{m}
$$

We write $x = a^{-1} \pmod{m}$.

!!! info "Existence Condition"

    The modular inverse $a^{-1} \pmod{m}$ exists if and only if $\gcd(a, m) = 1$ (that is, $a$ and $m$ are coprime).

**Proof.** The equation $ax \equiv 1 \pmod{m}$ is equivalent to $ax + my = 1$ for some integer $y$. By Bezout's identity (see [Bezout's Identity](../divisibility/bezout.md)), this has a solution if and only if $\gcd(a, m) \mid 1$, which requires $\gcd(a, m) = 1$. $\square$

!!! warning "Uniqueness"

    When it exists, the modular inverse is unique modulo $m$. If both $x_1$ and $x_2$ satisfy $ax \equiv 1$, then $a(x_1 - x_2) \equiv 0 \pmod{m}$. Since $\gcd(a, m) = 1$, we get $x_1 \equiv x_2 \pmod{m}$.

## Method 1: Extended Euclidean Algorithm

The most general method computes the inverse using the extended Euclidean algorithm (see [Extended Euclidean](../divisibility/extended.md)). Given $\gcd(a, m) = 1$, the extended GCD returns $x, y$ with $ax + my = 1$. Reducing modulo $m$:

$$
ax \equiv 1 \pmod{m}
$$

So $a^{-1} \equiv x \pmod{m}$.

**Time complexity:** $O(\log m)$.

This method works for any modulus $m$, prime or composite.

## Method 2: Fermat's Little Theorem (Prime Modulus)

When $m = p$ is prime, Fermat's little theorem (see [Fermat's Little Theorem](fermat.md)) states $a^{p-1} \equiv 1 \pmod{p}$ for $\gcd(a, p) = 1$. Therefore:

$$
a^{-1} \equiv a^{p-2} \pmod{p}
$$

This is computed efficiently via modular exponentiation (see [Modular Exponentiation](exponentiation.md)) in $O(\log p)$ time.

!!! tip "Which Method to Use?"

    - **Prime modulus**: Fermat's method ($a^{p-2} \bmod p$) is simpler to implement
    - **Composite modulus**: use the extended Euclidean algorithm
    - **Performance**: both are $O(\log m)$; the extended GCD is typically faster by a constant factor since it avoids the overhead of modular exponentiation

## Worked Example

Finding $3^{-1} \pmod{7}$:

**Method 1 (Extended GCD):** Solve $3x + 7y = 1$. Running the extended Euclidean algorithm on $(3, 7)$:

- $7 = 2 \cdot 3 + 1$, so $1 = 7 - 2 \cdot 3$, giving $x = -2 \equiv 5 \pmod{7}$

**Method 2 (Fermat):** $3^{-1} \equiv 3^{7-2} = 3^5 \pmod{7}$. We compute $3^5 = 243 = 34 \cdot 7 + 5$, so $3^{-1} \equiv 5 \pmod{7}$.

**Verification:** $3 \cdot 5 = 15 = 2 \cdot 7 + 1 \equiv 1 \pmod{7}$. $\checkmark$

## Inverse Table for Small Moduli

For a prime $p$, every nonzero element has an inverse. The complete inverse table for $p = 7$:

| $a$ | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|
| $a^{-1} \pmod{7}$ | 1 | 4 | 5 | 2 | 3 | 6 |

Notice that the inverse function is a permutation of $\{1, 2, \ldots, p-1\}$.

## Implementation

```python
"""
Modular Inverse computation.

Provides two methods: extended Euclidean algorithm (works for any
coprime modulus) and Fermat's little theorem (prime modulus only).
"""


# === Extended GCD Method ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse_egcd(a: int, m: int) -> int:
    """Compute a^{-1} mod m using the extended Euclidean algorithm.

    Raises ValueError if gcd(a, m) != 1.
    """
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Inverse does not exist: gcd({a}, {m}) = {g}")
    return x % m


# === Fermat's Method (Prime Modulus) ===

def mod_inverse_fermat(a: int, p: int) -> int:
    """Compute a^{-1} mod p using Fermat's little theorem.

    Assumes p is prime and gcd(a, p) = 1.
    """
    return pow(a, p - 2, p)


# === Main ===

if __name__ == "__main__":
    # Compare both methods for prime modulus
    p = 7
    print(f"Modular inverses mod {p}:")
    for a in range(1, p):
        inv_egcd = mod_inverse_egcd(a, p)
        inv_fermat = mod_inverse_fermat(a, p)
        print(f"  {a}^(-1) = {inv_egcd} (EGCD), {inv_fermat} (Fermat), "
              f"verify: {a}*{inv_egcd} mod {p} = {(a * inv_egcd) % p}")

    # Composite modulus (only EGCD works)
    m = 12
    print(f"\nModular inverses mod {m} (EGCD only):")
    for a in range(1, m):
        try:
            inv = mod_inverse_egcd(a, m)
            print(f"  {a}^(-1) = {inv}, verify: {a}*{inv} mod {m} = {(a * inv) % m}")
        except ValueError:
            print(f"  {a}^(-1) does not exist (gcd({a},{m}) != 1)")
```

**Output:**

```
Modular inverses mod 7:
  1^(-1) = 1 (EGCD), 1 (Fermat), verify: 1*1 mod 7 = 1
  2^(-1) = 4 (EGCD), 4 (Fermat), verify: 2*4 mod 7 = 1
  3^(-1) = 5 (EGCD), 5 (Fermat), verify: 3*5 mod 7 = 1
  4^(-1) = 2 (EGCD), 2 (Fermat), verify: 4*2 mod 7 = 1
  5^(-1) = 3 (EGCD), 3 (Fermat), verify: 5*3 mod 7 = 1
  6^(-1) = 6 (EGCD), 6 (Fermat), verify: 6*6 mod 7 = 1

Modular inverses mod 12 (EGCD only):
  1^(-1) = 1, verify: 1*1 mod 12 = 1
  2^(-1) does not exist (gcd(2,12) != 1)
  3^(-1) does not exist (gcd(3,12) != 1)
  4^(-1) does not exist (gcd(4,12) != 1)
  5^(-1) = 5, verify: 5*5 mod 12 = 1
  6^(-1) does not exist (gcd(6,12) != 1)
  7^(-1) = 7, verify: 7*7 mod 12 = 1
  8^(-1) does not exist (gcd(8,12) != 1)
  9^(-1) does not exist (gcd(9,12) != 1)
  10^(-1) does not exist (gcd(10,12) != 1)
  11^(-1) = 11, verify: 11*11 mod 12 = 1
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
