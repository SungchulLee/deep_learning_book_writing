# Fermat's Little Theorem

When working modulo a prime $p$, exponentiation exhibits a remarkable periodicity: raising any nonzero integer to the power $p - 1$ always yields $1$. Fermat's little theorem captures this property and serves as the foundation for primality testing, efficient modular inverse computation, and the structure theory of finite fields.

## Theorem Statement

!!! info "Fermat's Little Theorem"

    If $p$ is prime and $\gcd(a, p) = 1$ (equivalently, $p \nmid a$), then

    $$
    a^{p-1} \equiv 1 \pmod{p}
    $$

An equivalent formulation that holds for all integers $a$ (including multiples of $p$) is:

$$
a^p \equiv a \pmod{p}
$$

## Proof

We present two proofs: one combinatorial and one algebraic.

### Proof 1: Counting Necklaces

Consider the $a^p$ strings of length $p$ over an alphabet of size $a$. Group strings by cyclic equivalence: two strings are equivalent if one is a cyclic rotation of the other. Each equivalence class has exactly $p$ elements (since $p$ is prime, no string of length $p$ has a period smaller than $p$ unless all characters are identical). There are $a$ constant strings. Therefore:

$$
a^p = a + p \cdot k
$$

for some nonnegative integer $k$, which gives $a^p \equiv a \pmod{p}$. $\square$

### Proof 2: Multiplicative Group

Consider the set $\{1, 2, \ldots, p-1\}$ under multiplication modulo $p$. Since $p$ is prime, this forms a group of order $p - 1$. For any element $a$ with $\gcd(a, p) = 1$, the map $x \mapsto ax \bmod p$ is a bijection on $\{1, 2, \ldots, p-1\}$. Therefore:

$$
\prod_{i=1}^{p-1} (a \cdot i) \equiv \prod_{i=1}^{p-1} i \pmod{p}
$$

The left side equals $a^{p-1} \cdot (p-1)!$ and the right side is $(p-1)!$. Since $\gcd((p-1)!, p) = 1$, we can cancel $(p-1)!$ to obtain $a^{p-1} \equiv 1 \pmod{p}$. $\square$

## Worked Examples

**Example 1.** Verify $2^{6} \equiv 1 \pmod{7}$:

$$
2^6 = 64 = 9 \cdot 7 + 1 \equiv 1 \pmod{7} \quad \checkmark
$$

**Example 2.** Compute $3^{100} \bmod 7$.

Since $3^6 \equiv 1 \pmod{7}$ by Fermat's theorem, write $100 = 6 \cdot 16 + 4$:

$$
3^{100} = (3^6)^{16} \cdot 3^4 \equiv 1^{16} \cdot 81 \equiv 81 \bmod 7 = 4 \pmod{7}
$$

## The Fermat Primality Test

!!! warning "Converse is False"

    The converse of Fermat's little theorem is **not** true. There exist composite numbers $n$ for which $a^{n-1} \equiv 1 \pmod{n}$ for some (or even all) bases $a$ coprime to $n$.

A **Carmichael number** is a composite $n$ satisfying $a^{n-1} \equiv 1 \pmod{n}$ for all $a$ with $\gcd(a, n) = 1$. The smallest Carmichael number is $561 = 3 \cdot 11 \cdot 17$.

Despite this limitation, the Fermat test provides a useful probabilistic filter:

1. Pick a random base $a \in \{2, 3, \ldots, n-2\}$
2. Compute $a^{n-1} \bmod n$ using fast exponentiation (see [Modular Exponentiation](exponentiation.md))
3. If $a^{n-1} \not\equiv 1 \pmod{n}$, then $n$ is **definitely composite** ($a$ is a *Fermat witness*)
4. If $a^{n-1} \equiv 1 \pmod{n}$, then $n$ is **probably prime**

The Miller-Rabin test (see [Miller-Rabin Test](../primality/miller_rabin.md)) strengthens this by detecting more witnesses.

## Application: Modular Inverse

For prime $p$ and $\gcd(a, p) = 1$, Fermat's theorem immediately yields:

$$
a^{-1} \equiv a^{p-2} \pmod{p}
$$

This follows from $a \cdot a^{p-2} = a^{p-1} \equiv 1 \pmod{p}$. See [Modular Inverse](inverse.md) for details.

## Connection to Euler's Theorem

Fermat's little theorem is a special case of Euler's theorem (see [Euler's Totient](totient.md)). For any $m$ with $\gcd(a, m) = 1$:

$$
a^{\varphi(m)} \equiv 1 \pmod{m}
$$

When $m = p$ is prime, $\varphi(p) = p - 1$, recovering Fermat's result.

## Implementation

```python
"""
Fermat's Little Theorem: verification and primality testing.

Demonstrates the theorem with numerical examples and implements
the Fermat primality test.
"""


# === Fermat's Theorem Verification ===

def verify_fermat(a: int, p: int) -> bool:
    """Verify a^(p-1) = 1 (mod p) for prime p."""
    return pow(a, p - 1, p) == 1


# === Fermat Primality Test ===

def fermat_test(n: int, k: int = 10) -> str:
    """Probabilistic primality test using Fermat's little theorem.

    Args:
        n: The integer to test.
        k: Number of random bases to try.

    Returns:
        'composite' if a witness is found, 'probably prime' otherwise.
    """
    if n < 2:
        return "composite"
    if n <= 3:
        return "probably prime"

    import random
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow(a, n - 1, n) != 1:
            return "composite"
    return "probably prime"


# === Main ===

if __name__ == "__main__":
    # Verify theorem for p = 7
    p = 7
    print(f"Fermat's theorem verification for p = {p}:")
    for a in range(1, p):
        result = pow(a, p - 1, p)
        print(f"  {a}^{p-1} mod {p} = {result}")

    # Compute 3^100 mod 7
    print(f"\n3^100 mod 7 = {pow(3, 100, 7)}")

    # Primality testing
    print("\nFermat primality test:")
    test_numbers = [7, 13, 15, 561, 1009, 1729]
    for n in test_numbers:
        result = fermat_test(n, k=20)
        print(f"  {n}: {result}")

    # Show Carmichael number 561
    print(f"\nCarmichael number 561 = 3 * 11 * 17:")
    all_pass = all(pow(a, 560, 561) == 1
                   for a in range(2, 561) if pow(a, 1, 561) != 0)
    print(f"  All coprime bases pass: {all_pass}")
```

**Output:**

```
Fermat's theorem verification for p = 7:
  1^6 mod 7 = 1
  2^6 mod 7 = 1
  3^6 mod 7 = 1
  4^6 mod 7 = 1
  5^6 mod 7 = 1
  6^6 mod 7 = 1

3^100 mod 7 = 4

Fermat primality test:
  7: probably prime
  13: probably prime
  15: composite
  561: probably prime
  1009: probably prime
  1729: probably prime

Carmichael number 561 = 3 * 11 * 17:
  All coprime bases pass: True
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
