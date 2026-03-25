# Euler's Totient

Fermat's little theorem tells us that $a^{p-1} \equiv 1 \pmod{p}$ when $p$ is prime. But what happens when the modulus is composite? Euler's totient function $\varphi(n)$ counts how many integers below $n$ are coprime to $n$, and Euler's theorem generalizes Fermat's result to arbitrary moduli: $a^{\varphi(n)} \equiv 1 \pmod{n}$. This generalization is central to RSA cryptography, where the modulus $n = pq$ is a product of two primes.

## Definition

**Euler's totient function** $\varphi(n)$ counts the number of integers in $\{1, 2, \ldots, n\}$ that are coprime to $n$:

$$
\varphi(n) = |\{k : 1 \le k \le n, \gcd(k, n) = 1\}|
$$

!!! example "Small Values"

    | $n$ | Integers coprime to $n$ | $\varphi(n)$ |
    |-----|------------------------|-------------|
    | 1   | $\{1\}$ | 1 |
    | 6   | $\{1, 5\}$ | 2 |
    | 8   | $\{1, 3, 5, 7\}$ | 4 |
    | 12  | $\{1, 5, 7, 11\}$ | 4 |

## Computing the Totient

### Prime Powers

For a prime $p$, every integer from $1$ to $p - 1$ is coprime to $p$:

$$
\varphi(p) = p - 1
$$

For a prime power $p^k$, the integers not coprime to $p^k$ are exactly the multiples of $p$, of which there are $p^{k-1}$:

$$
\varphi(p^k) = p^k - p^{k-1} = p^{k-1}(p - 1) = p^k\left(1 - \frac{1}{p}\right)
$$

### Multiplicativity

The totient function is **multiplicative**: if $\gcd(m, n) = 1$, then:

$$
\varphi(mn) = \varphi(m) \cdot \varphi(n)
$$

**Proof sketch.** By the Chinese remainder theorem (see [CRT](crt.md)), the map $k \mapsto (k \bmod m, k \bmod n)$ is a bijection from $\mathbb{Z}/mn\mathbb{Z}$ to $\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$. Under this bijection, $\gcd(k, mn) = 1$ if and only if $\gcd(k, m) = 1$ and $\gcd(k, n) = 1$. $\square$

### General Formula

Combining prime power and multiplicativity results, for $n = p_1^{a_1} p_2^{a_2} \cdots p_k^{a_k}$:

$$
\varphi(n) = n \prod_{p \mid n} \left(1 - \frac{1}{p}\right) = n \cdot \frac{p_1 - 1}{p_1} \cdot \frac{p_2 - 1}{p_2} \cdots \frac{p_k - 1}{p_k}
$$

!!! example "Computing Totient"

    $\varphi(12) = 12 \cdot (1 - 1/2) \cdot (1 - 1/3) = 12 \cdot 1/2 \cdot 2/3 = 4$.

    $\varphi(100) = 100 \cdot (1 - 1/2) \cdot (1 - 1/5) = 100 \cdot 1/2 \cdot 4/5 = 40$.

## Euler's Theorem

!!! info "Euler's Theorem"

    If $\gcd(a, n) = 1$, then

    $$
    a^{\varphi(n)} \equiv 1 \pmod{n}
    $$

**Proof.** Let $r_1, r_2, \ldots, r_{\varphi(n)}$ be the integers in $\{1, \ldots, n\}$ coprime to $n$. Since $\gcd(a, n) = 1$, the products $ar_1, ar_2, \ldots, ar_{\varphi(n)}$ are a permutation of $r_1, r_2, \ldots, r_{\varphi(n)}$ modulo $n$. Taking the product of both sides:

$$
a^{\varphi(n)} \cdot \prod_{i=1}^{\varphi(n)} r_i \equiv \prod_{i=1}^{\varphi(n)} r_i \pmod{n}
$$

Since each $r_i$ is coprime to $n$, their product is also coprime to $n$ and can be cancelled, yielding $a^{\varphi(n)} \equiv 1 \pmod{n}$. $\square$

When $n = p$ is prime, $\varphi(p) = p - 1$, recovering Fermat's little theorem (see [Fermat's Little Theorem](fermat.md)).

## The Gauss Divisor Sum

A fundamental identity connects the totient to divisor sums.

!!! info "Gauss's Formula"

    For any positive integer $n$,

    $$
    \sum_{d \mid n} \varphi(d) = n
    $$

**Proof.** Partition $\{1, 2, \ldots, n\}$ by $\gcd(k, n)$. For each divisor $d$ of $n$, the number of integers $k \in \{1, \ldots, n\}$ with $\gcd(k, n) = d$ equals $\varphi(n/d)$. Summing over all divisors gives $\sum_{d \mid n} \varphi(n/d) = n$, which equals $\sum_{d \mid n} \varphi(d)$ since the sum ranges over the same set of divisors. $\square$

## Application to RSA

In RSA, the modulus is $n = pq$ for distinct primes $p, q$. The totient is:

$$
\varphi(n) = (p-1)(q-1)
$$

The public key exponent $e$ is chosen coprime to $\varphi(n)$, and the private key is $d = e^{-1} \bmod \varphi(n)$. Euler's theorem guarantees that $m^{ed} \equiv m \pmod{n}$ for messages $m$ coprime to $n$, enabling decryption.

## Implementation

```python
"""
Euler's Totient Function and Euler's Theorem.

Computes the totient function using the prime factorization formula
and verifies Euler's theorem with numerical examples.
"""


# === Totient Function ===

def euler_totient(n: int) -> int:
    """Compute Euler's totient function phi(n).

    Uses the product formula by finding all prime factors.

    Args:
        n: A positive integer.

    Returns:
        phi(n), the count of integers in [1, n] coprime to n.
    """
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


# === Brute Force Verification ===

def euler_totient_brute(n: int) -> int:
    """Compute phi(n) by counting coprimes directly."""
    from math import gcd
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


# === Main ===

if __name__ == "__main__":
    # Compute totient for small values
    print("Euler's totient function:")
    for n in [1, 2, 6, 8, 10, 12, 100]:
        phi = euler_totient(n)
        phi_brute = euler_totient_brute(n)
        print(f"  phi({n}) = {phi}  (brute force: {phi_brute})")

    # Verify Euler's theorem
    print("\nEuler's theorem verification:")
    test_cases = [(3, 10), (7, 12), (11, 15)]
    for a, n in test_cases:
        phi = euler_totient(n)
        result = pow(a, phi, n)
        print(f"  {a}^phi({n}) mod {n} = {a}^{phi} mod {n} = {result}")

    # Gauss divisor sum
    print("\nGauss divisor sum:")
    for n in [6, 12, 20]:
        divisors = [d for d in range(1, n + 1) if n % d == 0]
        total = sum(euler_totient(d) for d in divisors)
        print(f"  sum(phi(d) for d | {n}) = {total} = {n}")
```

**Output:**

```
Euler's totient function:
  phi(1) = 1  (brute force: 1)
  phi(2) = 1  (brute force: 1)
  phi(6) = 2  (brute force: 2)
  phi(8) = 4  (brute force: 4)
  phi(10) = 4  (brute force: 4)
  phi(12) = 4  (brute force: 4)
  phi(100) = 40  (brute force: 40)

Euler's theorem verification:
  3^phi(10) mod 10 = 3^4 mod 10 = 1
  7^phi(12) mod 12 = 7^4 mod 12 = 1
  11^phi(15) mod 15 = 11^8 mod 15 = 1

Gauss divisor sum:
  sum(phi(d) for d | 6) = 6 = 6
  sum(phi(d) for d | 12) = 12 = 12
  sum(phi(d) for d | 20) = 20 = 20
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
