# Modular Exponentiation

Computing $a^n \bmod m$ arises in cryptography (RSA encryption and decryption), primality testing (Miller-Rabin), and competitive programming. A naive approach multiplying $a$ by itself $n$ times requires $O(n)$ multiplications, which is infeasible when $n$ has hundreds of digits. The **repeated squaring** (binary exponentiation) technique reduces this to $O(\log n)$ multiplications by exploiting the binary representation of the exponent.

## The Repeated Squaring Idea

The key observation is that squaring halves the exponent. If $n$ is even:

$$
a^n = (a^{n/2})^2
$$

If $n$ is odd:

$$
a^n = a \cdot a^{n-1} = a \cdot (a^{(n-1)/2})^2
$$

This gives a divide-and-conquer strategy: express $n$ in binary and process its bits from most significant to least significant (or equivalently, from least significant to most significant).

## Algorithm

### Right-to-Left Binary Method

Process the bits of $n$ from least significant to most significant:

```
MODULAR-EXPONENTIATION(a, n, m):
    result = 1
    a = a mod m
    while n > 0:
        if n is odd:
            result = (result * a) mod m
        n = n >> 1            // right shift = floor division by 2
        a = (a * a) mod m     // square the base
    return result
```

### Worked Example

Computing $3^{13} \bmod 7$. The binary representation of $13$ is $1101_2$.

| Step | Bit | $n$ | $a$ | result |
|------|-----|-----|-----|--------|
| Init | --  | 13  | 3   | 1      |
| 1    | 1   | 6   | $3^2 = 2$ | $1 \cdot 3 = 3$ |
| 2    | 0   | 3   | $2^2 = 4$ | 3 |
| 3    | 1   | 1   | $4^2 = 2$ | $3 \cdot 4 = 5$ |
| 4    | 1   | 0   | $2^2 = 4$ | $5 \cdot 2 = 3$ |

All values are taken modulo 7. Result: $3^{13} \equiv 3 \pmod{7}$.

Verification: $3^6 = 729 = 104 \cdot 7 + 1$, so $3^6 \equiv 1 \pmod{7}$. Then $3^{13} = 3^{12} \cdot 3 = (3^6)^2 \cdot 3 \equiv 1 \cdot 3 = 3 \pmod{7}$. $\checkmark$

## Correctness

!!! info "Loop Invariant"

    At the start of each iteration, let $n_0$ be the original exponent. The invariant is:

    $$
    \text{result} \cdot a^n \equiv a_0^{n_0} \pmod{m}
    $$

    where $a_0$ is the original base.

**Proof.** Initially, $\text{result} = 1$ and $a^n = a_0^{n_0}$, so the invariant holds.

In each iteration:

- If $n$ is odd: $\text{result}' = \text{result} \cdot a$, $n' = (n-1)/2$, $a' = a^2$. Then $\text{result}' \cdot (a')^{n'} = \text{result} \cdot a \cdot a^{2 \cdot (n-1)/2} = \text{result} \cdot a^n$.
- If $n$ is even: $\text{result}' = \text{result}$, $n' = n/2$, $a' = a^2$. Then $\text{result}' \cdot (a')^{n'} = \text{result} \cdot a^{2 \cdot n/2} = \text{result} \cdot a^n$.

When $n = 0$, $\text{result} \cdot a^0 = \text{result} = a_0^{n_0} \bmod m$. $\square$

## Complexity

Each iteration halves $n$ (via right shift), so the loop executes $\lfloor \log_2 n \rfloor + 1$ times. Each iteration performs at most two modular multiplications. Therefore:

$$
O(\log n) \text{ modular multiplications}
$$

If each multiplication of numbers modulo $m$ costs $O((\log m)^2)$ with schoolbook multiplication (or $O(\log m \cdot \log \log m)$ with FFT-based methods), the total cost is:

$$
O(\log n \cdot (\log m)^2)
$$

!!! note "Comparison with Naive Method"

    The naive approach requires $O(n)$ multiplications. For $n = 2^{1000}$ (a typical RSA exponent size), this is astronomically infeasible. Repeated squaring needs only about 1000 multiplications.

## Recursive Version

```
MODULAR-EXPONENTIATION-RECURSIVE(a, n, m):
    if n = 0:
        return 1
    if n is odd:
        return (a * MODULAR-EXPONENTIATION-RECURSIVE(a, n-1, m)) mod m
    half = MODULAR-EXPONENTIATION-RECURSIVE(a, n/2, m)
    return (half * half) mod m
```

## Implementation

```python
"""
Modular Exponentiation via repeated squaring.

Computes a^n mod m in O(log n) multiplications using both
iterative and recursive approaches.
"""


# === Iterative Modular Exponentiation ===

def mod_pow(base: int, exp: int, mod: int) -> int:
    """Compute base^exp mod m using right-to-left binary method.

    Args:
        base: The base integer.
        exp: The nonnegative exponent.
        mod: The positive modulus.

    Returns:
        base^exp mod m.
    """
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


# === Recursive Modular Exponentiation ===

def mod_pow_recursive(base: int, exp: int, mod: int) -> int:
    """Compute base^exp mod m recursively."""
    if exp == 0:
        return 1
    if exp % 2 == 1:
        return (base * mod_pow_recursive(base, exp - 1, mod)) % mod
    half = mod_pow_recursive(base, exp // 2, mod)
    return (half * half) % mod


# === Main ===

if __name__ == "__main__":
    # Basic examples
    print(f"3^13 mod 7 = {mod_pow(3, 13, 7)}")
    print(f"2^10 mod 1000 = {mod_pow(2, 10, 1000)}")
    print(f"7^256 mod 13 = {mod_pow(7, 256, 13)}")

    # Large exponent (RSA-like)
    print(f"2^1000 mod 1000000007 = {mod_pow(2, 1000, 10**9 + 7)}")

    # Verify against Python built-in
    print(f"\nVerification against pow():")
    test_cases = [(3, 13, 7), (2, 10, 1000), (7, 256, 13), (123, 456, 789)]
    for b, e, m in test_cases:
        ours = mod_pow(b, e, m)
        builtin = pow(b, e, m)
        print(f"  {b}^{e} mod {m}: ours={ours}, pow={builtin}, match={ours == builtin}")
```

**Output:**

```
3^13 mod 7 = 3
2^10 mod 1000 = 24
7^256 mod 13 = 9
2^1000 mod 1000000007 = 688423210

Verification against pow():
  3^13 mod 7: ours=3, pow=3, match=True
  2^10 mod 1000: ours=24, pow=24, match=True
  7^256 mod 13: ours=9, pow=9, match=True
  123^456 mod 789: ours=699, pow=699, match=True
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
