# Greatest Common Divisor

Many problems in number theory, cryptography, and algorithm design require finding the largest factor shared by two integers. Simplifying fractions, computing modular inverses, and testing coprimality all reduce to computing the greatest common divisor. Euclid's algorithm solves this problem efficiently using a simple observation: the GCD of two numbers does not change when the larger number is replaced by its remainder upon division by the smaller.

## Definition

For integers $a$ and $b$, not both zero, the **greatest common divisor** $\gcd(a, b)$ is the largest positive integer $d$ such that $d \mid a$ and $d \mid b$.

Two basic properties follow immediately from the definition:

- $\gcd(a, 0) = |a|$ for any nonzero integer $a$, since every integer divides $0$
- $\gcd(a, b) = \gcd(b, a)$ (commutativity)

When $\gcd(a, b) = 1$, we say $a$ and $b$ are **coprime** (or **relatively prime**).

## The Division Algorithm Foundation

The key insight behind Euclid's algorithm is the following lemma.

!!! info "GCD Reduction Lemma"

    For any integers $a$ and $b$ with $b > 0$,

    $$
    \gcd(a, b) = \gcd(b, a \bmod b)
    $$

**Proof.** Let $d = \gcd(a, b)$. Write $a = qb + r$ where $r = a \bmod b$ and $0 \le r < b$. Since $d \mid a$ and $d \mid b$, we have $d \mid (a - qb) = r$, so $d$ divides both $b$ and $r$. Conversely, any common divisor of $b$ and $r$ also divides $qb + r = a$, so it divides both $a$ and $b$. Therefore, the set of common divisors of $(a, b)$ equals the set of common divisors of $(b, r)$, and their greatest elements coincide. $\square$

## Euclid's Algorithm

The reduction lemma suggests a recursive strategy: repeatedly replace $(a, b)$ with $(b, a \bmod b)$ until the remainder reaches zero. At that point, the nonzero element is the GCD.

### Iterative Version

```
EUCLID(a, b):
    while b != 0:
        a, b = b, a mod b
    return a
```

### Worked Example

Computing $\gcd(48, 18)$ step by step:

| Step | $a$ | $b$ | $a \bmod b$ |
|------|-----|-----|-------------|
| 1    | 48  | 18  | 12          |
| 2    | 18  | 12  | 6           |
| 3    | 12  | 6   | 0           |

When $b = 0$, we return $a = 6$. Therefore $\gcd(48, 18) = 6$.

## Correctness

**Termination.** At each step, the second argument $b$ is replaced by $a \bmod b$, which satisfies $0 \le a \bmod b < b$. Since $b$ is a strictly decreasing sequence of nonnegative integers, the algorithm terminates.

**Partial correctness.** By the GCD Reduction Lemma, each replacement preserves the GCD. When the algorithm terminates with $b = 0$, we have $\gcd(a, 0) = a$, which is the correct answer.

## Complexity Analysis

!!! tip "Time Complexity of Euclid's Algorithm"

    Euclid's algorithm computes $\gcd(a, b)$ in $O(\log(\min(a, b)))$ division steps.

The key observation is that the remainder decreases by at least a factor of two every two steps. Specifically, for $b > 0$:

$$
a \bmod b < \frac{a}{2}
$$

This holds because if $b \le a/2$, then $a \bmod b < b \le a/2$, and if $b > a/2$, then $a \bmod b = a - b < a/2$.

After two iterations, the value of $b$ is replaced by a number less than half its original value. Therefore, the algorithm performs at most $2\lfloor \log_2(\min(a, b)) \rfloor + 2$ division steps, giving $O(\log(\min(a, b)))$ time complexity.

!!! note "Connection to Fibonacci Numbers"

    The worst-case inputs for Euclid's algorithm are consecutive Fibonacci numbers. Computing $\gcd(F_{n+1}, F_n)$ requires exactly $n$ division steps, confirming the $\Theta(\log(\min(a, b)))$ bound since $F_n = \Theta(\varphi^n)$ where $\varphi = (1 + \sqrt{5})/2$ is the golden ratio.

## Implementation

```python
"""
Greatest Common Divisor via Euclid's Algorithm.

Demonstrates the iterative Euclidean algorithm for computing GCD,
including correctness verification on several test cases.
"""


# === Euclidean Algorithm ===

def gcd(a: int, b: int) -> int:
    """Compute gcd(a, b) using the iterative Euclidean algorithm.

    Args:
        a: First nonnegative integer.
        b: Second nonnegative integer.

    Returns:
        The greatest common divisor of a and b.
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


# === Recursive Version ===

def gcd_recursive(a: int, b: int) -> int:
    """Compute gcd(a, b) using the recursive Euclidean algorithm."""
    a, b = abs(a), abs(b)
    if b == 0:
        return a
    return gcd_recursive(b, a % b)


# === Main ===

if __name__ == "__main__":
    # Basic examples
    print(f"gcd(48, 18) = {gcd(48, 18)}")
    print(f"gcd(270, 192) = {gcd(270, 192)}")
    print(f"gcd(17, 13) = {gcd(17, 13)}")

    # Edge cases
    print(f"gcd(0, 5) = {gcd(0, 5)}")
    print(f"gcd(7, 0) = {gcd(7, 0)}")

    # Verify iterative matches recursive
    print(f"gcd_recursive(48, 18) = {gcd_recursive(48, 18)}")
```

**Output:**

```
gcd(48, 18) = 6
gcd(270, 192) = 6
gcd(17, 13) = 1
gcd(0, 5) = 5
gcd(7, 0) = 7
gcd_recursive(48, 18) = 6
```

## Applications

The GCD computation appears throughout mathematics and computer science:

- **Fraction simplification**: reduce $a/b$ by dividing both by $\gcd(a, b)$
- **Modular inverse**: finding $a^{-1} \pmod{m}$ requires $\gcd(a, m) = 1$ (see [Modular Inverse](../modular/inverse.md))
- **Bezout's identity**: expressing $\gcd(a, b) = ax + by$ via the extended Euclidean algorithm (see [Bezout's Identity](bezout.md))
- **RSA cryptography**: key generation requires coprimality checks
- **LCM computation**: $\operatorname{lcm}(a, b) = |ab| / \gcd(a, b)$ (see [LCM](lcm.md))

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
