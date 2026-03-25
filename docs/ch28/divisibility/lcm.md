# Least Common Multiple

Adding fractions with different denominators, synchronizing periodic events, and merging cyclic schedules all require finding a common multiple of two or more integers. The least common multiple (LCM) is the smallest such shared multiple, and it connects directly to the GCD through a simple but powerful identity that enables efficient computation.

## Definition

For positive integers $a$ and $b$, the **least common multiple** $\operatorname{lcm}(a, b)$ is the smallest positive integer $m$ such that $a \mid m$ and $b \mid m$.

Equivalently, $\operatorname{lcm}(a, b)$ is the smallest positive integer that appears in both the set of multiples of $a$ and the set of multiples of $b$.

!!! example "LCM by Listing Multiples"

    Multiples of 12: 12, 24, 36, 48, 60, 72, ...

    Multiples of 18: 18, 36, 54, 72, 90, ...

    Common multiples: 36, 72, 108, ...

    Therefore $\operatorname{lcm}(12, 18) = 36$.

## LCM-GCD Relationship

Listing multiples is impractical for large numbers. The following theorem provides an efficient alternative by reducing LCM computation to a single GCD call.

!!! info "LCM-GCD Identity"

    For any positive integers $a$ and $b$,

    $$
    \operatorname{lcm}(a, b) = \frac{a \cdot b}{\gcd(a, b)}
    $$

**Proof.** Write $a = \gcd(a, b) \cdot a'$ and $b = \gcd(a, b) \cdot b'$ where $\gcd(a', b') = 1$. Any common multiple of $a$ and $b$ must be divisible by $\gcd(a, b) \cdot a' \cdot b'$, since $a'$ and $b'$ are coprime and both must divide the multiple. Therefore the smallest common multiple is:

$$
\operatorname{lcm}(a, b) = \gcd(a, b) \cdot a' \cdot b' = \frac{a \cdot b}{\gcd(a, b)}
$$

$\square$

### Prime Factorization View

If $a = p_1^{\alpha_1} p_2^{\alpha_2} \cdots p_k^{\alpha_k}$ and $b = p_1^{\beta_1} p_2^{\beta_2} \cdots p_k^{\beta_k}$ (allowing zero exponents), then:

$$
\gcd(a, b) = \prod_{i=1}^{k} p_i^{\min(\alpha_i, \beta_i)}
$$

$$
\operatorname{lcm}(a, b) = \prod_{i=1}^{k} p_i^{\max(\alpha_i, \beta_i)}
$$

This makes the identity $\gcd(a, b) \cdot \operatorname{lcm}(a, b) = a \cdot b$ transparent: for each prime $p_i$, $\min(\alpha_i, \beta_i) + \max(\alpha_i, \beta_i) = \alpha_i + \beta_i$.

## Complexity

Since the GCD can be computed in $O(\log(\min(a, b)))$ time using Euclid's algorithm (see [GCD](gcd.md)), the LCM computation requires only one GCD call plus a single multiplication and division. The overall time complexity is:

$$
O(\log(\min(a, b)))
$$

!!! warning "Overflow Prevention"

    When computing $\operatorname{lcm}(a, b) = a \cdot b / \gcd(a, b)$, the intermediate product $a \cdot b$ may overflow. A safer computation divides first: $\operatorname{lcm}(a, b) = (a / \gcd(a, b)) \cdot b$. This is exact because $\gcd(a, b)$ divides $a$.

## LCM of Multiple Integers

The LCM extends to more than two integers via the associative property:

$$
\operatorname{lcm}(a_1, a_2, \ldots, a_n) = \operatorname{lcm}(\operatorname{lcm}(a_1, a_2), a_3, \ldots, a_n)
$$

This allows iterative computation by folding the LCM function across a list.

## Implementation

```python
"""
Least Common Multiple via the GCD identity.

Demonstrates LCM computation for two integers and for a list of integers,
using the identity lcm(a, b) = a * b / gcd(a, b).
"""

import math
from functools import reduce


# === LCM of Two Integers ===

def lcm(a: int, b: int) -> int:
    """Compute lcm(a, b) using the GCD identity.

    Uses the overflow-safe form: (a // gcd) * b.

    Args:
        a: First positive integer.
        b: Second positive integer.

    Returns:
        The least common multiple of a and b.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a) // math.gcd(abs(a), abs(b)) * abs(b)


# === LCM of Multiple Integers ===

def lcm_list(numbers: list[int]) -> int:
    """Compute the LCM of a list of positive integers."""
    return reduce(lcm, numbers)


# === Main ===

if __name__ == "__main__":
    # Basic examples
    print(f"lcm(12, 18) = {lcm(12, 18)}")
    print(f"lcm(4, 6) = {lcm(4, 6)}")
    print(f"lcm(7, 13) = {lcm(7, 13)}")

    # Edge cases
    print(f"lcm(5, 0) = {lcm(5, 0)}")
    print(f"lcm(1, 100) = {lcm(1, 100)}")

    # Multiple integers
    print(f"lcm(2, 3, 4, 5) = {lcm_list([2, 3, 4, 5])}")
    print(f"lcm(6, 10, 15) = {lcm_list([6, 10, 15])}")
```

**Output:**

```
lcm(12, 18) = 36
lcm(4, 6) = 12
lcm(7, 13) = 91
lcm(5, 0) = 0
lcm(1, 100) = 100
lcm(2, 3, 4, 5) = 60
lcm(6, 10, 15) = 30
```

## Applications

- **Fraction arithmetic**: the LCM of denominators gives the least common denominator
- **Scheduling**: determining when periodic events with different periods will next coincide
- **Competitive programming**: many problems involving cycles or periodicity reduce to LCM
- **Cryptography**: Carmichael's function $\lambda(n)$, used in RSA, is defined via LCM of prime power totients

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
