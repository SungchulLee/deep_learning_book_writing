# Extended Euclidean Algorithm

The standard Euclidean algorithm computes $\gcd(a, b)$ but discards information along the way. The **extended Euclidean algorithm** augments each step to also track the Bezout coefficients $x$ and $y$ satisfying $ax + by = \gcd(a, b)$. This makes it the core subroutine for computing modular inverses, solving linear Diophantine equations, and implementing RSA key generation.

## From Euclid to Extended Euclid

Recall that Euclid's algorithm (see [GCD](gcd.md)) applies the recurrence $\gcd(a, b) = \gcd(b, a \bmod b)$ until the remainder is zero. The extended version maintains auxiliary variables that express the current remainder as a linear combination of the original inputs $a$ and $b$ at every step.

### The Key Recurrence

Suppose that at a recursive level we have computed $\gcd(b, a \bmod b) = g$ with coefficients $x_1, y_1$ such that:

$$
b \cdot x_1 + (a \bmod b) \cdot y_1 = g
$$

Since $a \bmod b = a - \lfloor a/b \rfloor \cdot b$, we substitute:

$$
b \cdot x_1 + (a - \lfloor a/b \rfloor \cdot b) \cdot y_1 = g
$$

Rearranging:

$$
a \cdot y_1 + b \cdot (x_1 - \lfloor a/b \rfloor \cdot y_1) = g
$$

Comparing with $ax + by = g$, the coefficients at the current level are:

$$
x = y_1, \quad y = x_1 - \lfloor a/b \rfloor \cdot y_1
$$

### Base Case

When $b = 0$, we have $\gcd(a, 0) = a$, with the trivial representation $a \cdot 1 + 0 \cdot 0 = a$. So the base case returns $(g, x, y) = (a, 1, 0)$.

## Algorithm

```
EXTENDED-EUCLID(a, b):
    if b = 0:
        return (a, 1, 0)
    (g, x₁, y₁) = EXTENDED-EUCLID(b, a mod b)
    x = y₁
    y = x₁ - ⌊a/b⌋ · y₁
    return (g, x, y)
```

### Iterative Version

The recursive version can be converted to an iterative form that avoids stack overhead. We maintain two pairs of coefficients $(x_{\text{old}}, y_{\text{old}})$ and $(x_{\text{new}}, y_{\text{new}})$ and update them at each step.

```
EXTENDED-EUCLID-ITERATIVE(a, b):
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r ≠ 0:
        q = ⌊old_r / r⌋
        old_r, r = r, old_r - q · r
        old_x, x = x, old_x - q · x
        old_y, y = y, old_y - q · y
    return (old_r, old_x, old_y)
```

## Worked Example

Computing the extended GCD of $a = 48$ and $b = 18$:

| Step | $a$ | $b$ | $q$ | $x$ | $y$ |
|------|-----|-----|-----|-----|-----|
| Init | 48  | 18  | --  | 1, 0 | 0, 1 |
| 1    | 18  | 12  | 2   | 0, 1 | 1, -2 |
| 2    | 12  | 6   | 1   | 1, -1 | -2, 3 |
| 3    | 6   | 0   | 2   | -1 | 3 |

Result: $\gcd(48, 18) = 6$ with $x = -1$ and $y = 3$.

Verification: $48 \cdot (-1) + 18 \cdot 3 = -48 + 54 = 6$. $\checkmark$

## Correctness

!!! info "Correctness of Extended Euclidean Algorithm"

    The algorithm returns $(g, x, y)$ such that $g = \gcd(a, b)$ and $ax + by = g$.

**Proof by induction on the number of recursive calls.**

*Base case.* When $b = 0$: returns $(a, 1, 0)$. Indeed $a \cdot 1 + 0 \cdot 0 = a = \gcd(a, 0)$. $\checkmark$

*Inductive step.* Assume the recursive call returns correct $(g, x_1, y_1)$ with $b \cdot x_1 + (a \bmod b) \cdot y_1 = g = \gcd(b, a \bmod b) = \gcd(a, b)$. The derivation above shows that setting $x = y_1$ and $y = x_1 - \lfloor a/b \rfloor \cdot y_1$ gives $ax + by = g$. $\square$

## Complexity

The extended Euclidean algorithm performs the same number of division steps as the standard Euclidean algorithm, with constant additional work per step for maintaining the coefficients. Therefore, its time complexity is:

$$
O(\log(\min(a, b)))
$$

The space complexity is $O(\log(\min(a, b)))$ for the recursive version (due to the call stack) and $O(1)$ for the iterative version.

## Implementation

```python
"""
Extended Euclidean Algorithm.

Computes gcd(a, b) along with Bezout coefficients x, y such that
a*x + b*y = gcd(a, b). Provides both recursive and iterative versions.
"""


# === Recursive Extended GCD ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Compute (g, x, y) such that a*x + b*y = g = gcd(a, b).

    Recursive implementation following the standard derivation.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Tuple (g, x, y) where g = gcd(a, b) and a*x + b*y = g.
    """
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


# === Iterative Extended GCD ===

def extended_gcd_iterative(a: int, b: int) -> tuple[int, int, int]:
    """Compute (g, x, y) such that a*x + b*y = g = gcd(a, b).

    Iterative implementation using O(1) extra space.
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


# === Main ===

if __name__ == "__main__":
    # Recursive version
    test_cases = [(48, 18), (270, 192), (35, 15), (17, 13), (100, 0)]
    print("Recursive extended GCD:")
    for a, b in test_cases:
        g, x, y = extended_gcd(a, b)
        print(f"  gcd({a}, {b}) = {g},  {a}*({x}) + {b}*({y}) = {a*x + b*y}")

    # Iterative version
    print("\nIterative extended GCD:")
    for a, b in test_cases:
        g, x, y = extended_gcd_iterative(a, b)
        print(f"  gcd({a}, {b}) = {g},  {a}*({x}) + {b}*({y}) = {a*x + b*y}")
```

**Output:**

```
Recursive extended GCD:
  gcd(48, 18) = 6,  48*(-1) + 18*(3) = 6
  gcd(270, 192) = 6,  270*(-5) + 192*(7) = 6
  gcd(35, 15) = 5,  35*(1) + 15*(-2) = 5
  gcd(17, 13) = 1,  17*(-3) + 13*(4) = 1
  gcd(100, 0) = 100,  100*(1) + 0*(0) = 100

Iterative extended GCD:
  gcd(48, 18) = 6,  48*(-1) + 18*(3) = 6
  gcd(270, 192) = 6,  270*(-5) + 192*(7) = 6
  gcd(35, 15) = 5,  35*(1) + 15*(-2) = 5
  gcd(17, 13) = 1,  17*(-3) + 13*(4) = 1
  gcd(100, 0) = 100,  100*(1) + 0*(0) = 100
```

## Applications

- **Modular inverse**: computing $a^{-1} \pmod{m}$ when $\gcd(a, m) = 1$ (see [Modular Inverse](../modular/inverse.md))
- **Linear Diophantine equations**: finding integer solutions to $ax + by = c$
- **RSA key generation**: computing the private key $d \equiv e^{-1} \pmod{\lambda(n)}$
- **Continued fractions**: the quotients in the extended GCD correspond to partial quotients

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
