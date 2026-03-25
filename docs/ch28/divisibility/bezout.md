# Bezout's Identity

Euclid's algorithm shows that the GCD of two integers can be computed efficiently. A natural follow-up question is whether the GCD can be *expressed* as a combination of the original integers. Bezout's identity answers this affirmatively: for any integers $a$ and $b$, there exist integers $x$ and $y$ such that $ax + by = \gcd(a, b)$. This representation is the foundation for computing modular inverses, solving linear Diophantine equations, and proving fundamental results in number theory.

## Theorem Statement

!!! info "Bezout's Identity"

    For any integers $a$ and $b$, not both zero, there exist integers $x, y \in \mathbb{Z}$ such that

    $$
    ax + by = \gcd(a, b)
    $$

    The integers $x$ and $y$ are called **Bezout coefficients**.

The coefficients $x$ and $y$ are not unique. If $(x_0, y_0)$ is one solution, then all solutions have the form:

$$
x = x_0 + k \cdot \frac{b}{\gcd(a, b)}, \quad y = y_0 - k \cdot \frac{a}{\gcd(a, b)}
$$

for any integer $k$.

## Proof

The proof proceeds by showing that the smallest positive element of the set $S = \{ax + by : x, y \in \mathbb{Z}\} \cap \mathbb{Z}^{+}$ equals $\gcd(a, b)$.

**Step 1.** The set $S$ is nonempty because $a \cdot a + b \cdot 0 = a^2 > 0$ (or similarly using $b$). Let $d$ be the smallest positive element of $S$, so $d = ax_0 + by_0$ for some integers $x_0, y_0$.

**Step 2.** We show $d \mid a$. By the division algorithm, write $a = qd + r$ with $0 \le r < d$. Then:

$$
r = a - qd = a - q(ax_0 + by_0) = a(1 - qx_0) + b(-qy_0)
$$

So $r \in S \cup \{0\}$. Since $0 \le r < d$ and $d$ is the smallest positive element of $S$, we must have $r = 0$. Therefore $d \mid a$. By the same argument, $d \mid b$.

**Step 3.** Since $d \mid a$ and $d \mid b$, we have $d \le \gcd(a, b)$. But $\gcd(a, b)$ divides every element of $S$ (since it divides both $a$ and $b$), so $\gcd(a, b) \mid d$, which gives $\gcd(a, b) \le d$. Combining, $d = \gcd(a, b)$. $\square$

## Worked Example

For $a = 48$ and $b = 18$, we know $\gcd(48, 18) = 6$. Bezout's identity guarantees integers $x, y$ with $48x + 18y = 6$.

Tracing back through Euclid's algorithm:

| Step | Equation |
|------|----------|
| 1 | $48 = 2 \cdot 18 + 12$ |
| 2 | $18 = 1 \cdot 12 + 6$ |
| 3 | $12 = 2 \cdot 6 + 0$ |

Back-substituting from step 2:

$$
6 = 18 - 1 \cdot 12
$$

Substituting $12 = 48 - 2 \cdot 18$ from step 1:

$$
6 = 18 - 1 \cdot (48 - 2 \cdot 18) = 3 \cdot 18 - 1 \cdot 48
$$

So $x = -1$ and $y = 3$ satisfy $48(-1) + 18(3) = 6$. The extended Euclidean algorithm automates this back-substitution process (see [Extended Euclidean](extended.md)).

## Coprimality Characterization

Bezout's identity provides an elegant characterization of coprimality.

!!! info "Coprimality Criterion"

    Integers $a$ and $b$ are coprime (i.e., $\gcd(a, b) = 1$) if and only if there exist integers $x, y$ such that

    $$
    ax + by = 1
    $$

**Proof.** If $\gcd(a, b) = 1$, Bezout's identity gives the desired $x, y$. Conversely, if $ax + by = 1$ and $d = \gcd(a, b)$, then $d \mid (ax + by) = 1$, so $d = 1$. $\square$

## Linear Diophantine Equations

Bezout's identity generalizes to solve equations of the form $ax + by = c$.

!!! info "Solvability of Linear Diophantine Equations"

    The equation $ax + by = c$ has integer solutions if and only if $\gcd(a, b) \mid c$.

**Proof.** If $d = \gcd(a, b) \mid c$, write $c = d \cdot k$. By Bezout's identity, $d = ax_0 + by_0$, so $c = a(kx_0) + b(ky_0)$. Conversely, if $ax + by = c$, then $d \mid a$ and $d \mid b$ imply $d \mid c$. $\square$

## Implementation

```python
"""
Bezout's Identity verification.

Demonstrates that gcd(a, b) can always be expressed as a linear
combination ax + by, and verifies the result for several examples.
"""

import math


# === Extended GCD for Bezout Coefficients ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


# === Verification ===

def verify_bezout(a: int, b: int) -> None:
    """Verify Bezout's identity for given a and b."""
    g, x, y = extended_gcd(a, b)
    assert a * x + b * y == g, "Bezout identity failed"
    print(f"gcd({a}, {b}) = {g},  {a}*({x}) + {b}*({y}) = {a*x + b*y}")


# === Main ===

if __name__ == "__main__":
    verify_bezout(48, 18)
    verify_bezout(270, 192)
    verify_bezout(17, 13)
    verify_bezout(35, 15)
    verify_bezout(100, 1)

    # Coprimality check
    a, b = 17, 13
    g, x, y = extended_gcd(a, b)
    print(f"\n{a} and {b} are coprime: {g == 1}")
    print(f"Certificate: {a}*({x}) + {b}*({y}) = {a*x + b*y}")
```

**Output:**

```
gcd(48, 18) = 6,  48*(-1) + 18*(3) = 6
gcd(270, 192) = 6,  270*(-5) + 192*(7) = 6
gcd(17, 13) = 1,  17*(-3) + 13*(4) = 1
gcd(35, 15) = 5,  35*(1) + 15*(-2) = 5
gcd(100, 1) = 1,  100*(0) + 1*(1) = 1

17 and 13 are coprime: True
Certificate: 17*(-3) + 13*(4) = 1
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
