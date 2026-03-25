# Modular Operations

Cryptographic protocols, hash functions, and competitive programming problems all perform arithmetic where only the remainder matters. Modular arithmetic provides a rigorous framework for "clock arithmetic," where numbers wrap around after reaching a modulus. This page establishes the congruence relation and the rules for addition, subtraction, and multiplication modulo $m$.

## Congruence Relation

For a positive integer $m$, we say $a$ is **congruent** to $b$ modulo $m$, written:

$$
a \equiv b \pmod{m}
$$

if and only if $m \mid (a - b)$, that is, $m$ divides the difference $a - b$. Equivalently, $a$ and $b$ leave the same remainder when divided by $m$.

!!! info "Congruence as Equivalence Relation"

    Congruence modulo $m$ is an equivalence relation on $\mathbb{Z}$: it is reflexive ($a \equiv a$), symmetric ($a \equiv b \Rightarrow b \equiv a$), and transitive ($a \equiv b$ and $b \equiv c \Rightarrow a \equiv c$). The equivalence classes are called **residue classes** modulo $m$.

The set of residue classes modulo $m$ is denoted $\mathbb{Z}/m\mathbb{Z} = \{0, 1, 2, \ldots, m-1\}$, where each element represents its entire equivalence class.

## Addition

If $a \equiv a' \pmod{m}$ and $b \equiv b' \pmod{m}$, then:

$$
a + b \equiv a' + b' \pmod{m}
$$

**Proof.** By hypothesis, $m \mid (a - a')$ and $m \mid (b - b')$. Adding: $m \mid ((a + b) - (a' + b'))$, so $a + b \equiv a' + b' \pmod{m}$. $\square$

!!! example "Modular Addition"

    With $m = 7$: since $15 \equiv 1 \pmod{7}$ and $20 \equiv 6 \pmod{7}$,

    $$
    15 + 20 = 35 \equiv 0 \pmod{7}
    $$

    And indeed $1 + 6 = 7 \equiv 0 \pmod{7}$. $\checkmark$

## Subtraction

If $a \equiv a' \pmod{m}$ and $b \equiv b' \pmod{m}$, then:

$$
a - b \equiv a' - b' \pmod{m}
$$

The proof is identical to addition, replacing $+$ with $-$.

## Multiplication

If $a \equiv a' \pmod{m}$ and $b \equiv b' \pmod{m}$, then:

$$
a \cdot b \equiv a' \cdot b' \pmod{m}
$$

**Proof.** Write $a = a' + km$ and $b = b' + lm$ for integers $k, l$. Then:

$$
ab = a'b' + a'lm + b'km + klm^2 = a'b' + m(a'l + b'k + klm)
$$

So $m \mid (ab - a'b')$, giving $ab \equiv a'b' \pmod{m}$. $\square$

!!! tip "Practical Consequence"

    These properties allow reducing intermediate results modulo $m$ at any point during a computation. When computing $(a \cdot b) \bmod m$, we can first reduce $a$ and $b$ modulo $m$, then multiply, then reduce again. This prevents integer overflow in implementations.

## Exponentiation

Repeated application of the multiplication rule gives:

$$
a^n \equiv (a \bmod m)^n \pmod{m}
$$

For efficient computation of $a^n \bmod m$, see [Modular Exponentiation](exponentiation.md).

## Division (Caution)

!!! warning "Division Does Not Always Work"

    Unlike addition and multiplication, **division is not always valid** in modular arithmetic. From $ac \equiv bc \pmod{m}$, we **cannot** conclude $a \equiv b \pmod{m}$ in general.

    For example, $2 \cdot 3 \equiv 2 \cdot 6 \pmod{6}$ (both are $12 \equiv 0$), but $3 \not\equiv 6 \pmod{6}$.

Division is valid only when $\gcd(c, m) = 1$. In that case, $c$ has a modular inverse $c^{-1}$, and we can "divide" by multiplying both sides by $c^{-1}$ (see [Modular Inverse](inverse.md)).

## Properties Summary

For any integers $a, b, c$ and positive integer $m$:

| Property | Statement |
|----------|-----------|
| Closure | $(a + b) \bmod m \in \{0, \ldots, m-1\}$ |
| Commutativity | $a + b \equiv b + a$, $a \cdot b \equiv b \cdot a$ |
| Associativity | $(a + b) + c \equiv a + (b + c)$, $(ab)c \equiv a(bc)$ |
| Distributivity | $a(b + c) \equiv ab + ac$ |
| Identity | $a + 0 \equiv a$, $a \cdot 1 \equiv a$ |
| Additive inverse | $a + (m - a) \equiv 0$ |

These properties make $(\mathbb{Z}/m\mathbb{Z}, +, \cdot)$ a **commutative ring**. When $m$ is prime, every nonzero element has a multiplicative inverse, making it a **field**.

## Implementation

```python
"""
Modular arithmetic operations.

Demonstrates addition, subtraction, multiplication, and exponentiation
modulo m with overflow-safe intermediate reductions.
"""


# === Modular Operations ===

def mod_add(a: int, b: int, m: int) -> int:
    """Compute (a + b) mod m."""
    return ((a % m) + (b % m)) % m


def mod_sub(a: int, b: int, m: int) -> int:
    """Compute (a - b) mod m, ensuring nonnegative result."""
    return ((a % m) - (b % m) + m) % m


def mod_mul(a: int, b: int, m: int) -> int:
    """Compute (a * b) mod m."""
    return ((a % m) * (b % m)) % m


def mod_pow(base: int, exp: int, m: int) -> int:
    """Compute base^exp mod m using repeated squaring."""
    result = 1
    base = base % m
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % m
        exp //= 2
        base = (base * base) % m
    return result


# === Main ===

if __name__ == "__main__":
    m = 7
    print(f"Modular arithmetic with m = {m}")
    print(f"  (15 + 20) mod {m} = {mod_add(15, 20, m)}")
    print(f"  (15 - 20) mod {m} = {mod_sub(15, 20, m)}")
    print(f"  (15 * 20) mod {m} = {mod_mul(15, 20, m)}")
    print(f"  3^10 mod {m} = {mod_pow(3, 10, m)}")

    # Verify consistency
    print(f"\nVerification:")
    print(f"  (15 + 20) = {15 + 20}, {35 % m} = {mod_add(15, 20, m)}")
    print(f"  (15 * 20) = {15 * 20}, {300 % m} = {mod_mul(15, 20, m)}")
```

**Output:**

```
Modular arithmetic with m = 7
  (15 + 20) mod 7 = 0
  (15 - 20) mod 7 = 2
  (15 * 20) mod 7 = 6
  3^10 mod 7 = 4

Verification:
  (15 + 20) = 35, 0 = 0
  (15 * 20) = 300, 6 = 6
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
