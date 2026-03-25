# Multiplication Method

The division method ties hash function quality to the choice of table size $m$, requiring careful selection of a prime far from powers of two. The **multiplication method** eliminates this sensitivity: it works well for any table size $m$, including powers of two. This flexibility makes it a practical alternative in systems where the table size must be a power of two for memory alignment or bitwise optimization reasons.

## Definition

The multiplication method computes the hash of an integer key $k$ in two steps:

1. Multiply $k$ by a constant $A \in (0, 1)$ and extract the fractional part.
2. Scale the fractional part by the table size $m$ and take the floor.

Formally:

$$
h(k) = \lfloor m \cdot (kA \bmod 1) \rfloor
$$

where $kA \bmod 1 = kA - \lfloor kA \rfloor$ denotes the fractional part of $kA$. The result $h(k)$ lies in $\{0, 1, \ldots, m-1\}$.

The key insight is that multiplying by an irrational (or irrational-like) constant $A$ and extracting the fractional part effectively "scrambles" the key, distributing keys uniformly regardless of their original structure.

## Choice of the Constant A

The constant $A$ can be any value in $(0, 1)$, but some choices yield better distribution than others.

**Knuth's recommendation.** Donald Knuth showed that the golden ratio conjugate:

$$
A = \frac{\sqrt{5} - 1}{2} \approx 0.6180339887\ldots
$$

produces particularly good distribution. This value is optimal in a precise sense: it generates the most uniformly spaced fractional parts $\{kA \bmod 1 : k = 1, 2, \ldots, n\}$ among all choices of $A$, a consequence of the three-distance theorem in number theory.

**Why the golden ratio works.** The continued fraction expansion of the golden ratio conjugate is $[0; 1, 1, 1, \ldots]$, making it the "most irrational" number. Its multiples $kA$ avoid clustering near any rational fraction, which translates to hash values that avoid clustering near any particular slot.

## Efficient Bit-Level Implementation

When the machine word size is $w$ bits (typically $w = 32$ or $w = 64$), the multiplication method can be implemented using only integer multiplication and bit shifts, avoiding floating-point arithmetic entirely.

Choose $A$ and represent it as the integer $s = \lfloor A \cdot 2^w \rfloor$. For $w = 32$ and Knuth's constant:

$$
s = \lfloor 0.6180339887 \cdot 2^{32} \rfloor = 2654435769
$$

The hash computation becomes:

$$
h(k) = (k \cdot s \bmod 2^w) \gg (w - p)
$$

where $m = 2^p$ and $\gg$ denotes a right bit shift. The steps are:

1. Multiply: $k \cdot s$ produces a $2w$-bit product.
2. Extract: take the lower $w$ bits (i.e., $k \cdot s \bmod 2^w$), which corresponds to the fractional part.
3. Shift: the top $p$ bits of this $w$-bit quantity give $h(k)$, equivalent to multiplying by $m = 2^p$ and taking the floor.

This implementation uses only integer arithmetic and runs in $O(1)$ time on all architectures.

??? example "Step-by-Step Computation"

    Let $w = 8$ (for small illustration), $m = 2^3 = 8$ (so $p = 3$), and $A \approx 0.618$:

    $$
    s = \lfloor 0.618 \times 256 \rfloor = 158
    $$

    For $k = 123$:

    $$
    k \cdot s = 123 \times 158 = 19434
    $$

    $$
    19434 \bmod 256 = 19434 - 75 \times 256 = 19434 - 19200 = 234
    $$

    $$
    234 \text{ in binary} = 11101010_2
    $$

    Extract the top $p = 3$ bits: $111_2 = 7$.

    $$
    h(123) = 7
    $$

## Comparison with the Division Method

| Property | Division Method | Multiplication Method |
|---|---|---|
| Formula | $h(k) = k \bmod m$ | $h(k) = \lfloor m(kA \bmod 1) \rfloor$ |
| Table size restriction | $m$ should be prime, not near $2^p$ | Any $m$ works; $m = 2^p$ is ideal |
| Sensitivity to $m$ | High | Low |
| Implementation | Single division/modulo | Multiply + shift (no division) |
| Bit utilization | Low-order bits only when $m = 2^p$ | All bits contribute |

The multiplication method is generally preferred when the table size must be a power of two. The division method is simpler when a prime table size can be chosen freely.

## Implementation

```python
"""
Multiplication method hash table.

Demonstrates h(k) = floor(m * (k * A mod 1)) using both
floating-point and bit-shift implementations.
"""

import math


# === Multiplication Method Hash Functions ===

GOLDEN_RATIO_CONJUGATE = (math.sqrt(5) - 1) / 2  # ~0.6180339887


def hash_multiply_float(key: int, m: int, A: float = GOLDEN_RATIO_CONJUGATE) -> int:
    """Compute hash using the floating-point multiplication method."""
    return int(m * ((key * A) % 1))


def hash_multiply_bitshift(key: int, p: int, w: int = 32) -> int:
    """Compute hash using the bit-shift multiplication method.

    Args:
        key: Integer key to hash.
        p: Log2 of table size (m = 2^p).
        w: Machine word size in bits.

    Returns:
        Hash value in {0, 1, ..., 2^p - 1}.
    """
    s = int(GOLDEN_RATIO_CONJUGATE * (2 ** w))
    return ((key * s) % (2 ** w)) >> (w - p)


# === Demonstration ===

if __name__ == "__main__":
    m = 16  # table size (power of two)
    p = 4   # log2(m)

    keys = [10, 22, 37, 45, 59, 72, 88, 100]
    print("Floating-point multiplication method (m=16):")
    for k in keys:
        h = hash_multiply_float(k, m)
        frac = (k * GOLDEN_RATIO_CONJUGATE) % 1
        print(f"  h({k:3d}) = floor({m} * {frac:.4f}) = {h}")

    print("\nBit-shift multiplication method (m=16, w=32):")
    for k in keys:
        h = hash_multiply_bitshift(k, p)
        print(f"  h({k:3d}) = {h}")
```

**Output:**
```
Floating-point multiplication method (m=16):
  h( 10) = floor(16 * 0.1803) = 2
  h( 22) = floor(16 * 0.5967) = 9
  h( 37) = floor(16 * 0.8675) = 13
  h( 45) = floor(16 * 0.8115) = 12
  h( 59) = floor(16 * 0.4639) = 7
  h( 72) = floor(16 * 0.4987) = 7
  h( 88) = floor(16 * 0.3871) = 6
  h(100) = floor(16 * 0.8034) = 12

Bit-shift multiplication method (m=16, w=32):
  h( 10) = 2
  h( 22) = 9
  h( 37) = 13
  h( 45) = 12
  h( 59) = 7
  h( 72) = 7
  h( 88) = 6
  h(100) = 12
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
