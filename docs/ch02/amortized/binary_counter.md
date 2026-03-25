# Binary Counter

The binary counter is one of the most instructive examples in amortized analysis. At first glance, incrementing a binary counter seems potentially expensive because a single increment can flip many bits (for instance, $0111\ldots1 \to 1000\ldots0$ flips all $k$ bits). A naive analysis of $n$ increments gives $O(nk)$. Amortized analysis reveals the true cost: only $O(n)$ total bit flips, or $O(1)$ amortized per increment.

## Problem Setup

A $k$-bit binary counter is stored as an array $A[0..k-1]$ where $A[0]$ is the least significant bit. The single supported operation is `INCREMENT`:

```
INCREMENT(A):
    i = 0
    while i < k and A[i] == 1:
        A[i] = 0      # reset bit (flip 1 → 0)
        i = i + 1
    if i < k:
        A[i] = 1      # set bit (flip 0 → 1)
```

The cost of an increment is the number of bits flipped. In the worst case, a single increment flips all $k$ bits, so the worst-case cost per operation is $O(k)$.

## Aggregate Analysis

The aggregate method counts total bit flips across all $n$ increments.

**Observation:** Bit $j$ flips only when the counter value is a multiple of $2^j$. Over $n$ increments starting from 0:

- Bit 0 flips every increment: $n$ times
- Bit 1 flips every 2nd increment: $\lfloor n/2 \rfloor$ times
- Bit 2 flips every 4th increment: $\lfloor n/4 \rfloor$ times
- Bit $j$ flips $\lfloor n/2^j \rfloor$ times

The total number of bit flips is:

$$
T(n) = \sum_{j=0}^{k-1} \left\lfloor \frac{n}{2^j} \right\rfloor < n \sum_{j=0}^{\infty} \frac{1}{2^j} = 2n
$$

Therefore the amortized cost per increment is:

$$
\hat{c} = \frac{T(n)}{n} < 2 = O(1)
$$

## Accounting Analysis

The accounting method assigns an amortized cost of $\hat{c} = 2$ to each increment:

- **1 unit** pays for setting one bit from 0 to 1 (every increment sets exactly one bit).
- **1 unit** is stored as credit on the bit that was just set.

When a bit is later reset from 1 to 0, the credit stored on that bit pays for the reset. Since each bit that gets reset was previously set (and received 1 unit of credit at that time), every reset is prepaid.

**Credit invariant:** The total credit equals the number of 1-bits in the counter, which is always non-negative. This confirms that the amortized cost of 2 per increment is valid.

## Potential Analysis

Define the potential function $\Phi$ as the number of 1-bits in the counter:

$$
\Phi(D_i) = \text{number of 1-bits after operation } i
$$

This potential satisfies $\Phi(D_0) = 0$ and $\Phi(D_i) \geq 0$ for all $i$.

Suppose the $i$-th increment resets $t_i$ bits from 1 to 0 and then sets at most one bit from 0 to 1. The actual cost is $c_i = t_i + 1$ (assuming the counter does not overflow). The change in potential is:

$$
\Phi(D_i) - \Phi(D_{i-1}) = (1 - t_i)
$$

because the increment removes $t_i$ ones and adds 1 one. The amortized cost is:

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = (t_i + 1) + (1 - t_i) = 2
$$

The $t_i$ terms cancel, giving a constant amortized cost of 2 per increment regardless of how many bits are flipped.

## Python Example

```python
"""
Binary counter amortized analysis demonstration.

Implements a binary counter and tracks bit flips per increment,
verifying that the total cost grows linearly (not quadratically).
"""


# ===================================================================
# Binary Counter Implementation
# ===================================================================
class BinaryCounter:
    """k-bit binary counter with bit-flip cost tracking."""

    def __init__(self, num_bits):
        self.bits = [0] * num_bits
        self.k = num_bits
        self.total_flips = 0
        self.num_increments = 0

    def increment(self):
        """Increment by 1 and return the number of bit flips."""
        flips = 0
        i = 0
        # Reset consecutive trailing 1-bits
        while i < self.k and self.bits[i] == 1:
            self.bits[i] = 0
            flips += 1
            i += 1
        # Set the next 0-bit (if counter hasn't overflowed)
        if i < self.k:
            self.bits[i] = 1
            flips += 1
        self.total_flips += flips
        self.num_increments += 1
        return flips

    def value(self):
        """Return the current decimal value."""
        return sum(b * (2 ** i) for i, b in enumerate(self.bits))

    def ones_count(self):
        """Return the number of 1-bits (potential function)."""
        return sum(self.bits)


# ===================================================================
# Aggregate Analysis Verification
# ===================================================================
def verify_aggregate(n, k=16):
    """Verify that total flips < 2n for n increments."""
    counter = BinaryCounter(k)
    for _ in range(n):
        counter.increment()
    ratio = counter.total_flips / n
    print(f"n={n}: total flips={counter.total_flips}, "
          f"ratio={ratio:.4f}, bound=2.0")
    return counter.total_flips < 2 * n


# ===================================================================
# Per-Bit Flip Frequency Verification
# ===================================================================
def verify_bit_frequencies(n, k=8):
    """Verify that bit j flips floor(n / 2^j) times."""
    flip_count = [0] * k
    bits = [0] * k
    for _ in range(n):
        i = 0
        while i < k and bits[i] == 1:
            bits[i] = 0
            flip_count[i] += 1
            i += 1
        if i < k:
            bits[i] = 1
            flip_count[i] += 1

    print(f"\nBit flip frequencies for n={n} increments:")
    print(f"{'Bit':>4} {'Actual':>8} {'n/2^j':>8}")
    print("-" * 22)
    for j in range(k):
        expected = n // (2 ** j)
        # Each bit j flips when set (floor(n/2^j) times when it goes 0->1)
        # and when reset, total flips = floor(n/2^j) for set + floor(n/2^j)
        # Actually bit j changes state every 2^j increments
        print(f"{j:>4} {flip_count[j]:>8} {expected:>8}")


# ===================================================================
# Demonstration
# ===================================================================
if __name__ == "__main__":
    print("=== Aggregate Analysis ===")
    for n in [100, 1000, 10000]:
        verify_aggregate(n)

    verify_bit_frequencies(64)

    print("\n=== Step-by-Step (first 16 increments) ===")
    counter = BinaryCounter(8)
    print(f"{'Step':>5} {'Binary':>10} {'Flips':>6} "
          f"{'Total':>6} {'Ones':>5} {'Amortized':>10}")
    print("-" * 46)
    for step in range(1, 17):
        flips = counter.increment()
        binary = ''.join(str(b) for b in reversed(counter.bits))
        amortized = counter.total_flips / step
        print(
            f"{step:>5} {binary:>10} {flips:>6} "
            f"{counter.total_flips:>6} {counter.ones_count():>5} "
            f"{amortized:>10.2f}"
        )
```

## Decrement Operation

If the counter also supports `DECREMENT`, the amortized analysis changes significantly. A sequence of alternating increments and decrements on the value $2^{k-1} - 1$ (binary $0111\ldots1$) would flip $k$ bits every operation, making the amortized cost $\Theta(k)$ per operation. The potential function approach fails because decrementing can increase the number of 1-bits, so the potential no longer absorbs the cost. This shows that amortized analysis results depend on the set of allowed operations.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.
