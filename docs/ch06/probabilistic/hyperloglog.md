# HyperLogLog

Counting the exact number of distinct elements in a large data stream requires space proportional to the cardinality itself.  For streams with billions of elements, this is impractical.  The **HyperLogLog** algorithm (Flajolet et al., 2007) estimates the number of distinct elements using only $O(\log \log n + \log n)$ bits per register -- in practice, about 1.5 KB of memory can estimate cardinalities up to $10^9$ with a standard error of approximately 2%.

## The Intuition

The algorithm rests on a probabilistic observation.  When hashing elements uniformly at random, the probability that a hash value starts with $k$ leading zeros is $2^{-k}$.  If we observe a hash with $k$ leading zeros, it is likely that approximately $2^k$ distinct elements have been hashed.  Tracking the maximum number of leading zeros across all hashed elements gives a rough estimate of the cardinality.

However, a single such estimate has high variance.  HyperLogLog reduces this variance by splitting the stream into $m$ substreams using the first $p = \log_2 m$ bits of each hash, then tracking the maximum leading-zero count in each substream independently, and finally combining the $m$ estimates using a harmonic mean.

## Algorithm

### Setup

Choose a hash function $h : U \to \{0, 1\}^{L}$ (typically $L = 64$ bits).  Fix $p = \log_2 m$ where $m$ is a power of 2 (the number of registers).  Initialize $m$ registers $M[0], M[1], \ldots, M[m-1]$ to 0.

### Add Element

For each element $x$:

1. Compute $h(x)$.
2. Use the first $p$ bits of $h(x)$ as the register index $j$.
3. Let $w$ be the remaining $L - p$ bits.
4. Let $\rho(w)$ be the position of the leftmost 1-bit in $w$ (i.e., $\rho(w) = 1 + \lfloor \log_2(1/w) \rfloor$ for $w > 0$).
5. Update: $M[j] \leftarrow \max(M[j], \rho(w))$.

### Estimate Cardinality

The raw estimate uses the harmonic mean of $2^{M[j]}$ across all registers:

$$
E = \alpha_m \cdot m^2 \cdot \left( \sum_{j=0}^{m-1} 2^{-M[j]} \right)^{-1}
$$

where $\alpha_m$ is a bias-correction constant:

$$
\alpha_m = \left( m \int_0^\infty \left( \log_2 \frac{2 + u}{1 + u} \right)^m du \right)^{-1}
$$

For practical values: $\alpha_{16} = 0.673$, $\alpha_{32} = 0.697$, $\alpha_{64} = 0.709$, and $\alpha_m = 0.7213 / (1 + 1.079/m)$ for $m \geq 128$.

## Error Analysis

**Theorem.** The standard error of the HyperLogLog estimate with $m$ registers is:

$$
\frac{\sigma}{\hat{n}} \approx \frac{1.04}{\sqrt{m}}
$$

This means that with $m = 2^{10} = 1024$ registers (using 5 bits each, total $\approx$ 640 bytes), the standard error is about 3.25%.  With $m = 2^{14} = 16384$ registers ($\approx$ 12 KB), the standard error drops to about 0.81%.

!!! tip "Practical accuracy"
    The standard HyperLogLog implementation used by Redis (`PFADD`, `PFCOUNT`) uses $m = 16384$ registers with 6 bits each, consuming 12 KB total.  This provides a standard error below 1% for cardinalities ranging from 0 to $2^{64}$.

## Small and Large Range Corrections

The raw harmonic mean estimate works well in the intermediate range but needs corrections at the extremes.

**Small range correction.** When $E \leq \frac{5}{2} m$ and some registers are still zero, use a linear counting estimate instead:

$$
E^* = m \ln \frac{m}{V}
$$

where $V$ is the number of registers equal to zero.  This correction handles the bias that occurs when many registers have not yet been touched.

**Large range correction.** When $E > \frac{1}{30} \cdot 2^{L}$ (where $L$ is the hash length), hash collisions become significant.  Apply the correction:

$$
E^* = -2^L \ln\left(1 - \frac{E}{2^L}\right)
$$

## Mergeability

A key property of HyperLogLog is that two sketches can be merged by taking the component-wise maximum of their registers:

$$
M_{\text{merged}}[j] = \max(M_A[j], M_B[j]) \quad \text{for } j = 0, 1, \ldots, m-1
$$

This makes HyperLogLog ideal for distributed counting: each node computes a local sketch, and the central aggregator merges them in $O(m)$ time.

## Implementation

```python
"""HyperLogLog cardinality estimation."""

import hashlib
import math


# === HyperLogLog ===

class HyperLogLog:
    """Estimate the number of distinct elements using O(m) space."""

    def __init__(self, p: int = 10):
        self.p = p
        self.m = 1 << p  # number of registers
        self.registers = [0] * self.m
        # Bias correction constant
        if self.m >= 128:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)
        elif self.m == 64:
            self.alpha = 0.709
        elif self.m == 32:
            self.alpha = 0.697
        else:
            self.alpha = 0.673

    def _hash(self, item: str) -> int:
        """Hash an item to a 64-bit integer."""
        digest = hashlib.sha256(item.encode()).hexdigest()
        return int(digest[:16], 16)  # 64-bit hash

    @staticmethod
    def _rho(w: int, max_bits: int) -> int:
        """Position of the leftmost 1-bit (1-indexed)."""
        if w == 0:
            return max_bits + 1
        pos = 1
        while (w >> (max_bits - pos)) & 1 == 0:
            pos += 1
        return pos

    def add(self, item: str) -> None:
        """Add an element to the sketch."""
        h = self._hash(item)
        j = h & (self.m - 1)  # first p bits as register index
        w = h >> self.p  # remaining bits
        self.registers[j] = max(self.registers[j], self._rho(w, 64 - self.p))

    def count(self) -> int:
        """Estimate the number of distinct elements."""
        # Raw harmonic mean estimate
        indicator = sum(2.0 ** (-r) for r in self.registers)
        estimate = self.alpha * self.m * self.m / indicator

        # Small range correction
        if estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                estimate = self.m * math.log(self.m / zeros)

        # Large range correction (for 64-bit hashes)
        two_to_64 = 2.0 ** 64
        if estimate > two_to_64 / 30.0:
            estimate = -two_to_64 * math.log(1.0 - estimate / two_to_64)

        return int(estimate)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Merge two HyperLogLog sketches."""
        assert self.p == other.p, "Cannot merge sketches with different p"
        result = HyperLogLog(self.p)
        result.registers = [
            max(a, b) for a, b in zip(self.registers, other.registers)
        ]
        return result


# === Demonstration ===

if __name__ == "__main__":
    hll = HyperLogLog(p=10)  # 1024 registers

    # Add elements with known cardinality
    n = 100_000
    for i in range(n):
        hll.add(f"element-{i}")

    estimate = hll.count()
    error = abs(estimate - n) / n * 100
    print(f"True cardinality: {n}")
    print(f"Estimated:        {estimate}")
    print(f"Relative error:   {error:.2f}%")
    print(f"Memory:           {len(hll.registers) * 6} bits "
          f"({len(hll.registers) * 6 / 8:.0f} bytes)")
```

## Complexity Summary

| Operation | Time | Space |
|---|---|---|
| Add element | $O(1)$ | -- |
| Estimate cardinality | $O(m)$ | -- |
| Merge two sketches | $O(m)$ | -- |
| Total space | -- | $O(m)$ registers |

With $m = 2^p$ registers of 6 bits each, the total space is $6 \cdot 2^p$ bits.

## Reference

- Flajolet, P., Fusy, E., Gandouet, O., & Meunier, F. (2007). HyperLogLog: The analysis of a near-optimal cardinality estimation algorithm. *Conference on Analysis of Algorithms (AofA)*.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 11. MIT Press.
