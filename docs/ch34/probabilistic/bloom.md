# Bloom Filters

A hash set answers membership queries exactly but uses space proportional to the number of stored elements. When the universe is large and a small probability of **false positives** is acceptable, a **Bloom filter** provides the same interface using far less memory. It guarantees no false negatives: if an element was inserted, the filter always reports it as present.

## Structure

A Bloom filter consists of:

- A **bit array** $B$ of $m$ bits, initially all zero.
- A family of $k$ independent hash functions $h_1, h_2, \ldots, h_k$, each mapping elements to $\{0, 1, \ldots, m-1\}$.

## Operations

**Insert($x$)**: Set $B[h_i(x)] = 1$ for all $i \in \{1, \ldots, k\}$.

**Query($x$)**: Return `True` if $B[h_i(x)] = 1$ for all $i$; otherwise return `False`.

- If $x$ was inserted, all $k$ positions were set, so the query always returns `True` (no false negatives).
- If $x$ was never inserted, some positions may still be set by other elements, causing a false positive.

**Delete**: Not supported in a standard Bloom filter, since clearing a bit might remove evidence of other elements.

## False Positive Probability

After inserting $n$ elements into a filter with $m$ bits and $k$ hash functions, each bit remains zero with probability:

$$
\Pr[\text{bit } j = 0] = \left(1 - \frac{1}{m}\right)^{kn} \approx e^{-kn/m}
$$

A false positive occurs when all $k$ hash positions for a non-member happen to be set:

$$
P_{\text{fp}} \approx \left(1 - e^{-kn/m}\right)^k
$$

## Optimal Number of Hash Functions

Minimizing $P_{\text{fp}}$ with respect to $k$ (treating $m$ and $n$ as fixed) yields:

$$
k^* = \frac{m}{n} \ln 2
$$

At this optimum, the false positive rate simplifies to:

$$
P_{\text{fp}}^* = \left(\frac{1}{2}\right)^k = 2^{-k}
$$

For a target false positive rate $\epsilon$, the required number of bits per element is:

$$
\frac{m}{n} = -\frac{\ln \epsilon}{(\ln 2)^2} \approx 1.44 \log_2 \frac{1}{\epsilon}
$$

!!! example "Practical sizing"
    For $\epsilon = 1\%$ (1 in 100 false positives), we need about $m/n \approx 9.6$ bits per element and $k = 7$ hash functions. This is remarkably space-efficient: storing $10^6$ elements requires only about 1.2 MB regardless of element size.

## Complexity

| Operation | Time | Space |
|---|---|---|
| Insert | $O(k)$ | -- |
| Query | $O(k)$ | -- |
| Total structure | -- | $O(m)$ bits |

Since $k$ is typically a small constant (3--10), all operations run in $O(1)$ time.

## Implementation

```python
"""
Bloom Filter -- space-efficient probabilistic membership test.

Uses k independent hash functions over an m-bit array.  Guarantees
no false negatives; false positive rate is approximately (1-e^{-kn/m})^k.
"""

import hashlib
import math


# === Bloom Filter =============================================================

class BloomFilter:
    """Probabilistic set supporting add and membership query."""

    def __init__(self, expected_items: int, fp_rate: float = 0.01):
        """Size the filter for *expected_items* at the given false positive rate."""
        self.n_expected = expected_items
        self.fp_rate = fp_rate
        # Optimal number of bits
        self.m = max(1, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        # Optimal number of hash functions
        self.k = max(1, int((self.m / expected_items) * math.log(2)))
        self.bits = [False] * self.m
        self.count = 0

    def _hashes(self, item: str) -> list[int]:
        """Compute k hash positions for *item*."""
        positions = []
        for i in range(self.k):
            digest = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            positions.append(int(digest, 16) % self.m)
        return positions

    def add(self, item: str) -> None:
        """Insert *item* into the filter."""
        for pos in self._hashes(item):
            self.bits[pos] = True
        self.count += 1

    def query(self, item: str) -> bool:
        """Test whether *item* is possibly in the set."""
        return all(self.bits[pos] for pos in self._hashes(item))


# === Main =====================================================================

if __name__ == "__main__":
    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    print(f"Filter size: {bf.m} bits, {bf.k} hash functions")

    # Insert some items
    inserted = ["apple", "banana", "cherry", "date", "elderberry"]
    for word in inserted:
        bf.add(word)

    # Query inserted items (should all be True)
    print("\nInserted items:")
    for word in inserted:
        print(f"  {word}: {bf.query(word)}")

    # Query non-inserted items (should mostly be False)
    print("\nNon-inserted items:")
    for word in ["fig", "grape", "honeydew", "kiwi"]:
        print(f"  {word}: {bf.query(word)}")
```

**Output:**

```
Filter size: 958 bits, 6 hash functions

Inserted items:
  apple: True
  banana: True
  cherry: True
  date: True
  elderberry: True

Non-inserted items:
  fig: False
  grape: False
  honeydew: False
  kiwi: False
```

All inserted items return `True` (guaranteed), while non-inserted items return `False` in this small example. With more elements approaching the filter capacity, some false positives would appear at the configured 1% rate.

## Reference

- Bloom, B.H. "Space/Time Trade-offs in Hash Coding with Allowable Errors." *CACM*, 1970
- Mitzenmacher, M. and Upfal, E. *Probability and Computing*. Cambridge University Press, 2005
