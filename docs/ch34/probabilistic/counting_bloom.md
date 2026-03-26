# Counting Bloom Filters

A standard Bloom filter supports only insertion and membership queries -- deletion is impossible because clearing a bit might erase evidence of other elements. A **counting Bloom filter** replaces each bit with an integer counter, enabling deletion while preserving the space-efficient probabilistic membership test.

## Motivation

Consider a network router that uses a Bloom filter to cache routing decisions. When a route is withdrawn, the router must remove the entry. A standard Bloom filter cannot do this safely, but a counting Bloom filter can decrement counters instead of clearing bits, supporting dynamic sets where elements come and go.

## Structure

A counting Bloom filter consists of:

- An array $C$ of $m$ counters, each initialized to zero.
- A family of $k$ hash functions $h_1, \ldots, h_k$ mapping elements to $\{0, 1, \ldots, m-1\}$.

## Operations

**Insert($x$)**: Increment $C[h_i(x)]$ for all $i \in \{1, \ldots, k\}$.

**Delete($x$)**: Decrement $C[h_i(x)]$ for all $i \in \{1, \ldots, k\}$. Only delete elements known to have been inserted; deleting a non-member introduces **false negatives**.

**Query($x$)**: Return `True` if $C[h_i(x)] > 0$ for all $i$; otherwise return `False`.

## False Positive Analysis

The false positive probability is the same as a standard Bloom filter:

$$
P_{\text{fp}} \approx \left(1 - e^{-kn/m}\right)^k
$$

where $n$ is the number of currently inserted elements. Deletions reduce $n$, which decreases the false positive rate -- a desirable property.

## Counter Overflow

Each counter must be wide enough to avoid overflow. With $k$ hash functions and $n$ elements, the expected count at any position is $kn/m$. The probability that a counter reaches value $c$ follows a Poisson tail:

$$
\Pr[C_j \ge c] \le \frac{(kn/m)^c}{c!}
$$

In practice, 4-bit counters (max value 15) suffice for most applications, and overflow events are extremely rare when $m/n$ is properly sized.

!!! warning "Counter underflow"
    Never decrement a counter below zero. If an element is deleted without having been inserted, counters may go negative, introducing false negatives. Guard against this in implementation.

## Space Comparison

| Structure | Space per Slot | Supports Delete |
|---|---|---|
| Standard Bloom filter | 1 bit | No |
| Counting Bloom filter (4-bit) | 4 bits | Yes |
| Counting Bloom filter (8-bit) | 8 bits | Yes |

A 4-bit counting Bloom filter uses 4x the space of a standard Bloom filter -- still dramatically less than a hash table.

## Implementation

```python
"""
Counting Bloom Filter -- probabilistic set with deletion support.

Replaces the bit array of a standard Bloom filter with integer
counters so that elements can be removed without affecting other members.
"""

import hashlib
import math


# === Counting Bloom Filter ====================================================

class CountingBloomFilter:
    """Bloom filter with counters supporting insert, delete, and query."""

    def __init__(self, expected_items: int, fp_rate: float = 0.01,
                 counter_bits: int = 4):
        self.n_expected = expected_items
        self.fp_rate = fp_rate
        self.max_count = (1 << counter_bits) - 1
        # Optimal sizing
        self.m = max(1, int(-expected_items * math.log(fp_rate)
                            / (math.log(2) ** 2)))
        self.k = max(1, int((self.m / expected_items) * math.log(2)))
        self.counters = [0] * self.m

    def _hashes(self, item: str) -> list[int]:
        """Compute k hash positions for *item*."""
        positions = []
        for i in range(self.k):
            digest = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            positions.append(int(digest, 16) % self.m)
        return positions

    def add(self, item: str) -> None:
        """Insert *item* by incrementing counters."""
        for pos in self._hashes(item):
            if self.counters[pos] < self.max_count:
                self.counters[pos] += 1

    def remove(self, item: str) -> None:
        """Remove *item* by decrementing counters (must have been inserted)."""
        for pos in self._hashes(item):
            if self.counters[pos] > 0:
                self.counters[pos] -= 1

    def query(self, item: str) -> bool:
        """Test whether *item* is possibly in the set."""
        return all(self.counters[pos] > 0 for pos in self._hashes(item))


# === Main =====================================================================

if __name__ == "__main__":
    cbf = CountingBloomFilter(expected_items=100, fp_rate=0.01)
    print(f"Filter: {cbf.m} counters, {cbf.k} hash functions")

    # Insert items
    for word in ["apple", "banana", "cherry"]:
        cbf.add(word)

    print("\nAfter inserting apple, banana, cherry:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cbf.query(word)}")

    # Delete banana
    cbf.remove("banana")
    print("\nAfter deleting banana:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cbf.query(word)}")
```

**Output:**

```
Filter: 958 counters, 6 hash functions

After inserting apple, banana, cherry:
  apple: True
  banana: True
  cherry: True
  date: False

After deleting banana:
  apple: True
  banana: False
  cherry: True
  date: False
```

After deletion, `banana` correctly reports `False`, while `apple` and `cherry` remain unaffected. This is the key advantage over a standard Bloom filter: elements can be removed without disturbing other members.

## Reference

- Fan, L., Cao, P., Almeida, J., and Broder, A.Z. "Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol." *IEEE/ACM Trans. Networking*, 2000
- Mitzenmacher, M. and Upfal, E. *Probability and Computing*. Cambridge University Press, 2005
