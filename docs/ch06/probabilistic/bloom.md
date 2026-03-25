# Bloom Filters

A standard hash set stores every element explicitly, consuming $O(n)$ space for $n$ elements. When the goal is simply to test membership --- "has this element been seen before?" --- a **Bloom filter** answers this question using far less memory, at the cost of a small probability of false positives. A Bloom filter never produces false negatives: if it says an element is absent, it is definitely absent.

## Structure

A Bloom filter consists of:

- A **bit array** $B[0 \ldots m-1]$ of $m$ bits, all initially set to 0.
- A set of $k$ independent hash functions $h_1, h_2, \ldots, h_k$, each mapping elements to $\{0, 1, \ldots, m-1\}$.

**Insert** element $x$: for each $i \in \{1, \ldots, k\}$, set $B[h_i(x)] = 1$.

**Query** element $x$: return "possibly present" if $B[h_i(x)] = 1$ for all $i$; return "definitely absent" if any $B[h_i(x)] = 0$.

Because different elements may set the same bits, a query can return "possibly present" for an element that was never inserted --- a **false positive**. However, if any of the $k$ bits is 0, the element was never inserted --- **no false negatives**.

## False Positive Probability

After inserting $n$ elements into a Bloom filter with $m$ bits and $k$ hash functions, the probability that a specific bit is still 0 is

$$
\left(1 - \frac{1}{m}\right)^{kn} \approx e^{-kn/m}
$$

A false positive occurs when all $k$ bits for a non-member are set to 1. The false positive probability is

$$
P(\text{FP}) \approx \left(1 - e^{-kn/m}\right)^k
$$

## Optimal Number of Hash Functions

To minimize $P(\text{FP})$ for given $m$ and $n$, take the derivative with respect to $k$ and set it to zero. The optimal number of hash functions is

$$
k^* = \frac{m}{n} \ln 2 \approx 0.693 \cdot \frac{m}{n}
$$

Substituting back, the minimum false positive rate at optimal $k$ is

$$
P(\text{FP})_{\min} = \left(\frac{1}{2}\right)^{k^*} = 2^{-(m/n) \ln 2}
$$

## Bits Per Element

For a target false positive rate $\epsilon$, the required number of bits per element is

$$
\frac{m}{n} = -\frac{\ln \epsilon}{(\ln 2)^2} \approx -1.44 \log_2 \epsilon
$$

| Target FP rate $\epsilon$ | Bits per element $m/n$ | Hash functions $k^*$ |
|---|---|---|
| 1% | 9.6 | 7 |
| 0.1% | 14.4 | 10 |
| 0.01% | 19.2 | 13 |

A 1% false positive rate requires fewer than 10 bits per element --- far less than storing the elements themselves (which might be strings of hundreds of bytes).

## Limitations

- **No deletion**: setting a bit to 0 could affect other elements. Counting Bloom filters replace single bits with counters to support deletion.
- **No enumeration**: the filter cannot list the elements it contains.
- **Fixed capacity**: the false positive rate increases as more elements are inserted beyond the designed capacity.

## Applications

Bloom filters are used in:

- **Web caching**: Chrome's safe browsing checks URLs against a Bloom filter before querying the server.
- **Database query optimization**: avoid expensive disk reads for non-existent keys.
- **Network routing**: detect duplicate packets in high-speed switches.
- **Spell checking**: quickly reject words that are definitely not in the dictionary.

## Python Implementation

```python
"""
Bloom filter implementation.

Demonstrates the space-efficient probabilistic data structure
for approximate membership testing with no false negatives.
"""

import hashlib
import math


# === Bloom Filter ===

class BloomFilter:
    """Bloom filter with configurable size and hash count."""

    def __init__(self, expected_items, fp_rate=0.01):
        # Compute optimal m and k
        self.size = self._optimal_size(expected_items, fp_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        self.bits = [False] * self.size
        self.count = 0

    @staticmethod
    def _optimal_size(n, p):
        """Compute optimal bit array size for n items and FP rate p."""
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_hashes(m, n):
        """Compute optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))

    def _hashes(self, item):
        """Generate k hash values for the given item."""
        for i in range(self.num_hashes):
            h = int(hashlib.md5(
                f"{item}:{i}".encode()
            ).hexdigest(), 16)
            yield h % self.size

    def add(self, item):
        """Add item to the Bloom filter."""
        for h in self._hashes(item):
            self.bits[h] = True
        self.count += 1

    def query(self, item):
        """Check if item is possibly in the filter.

        Returns True if possibly present, False if definitely absent.
        """
        return all(self.bits[h] for h in self._hashes(item))


# === Demonstration ===

if __name__ == "__main__":
    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    print(f"Bit array size: {bf.size}")
    print(f"Hash functions: {bf.num_hashes}")

    # Add some items
    present = ["apple", "banana", "cherry", "date", "elderberry"]
    for item in present:
        bf.add(item)

    # Query present and absent items
    test = ["apple", "banana", "grape", "melon", "cherry", "kiwi"]
    for item in test:
        status = "possibly present" if bf.query(item) else "definitely absent"
        print(f"  {item:12s} -> {status}")
```

**Output:**
```
Bit array size: 959
Hash functions: 7
  apple        -> possibly present
  banana       -> possibly present
  grape        -> definitely absent
  melon        -> definitely absent
  cherry       -> possibly present
  kiwi         -> definitely absent
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Bloom, B. H. "Space/Time Trade-offs in Hash Coding with Allowable Errors." *Communications of the ACM*, 13(7), 1970.
