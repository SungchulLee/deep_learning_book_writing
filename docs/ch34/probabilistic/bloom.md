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

## Exercises

**Exercise 1.**
A Bloom filter uses $m$ bits and $k$ hash functions to store $n$ elements. Derive the false positive probability as a function of $m$, $k$, and $n$.

??? success "Solution to Exercise 1"
    After inserting $n$ elements with $k$ hash functions, each of the $kn$ hash outputs sets one of $m$ bits to 1. The probability that a specific bit remains 0 after all insertions is $(1 - 1/m)^{kn} \approx e^{-kn/m}$. A false positive occurs when all $k$ bits checked for a non-member are 1. Each bit is independently 1 with probability $1 - e^{-kn/m}$, so the false positive probability is:

    $$
    p = \left(1 - e^{-kn/m}\right)^k
    $$

    Minimizing over $k$ gives the optimal $k = (m/n) \ln 2$, yielding $p = (1/2)^k = (0.6185)^{m/n}$. $\square$

---

**Exercise 2.**
For a Bloom filter storing $n = 10^6$ elements with a target false positive rate of $1\%$, compute the optimal number of hash functions $k$ and the required number of bits $m$.

??? success "Solution to Exercise 2"
    The optimal bit count is $m = -(n \ln p) / (\ln 2)^2$ where $p = 0.01$. Substituting: $m = -(10^6 \times \ln 0.01) / (0.693)^2 = -(10^6 \times (-4.605)) / 0.480 = 9.59 \times 10^6 \approx 9.6 \times 10^6$ bits $\approx 1.2$ MB. The optimal number of hash functions is $k = (m/n) \ln 2 = 9.6 \times \ln 2 \approx 6.6$, rounded to $k = 7$. With $k = 7$ and $m = 9.6 \times 10^6$, the actual false positive rate is $(1 - e^{-7 \times 10^6 / 9.6 \times 10^6})^7 \approx (0.518)^7 \approx 0.008 = 0.8\%$, just below the $1\%$ target. $\square$

---

**Exercise 3.**
Prove that a Bloom filter cannot have false negatives (if an element was inserted, the filter always reports it as present).

??? success "Solution to Exercise 3"
    When element $x$ is inserted, the filter sets bits at positions $h_1(x), h_2(x), \ldots, h_k(x)$ to 1. Once set to 1, a bit is never cleared (there is no delete operation in a standard Bloom filter). When querying for $x$, the filter checks the same positions $h_1(x), \ldots, h_k(x)$. Since these bits were set to 1 during insertion and are never cleared, all $k$ bits are still 1 at query time. The filter reports "present" whenever all $k$ bits are 1, so it always reports "present" for an inserted element. Therefore, false negatives are impossible. $\square$

---

**Exercise 4.**
A web crawler uses a Bloom filter to avoid revisiting URLs. After crawling $10^8$ URLs with a false positive rate of $0.1\%$, how many legitimate URLs are incorrectly skipped? Discuss the practical impact.

??? success "Solution to Exercise 4"
    With a $0.1\%$ false positive rate, for every URL not in the filter that is queried, there is a $0.001$ probability of a false positive (incorrectly reporting it as visited). If the crawler encounters $10^9$ unique URLs total, then $10^9 - 10^8 = 9 \times 10^8$ are not in the filter. The expected number of false positives is $9 \times 10^8 \times 0.001 = 9 \times 10^5 = 900{,}000$ URLs incorrectly skipped. Practical impact: the crawler misses roughly 0.1% of the web pages it should visit. For a search engine, this is acceptable -- the missed pages are unlikely to be systematically biased (false positives are random). The Bloom filter saves enormous memory: storing $10^8$ URLs as strings would require $\sim$10 GB, while the Bloom filter uses $\sim$120 MB. $\square$

---

**Exercise 5.**
Compare Bloom filters with hash sets for the membership testing problem. Under what conditions is each preferable?

??? success "Solution to Exercise 5"
    **Bloom filter**: space $O(n)$ bits with constant factor depending on the target false positive rate (roughly 10 bits/element for 1% FPR). No false negatives, but has false positives. Does not store the elements themselves. **Hash set**: space $O(n)$ with constant factor depending on element size (e.g., 8 bytes per pointer + element size per entry). Exact -- no false positives or negatives. Stores elements and supports enumeration. Bloom filters are preferable when: (1) memory is critical and elements are large (URLs, file hashes); (2) false positives are acceptable (cache checks, pre-filters before expensive lookups); (3) deletion is not needed. Hash sets are preferable when: (1) exact membership is required; (2) elements need to be retrieved or enumerated; (3) deletion support is needed; (4) the false positive rate of a Bloom filter (even at 0.01%) is unacceptable (e.g., security-critical deduplication). $\square$
