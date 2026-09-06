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

## Exercises

**Exercise 1.**
Explain why deletion is impossible in a standard Bloom filter and how counting Bloom filters solve this. What is the space overhead?

??? success "Solution to Exercise 1"
    In a standard Bloom filter, each bit position may be set by multiple elements. Clearing a bit during deletion would erase evidence of other elements that also hash to that position, potentially creating false negatives (which violate the Bloom filter guarantee). Counting Bloom filters replace each bit with an integer counter. Insert increments all $k$ counters; delete decrements all $k$ counters. A membership query checks whether all $k$ counters are $> 0$. Since counters are never set to negative values (assuming no bogus deletes), elements that were not deleted retain positive counts at all their positions. Space overhead: each counter needs $b$ bits instead of 1 bit. With 4-bit counters ($b = 4$, maximum count 15), the space is $4m$ bits instead of $m$ bits -- a 4x increase. The 4-bit choice is standard because the probability of a counter exceeding 15 is negligibly small for typical loads. $\square$

---

**Exercise 2.**
Derive the probability that a counter in a counting Bloom filter overflows (exceeds its maximum value) given $n$ elements, $m$ counters, $k$ hash functions, and counter width $b$ bits.

??? success "Solution to Exercise 2"
    Each insertion increments $k$ of the $m$ counters. The number of times a specific counter is incremented follows a Binomial distribution: $X \sim \text{Binomial}(nk, 1/m)$, since each of the $nk$ hash outputs independently hits this counter with probability $1/m$. The expected count is $\mu = nk/m$. The counter overflows if $X > 2^b - 1$. With $b = 4$, overflow occurs when $X > 15$. For $k = 7$ and $m/n = 10$ (optimal for 1% FPR): $\mu = 7/10 = 0.7$. Using a Poisson approximation: $P(X > 15) \approx \sum_{i=16}^{\infty} e^{-0.7} (0.7)^i / i! < 10^{-15}$. Over $m = 10^7$ counters, the expected number of overflows is $< 10^{-8}$, making overflow essentially impossible. $\square$

---

**Exercise 3.**
What happens if an element is deleted from a counting Bloom filter that was never inserted? Describe the failure mode and propose a safeguard.

??? success "Solution to Exercise 3"
    Deleting a non-member decrements $k$ counters that may have been incremented by other elements. This reduces those counters, potentially to 0, which creates false negatives for the elements that originally incremented them. This is a correctness violation: the no-false-negatives guarantee is broken. Example: elements A and B both hash to counter $i$. Delete non-member C, which also hashes to counter $i$. Counter $i$ decreases; if it reaches 0, both A and B produce false negatives. Safeguard: before deleting, query the filter to verify the element is (probably) present. If the query returns "not present," skip the deletion. This prevents most bogus deletes but is not foolproof (a false positive could cause an incorrect delete of a non-member). For guaranteed correctness, maintain a separate exact set alongside the counting Bloom filter. $\square$

---

**Exercise 4.**
Compare counting Bloom filters with cuckoo filters in terms of space efficiency, deletion support, and false positive rate. When is each preferable?

??? success "Solution to Exercise 4"
    **Counting Bloom filter**: 4x space of a standard Bloom filter (4-bit counters). Supports deletion. FPR depends on $m/n$ and $k$. For 1% FPR: $\sim$40 bits/element. **Cuckoo filter**: stores fingerprints in a cuckoo hash table. For 1% FPR with 8-bit fingerprints and 95% load factor: $\sim$8.5 bits/element. Supports deletion natively (remove the fingerprint). Counting Bloom filters are preferable when: (1) the system already uses Bloom filters and a drop-in replacement is needed; (2) the number of hash functions must be configurable. Cuckoo filters are preferable when: (1) space efficiency matters (4--5x more compact); (2) deletion is needed (inherent in the design, no counter overflow risk); (3) lookup performance matters (cuckoo filters check only 2 buckets vs. $k$ random positions for Bloom). $\square$

---

**Exercise 5.**
Implement a counting Bloom filter in pseudocode that supports insert, delete, and query operations. Include the counter overflow check.

??? success "Solution to Exercise 5"
    ```
    class CountingBloomFilter:
        init(m, k):
            counters = array of m integers, all 0
            hash_functions = k independent hash functions
    
        insert(x):
            for i in 1..k:
                pos = hash_functions[i](x) % m
                if counters[pos] < MAX_COUNT:
                    counters[pos] += 1
    
        delete(x):
            if not query(x):
                return  # safeguard against bogus deletes
            for i in 1..k:
                pos = hash_functions[i](x) % m
                if counters[pos] > 0:
                    counters[pos] -= 1
    
        query(x):
            for i in 1..k:
                pos = hash_functions[i](x) % m
                if counters[pos] == 0:
                    return False
            return True
    ```
    The overflow check in `insert` caps counters at `MAX_COUNT` (e.g., 15 for 4-bit counters). The delete safeguard queries first and the `> 0` check prevents underflow. $\square$
