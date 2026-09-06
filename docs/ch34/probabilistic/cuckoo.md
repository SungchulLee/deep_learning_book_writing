# Cuckoo Filters

Bloom filters are space-efficient but do not support deletion (without counting variants) and cannot report the items stored. **Cuckoo filters** address both limitations by storing compact fingerprints in a cuckoo hash table. They support insertion, deletion, and membership queries with better space efficiency than counting Bloom filters and comparable false positive rates.

## Structure

A cuckoo filter consists of:

- A hash table with $m$ buckets, each holding up to $b$ entries (typically $b = 4$).
- Each entry stores a **fingerprint** $f(x)$ of $f$ bits (not the full element $x$).
- Two hash functions that determine the two candidate buckets for each element.

The key insight is **partial-key cuckoo hashing**: given one bucket index and the fingerprint, the alternate bucket can be computed without knowing the original element:

$$
h_1(x) = \text{hash}(x)
$$

$$
h_2(x) = h_1(x) \oplus \text{hash}(f(x))
$$

This XOR relationship means $h_1(x) = h_2(x) \oplus \text{hash}(f(x))$, so the alternate bucket can be found from any bucket using only the stored fingerprint.

## Operations

**Insert($x$)**: Compute $f = f(x)$, $i_1 = h_1(x)$, $i_2 = h_2(x)$.

1. If bucket $i_1$ or $i_2$ has an empty slot, store $f$ there.
2. Otherwise, pick one bucket (say $i_1$), evict a random entry $f'$, store $f$ in its place.
3. Relocate $f'$ to its alternate bucket. If that bucket is also full, repeat eviction (up to a maximum number of kicks).
4. If the maximum kicks are exceeded, declare the table full and trigger a resize.

**Query($x$)**: Compute $f = f(x)$, $i_1 = h_1(x)$, $i_2 = h_2(x)$. Return `True` if $f$ appears in bucket $i_1$ or $i_2$.

**Delete($x$)**: Compute $f$, $i_1$, $i_2$. If $f$ is found in bucket $i_1$ or $i_2$, remove one copy.

## False Positive Analysis

A false positive occurs when a non-member's fingerprint matches a stored fingerprint in one of its two candidate buckets. With fingerprints of $f$ bits and bucket size $b$:

$$
P_{\text{fp}} \le \frac{2b}{2^f}
$$

For $b = 4$ and $f = 12$ bits: $P_{\text{fp}} \le 8/4096 \approx 0.2\%$.

## Space Efficiency

Each element requires $f$ bits of storage. The total space for $n$ elements at load factor $\alpha$ is:

$$
\text{bits per element} = \frac{f}{\alpha}
$$

With $b = 4$ buckets and semi-sorted buckets (encoding optimization), cuckoo filters achieve approximately $\alpha = 0.95$ and use fewer bits per element than Bloom filters for false positive rates below $3\%$.

!!! tip "When to prefer cuckoo filters"
    Use cuckoo filters over Bloom filters when you need deletion support, better space efficiency at low false positive rates (below 3%), or faster lookups (only two memory accesses vs $k$ for Bloom filters).

## Implementation

```python
"""
Cuckoo Filter -- space-efficient probabilistic set with deletion.

Stores fingerprints in a cuckoo hash table with partial-key
cuckoo hashing for bucket relocation.
"""

import hashlib
import random


# === Cuckoo Filter ============================================================

class CuckooFilter:
    """Probabilistic set supporting insert, delete, and query."""

    MAX_KICKS = 500

    def __init__(self, capacity: int, bucket_size: int = 4,
                 fingerprint_bits: int = 8):
        self.bucket_size = bucket_size
        self.fp_bits = fingerprint_bits
        self.fp_mask = (1 << fingerprint_bits) - 1
        self.num_buckets = max(1, capacity // bucket_size)
        self.buckets: list[list[int]] = [[] for _ in range(self.num_buckets)]
        self.count = 0

    def _fingerprint(self, item: str) -> int:
        """Compute a non-zero fingerprint."""
        h = int(hashlib.sha256(item.encode()).hexdigest(), 16)
        fp = (h & self.fp_mask) or 1  # ensure non-zero
        return fp

    def _hash(self, item: str) -> int:
        """Primary bucket index."""
        h = int(hashlib.md5(item.encode()).hexdigest(), 16)
        return h % self.num_buckets

    def _alt_index(self, index: int, fingerprint: int) -> int:
        """Alternate bucket via XOR with hashed fingerprint."""
        fp_hash = hash(fingerprint) % self.num_buckets
        return (index ^ fp_hash) % self.num_buckets

    def insert(self, item: str) -> bool:
        """Insert *item*. Returns False if the table is full."""
        fp = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._alt_index(i1, fp)

        if len(self.buckets[i1]) < self.bucket_size:
            self.buckets[i1].append(fp)
            self.count += 1
            return True
        if len(self.buckets[i2]) < self.bucket_size:
            self.buckets[i2].append(fp)
            self.count += 1
            return True

        # Eviction loop
        idx = random.choice([i1, i2])
        for _ in range(self.MAX_KICKS):
            evict_pos = random.randrange(len(self.buckets[idx]))
            fp, self.buckets[idx][evict_pos] = (
                self.buckets[idx][evict_pos], fp
            )
            idx = self._alt_index(idx, fp)
            if len(self.buckets[idx]) < self.bucket_size:
                self.buckets[idx].append(fp)
                self.count += 1
                return True

        return False  # table is full

    def query(self, item: str) -> bool:
        """Test whether *item* is possibly in the set."""
        fp = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._alt_index(i1, fp)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, item: str) -> bool:
        """Remove *item*. Returns False if not found."""
        fp = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._alt_index(i1, fp)
        if fp in self.buckets[i1]:
            self.buckets[i1].remove(fp)
            self.count -= 1
            return True
        if fp in self.buckets[i2]:
            self.buckets[i2].remove(fp)
            self.count -= 1
            return True
        return False


# === Main =====================================================================

if __name__ == "__main__":
    cf = CuckooFilter(capacity=100, fingerprint_bits=12)

    for word in ["apple", "banana", "cherry"]:
        cf.insert(word)

    print("After inserting apple, banana, cherry:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cf.query(word)}")

    cf.delete("banana")
    print("\nAfter deleting banana:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cf.query(word)}")
```

**Output:**

```
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

Deletion works correctly: removing `banana` does not affect `apple` or `cherry`, demonstrating the advantage of fingerprint-based storage over bit arrays.

## Reference

- Fan, B., Andersen, D.G., Kaminsky, M., and Mitzenmacher, M. "Cuckoo Filter: Practically Better Than Bloom." *CoNEXT*, 2014
- Pagh, R. and Rodler, F.F. "Cuckoo Hashing." *ESA*, 2001

## Exercises

**Exercise 1.**
Describe the structure of a cuckoo filter. How are elements stored, and how does a membership query work?

??? success "Solution to Exercise 1"
    A cuckoo filter stores **fingerprints** (compact hashes) of elements in a cuckoo hash table with $m$ buckets, each holding $b$ slots (typically $b = 4$). For element $x$: compute fingerprint $f = \text{fingerprint}(x)$ and two candidate bucket indices $i_1 = h(x)$ and $i_2 = i_1 \oplus h(f)$ (XOR-based alternate location). Insert $f$ into an available slot in $i_1$ or $i_2$. If both are full, evict a random existing fingerprint and relocate it to its alternate bucket, repeating up to a maximum number of kicks. A membership query computes $f$ and checks both $i_1$ and $i_2$: if $f$ is found in either bucket, return "present" (possibly a false positive); otherwise, return "absent." False positives occur when a different element's fingerprint collides with $f$ in one of the two candidate buckets. $\square$

---

**Exercise 2.**
Explain the "partial-key cuckoo hashing" trick that allows computing the alternate bucket location using only the fingerprint, without storing the original key.

??? success "Solution to Exercise 2"
    In standard cuckoo hashing, the two bucket locations are computed from the full key: $i_1 = h_1(x)$, $i_2 = h_2(x)$. During eviction, relocating a fingerprint requires knowing the original key (to compute the alternate location), which the filter does not store. Partial-key cuckoo hashing resolves this by defining $i_2 = i_1 \oplus h(f)$, where $f$ is the stored fingerprint. This relationship is symmetric: $i_1 = i_2 \oplus h(f)$. Given a fingerprint $f$ currently at bucket $i$, its alternate bucket is $i \oplus h(f)$ -- computable from $i$ and $f$ alone, without the original key. This trick makes eviction possible in the filter setting. The constraint is that the fingerprint must be sufficiently random for $h(f)$ to distribute evicted items across the table uniformly. $\square$

---

**Exercise 3.**
A cuckoo filter with 8-bit fingerprints and bucket size 4 is loaded to 95% capacity with $n = 10^6$ elements. Compute the false positive rate and total memory usage.

??? success "Solution to Exercise 3"
    With 8-bit fingerprints, two elements collide in a given bucket position with probability $1/2^8 = 1/256$. A query checks two buckets of size 4 each, so it examines up to $2b = 8$ fingerprints. The FPR is approximately $2b / 2^f = 8 / 256 = 3.125\%$ where $f = 8$ is the fingerprint size. At 95% load, the number of buckets is $n / (b \times 0.95) = 10^6 / (4 \times 0.95) \approx 263{,}158$. Total memory: $263{,}158 \times 4 \times 8$ bits $= 8{,}421{,}056$ bits $\approx 1.03$ MB, or about 8.4 bits per element. For comparison, a standard Bloom filter at 3.125% FPR uses $-(n \ln 0.03125) / (\ln 2)^2 \approx 7.2 \times 10^6$ bits $= 0.88$ MB (7.2 bits/element). The cuckoo filter uses slightly more space but supports deletion. $\square$

---

**Exercise 4.**
Describe the deletion operation in a cuckoo filter. Under what circumstances can deletion cause false negatives?

??? success "Solution to Exercise 4"
    Deletion: compute $f = \text{fingerprint}(x)$ and check buckets $i_1$ and $i_2$. If $f$ is found in either bucket, remove one copy. If not found, the element was not present (or was already deleted). False negatives can occur if two different elements $x$ and $y$ have the same fingerprint $f$ and share a bucket. If $x$ is inserted first and $y$ is inserted second (both with fingerprint $f$ in overlapping bucket $i$), and then $x$ is deleted, the deletion removes one copy of $f$ from bucket $i$. If it happens to remove the copy that $y$ also relies on (because they share the exact same fingerprint and bucket), a subsequent query for $y$ finds $f$ missing and returns "absent" -- a false negative. This can only happen when two elements have identical fingerprints and overlap in a candidate bucket. The probability is very low (proportional to $1/2^f$) but non-zero. $\square$

---

**Exercise 5.**
Compare cuckoo filters, Bloom filters, and counting Bloom filters across four dimensions: space per element, deletion support, lookup speed, and worst-case insert time.

??? success "Solution to Exercise 5"
    | Dimension | Bloom | Counting Bloom | Cuckoo |
    |---|---|---|---|
    | Space (1% FPR) | 9.6 bits/elem | 38.4 bits/elem | 12.6 bits/elem |
    | Deletion | No | Yes | Yes |
    | Lookup | $k$ random reads | $k$ random reads | 2 sequential reads |
    | Insert (worst) | $O(k)$ | $O(k)$ | $O(1/\epsilon)$ amortized |

    Bloom filters are the most space-efficient and have predictable insertion cost but lack deletion. Counting Bloom filters support deletion at 4x space cost. Cuckoo filters provide the best combination: moderate space, native deletion, and cache-friendly lookups (two buckets vs. $k$ scattered positions). However, cuckoo filter insertion can fail at high load factors, requiring table resizing, while Bloom filters never fail to insert. For write-heavy workloads at high load, Bloom filters are more predictable. $\square$
