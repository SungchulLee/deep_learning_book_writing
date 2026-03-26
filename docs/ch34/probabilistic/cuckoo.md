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
