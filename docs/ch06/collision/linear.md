# Linear Probing

Linear probing is the simplest open addressing strategy: when a collision occurs, the algorithm examines the next slot, then the next, and so on, wrapping around the table until an empty slot is found. This sequential access pattern makes linear probing extremely cache-friendly, which often makes it the fastest hash table strategy in practice despite its theoretical susceptibility to clustering.

## Probe Sequence

Given a hash function $h'(k)$ and a table of size $m$, linear probing defines the probe sequence

$$
h(k, i) = \bigl(h'(k) + i\bigr) \bmod m, \qquad i = 0, 1, 2, \ldots, m - 1
$$

The first probe goes to slot $h'(k)$, and each subsequent probe moves to the next consecutive slot modulo $m$. Because the step size is always 1, the probe sequence visits every slot exactly once before repeating.

## Primary Clustering

The main drawback of linear probing is **primary clustering**. When a contiguous block of occupied slots (a **cluster**) forms, any new key that hashes into any position within the cluster extends the cluster by one slot. Longer clusters are more likely to absorb new keys, causing them to grow even longer.

Formally, if a cluster has length $\ell$, the probability that the next insertion extends this cluster is proportional to

$$
\frac{\ell + 1}{m}
$$

because any key hashing to any of the $\ell$ slots within the cluster or the one slot immediately after it will be appended to the cluster. This positive feedback loop is the defining characteristic of primary clustering.

## Expected Number of Probes

Knuth's analysis (1963) gives the expected number of probes under the uniform hashing assumption for a table with load factor $\alpha = n/m$:

**Unsuccessful search (or insertion):**

$$
E[\text{probes}] \approx \frac{1}{2}\left(1 + \frac{1}{(1-\alpha)^2}\right)
$$

**Successful search:**

$$
E[\text{probes}] \approx \frac{1}{2}\left(1 + \frac{1}{1-\alpha}\right)
$$

These grow much faster than the corresponding formulas for uniform hashing ($\frac{1}{1-\alpha}$ and $\frac{1}{\alpha}\ln\frac{1}{1-\alpha}$), reflecting the cost of clustering:

| Load factor $\alpha$ | Unsuccessful (linear) | Unsuccessful (uniform) |
|---|---|---|
| 0.50 | 2.50 | 2.00 |
| 0.75 | 8.50 | 4.00 |
| 0.90 | 50.50 | 10.00 |
| 0.95 | 200.50 | 20.00 |

The practical takeaway: keep the load factor below $0.7$ for linear probing to remain efficient.

## Operations

**Insertion**: follow the probe sequence from $h'(k)$ until an empty slot or tombstone is found. Place the key there.

**Search**: follow the probe sequence from $h'(k)$ until the key is found or an empty slot is reached (indicating absence). Tombstones do not terminate the search.

**Deletion**: removing a key from an open-addressed table requires care. Two strategies exist:

1. **Tombstone**: mark the slot as deleted. Searches skip past tombstones, and insertions can reuse them. Over time, tombstones accumulate and degrade performance.
2. **Backward shift**: after removing a key, shift subsequent keys in the cluster backward to fill the gap. This avoids tombstones but requires more work per deletion.

## Cache Performance

Linear probing accesses memory sequentially. On modern hardware with cache lines of 64 bytes, a single cache miss can load 8 entries (assuming 8-byte keys), meaning the first probe fetches enough data for several subsequent probes. This makes linear probing significantly faster than chaining or double hashing in practice, especially for small keys.

## Python Implementation

```python
"""
Linear probing hash table implementation.

Demonstrates the simplest open addressing strategy with
sequential probing and tombstone-based deletion.
"""


# === Linear Probing Table ===

class LinearProbingTable:
    """Hash table using linear probing for collision resolution."""

    _EMPTY = None
    _DELETED = "<DELETED>"

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.table = [self._EMPTY] * capacity

    def _probe(self, key):
        """Generate the linear probe sequence."""
        start = hash(key) % self.capacity
        for i in range(self.capacity):
            yield (start + i) % self.capacity

    def insert(self, key, value):
        """Insert key-value pair using linear probing."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY or entry is self._DELETED:
                self.table[idx] = (key, value)
                self.size += 1
                return
            if entry[0] == key:
                self.table[idx] = (key, value)  # update
                return
        raise RuntimeError("Hash table is full")

    def search(self, key):
        """Search for key, return value or None."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY:
                return None
            if entry is not self._DELETED and entry[0] == key:
                return entry[1]
        return None

    def delete(self, key):
        """Delete key using tombstone marker."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY:
                return False
            if entry is not self._DELETED and entry[0] == key:
                self.table[idx] = self._DELETED
                self.size -= 1
                return True
        return False

    def display(self):
        """Show table contents for inspection."""
        for i, entry in enumerate(self.table):
            status = "empty" if entry is self._EMPTY else (
                "deleted" if entry is self._DELETED else f"{entry}")
            print(f"  [{i}] {status}")


# === Demonstration ===

if __name__ == "__main__":
    ht = LinearProbingTable(capacity=8)

    for k, v in [("cat", 1), ("dog", 2), ("rat", 3), ("bat", 4)]:
        ht.insert(k, v)

    print("Table after insertions:")
    ht.display()
    print(f"\nsearch('dog'): {ht.search('dog')}")
    print(f"search('fox'): {ht.search('fox')}")

    ht.delete("dog")
    print(f"\nAfter deleting 'dog':")
    print(f"search('dog'): {ht.search('dog')}")
    print(f"search('rat'): {ht.search('rat')}")
```

**Output:**
```
Table after insertions:
  [0] ('bat', 4)
  [1] empty
  [2] ('cat', 1)
  [3] ('dog', 2)
  [4] ('rat', 3)
  [5] empty
  [6] empty
  [7] empty

search('dog'): 2
search('fox'): None

After deleting 'dog':
search('dog'): None
search('rat'): 3
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Knuth, D. E. *The Art of Computer Programming*, Vol. 3: Sorting and Searching.
