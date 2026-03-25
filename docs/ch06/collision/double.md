# Double Hashing

Linear probing suffers from primary clustering and quadratic probing from secondary clustering. Double hashing eliminates both by using a second hash function to determine the probe step size, producing probe sequences that depend on the key itself. This makes double hashing one of the best open addressing strategies in practice.

## Probe Sequence

Double hashing uses two hash functions $h_1$ and $h_2$. The probe sequence for key $k$ is

$$
h(k, i) = \bigl(h_1(k) + i \cdot h_2(k)\bigr) \bmod m, \qquad i = 0, 1, 2, \ldots
$$

where $m$ is the table size. The first probe goes to $h_1(k)$; subsequent probes advance by $h_2(k)$ each time. Because the step size depends on $k$, different keys that collide at the same initial slot follow different probe sequences.

## Requirements on the Second Hash Function

For double hashing to examine every slot in the table, $h_2(k)$ must satisfy two conditions:

1. **Non-zero**: $h_2(k) \neq 0$ for all keys $k$, otherwise the probe sequence stays at the initial slot.
2. **Coprime to** $m$: the value $h_2(k)$ must be relatively prime to $m$ so that the probe sequence generates a full permutation of $\{0, 1, \ldots, m-1\}$.

A common strategy to guarantee both conditions is to choose $m$ to be prime and define

$$
h_2(k) = 1 + (k \bmod (m - 1))
$$

Since $m$ is prime and $1 \le h_2(k) \le m - 1$, we have $\gcd(h_2(k), m) = 1$ for every key.

An alternative is to set $m = 2^p$ (a power of two for fast modular arithmetic) and ensure $h_2(k)$ always returns an odd value:

$$
h_2(k) = 2 \cdot (k \bmod \lfloor m/2 \rfloor) + 1
$$

## Why Double Hashing Avoids Clustering

**Primary clustering** (linear probing): keys with different hash values that collide at the same slot follow identical subsequent probe sequences.

**Secondary clustering** (quadratic probing): keys with the same $h_1(k)$ follow the same probe sequence because the quadratic offset depends only on the probe number $i$, not on the key.

**Double hashing**: the step size $h_2(k)$ varies per key, so even keys that collide at $h_1(k)$ diverge immediately at the next probe. The number of distinct probe sequences is

$$
\Theta(m^2)
$$

compared to $\Theta(m)$ for linear or quadratic probing.

## Expected Number of Probes

Under the uniform hashing assumption, the expected number of probes for a table with load factor $\alpha = n/m$ is:

**Unsuccessful search:**

$$
E[\text{probes}] \le \frac{1}{1 - \alpha}
$$

**Successful search:**

$$
E[\text{probes}] \le \frac{1}{\alpha} \ln \frac{1}{1 - \alpha}
$$

These formulas are the same as the ideal uniform hashing bounds, which shows that double hashing closely approximates the theoretical optimum.

| Load factor $\alpha$ | Unsuccessful probes | Successful probes |
|---|---|---|
| 0.50 | 2.0 | 1.39 |
| 0.75 | 4.0 | 1.85 |
| 0.90 | 10.0 | 2.56 |
| 0.95 | 20.0 | 3.15 |

## Insertion and Deletion

**Insertion**: follow the probe sequence $h(k, 0), h(k, 1), \ldots$ until an empty slot or a tombstone is found, then place the key there.

**Search**: follow the same probe sequence until the key is found, an empty slot is reached (key absent), or all $m$ slots have been examined.

**Deletion**: as with all open addressing methods, simple removal breaks probe chains. A **tombstone** (deleted marker) must be placed so that searches continue past the deleted slot. Tombstones are reusable during insertion but still count as occupied during search.

## Python Implementation

```python
"""
Double hashing implementation for open addressing.

Uses two hash functions to compute probe sequences that
avoid both primary and secondary clustering.
"""


# === Double Hashing Table ===

class DoubleHashTable:
    """Open-addressed hash table using double hashing."""

    _EMPTY = None
    _DELETED = "<DELETED>"

    def __init__(self, capacity=11):
        # Capacity should be prime for best results
        self.capacity = capacity
        self.size = 0
        self.table = [self._EMPTY] * capacity

    def _h1(self, key):
        return hash(key) % self.capacity

    def _h2(self, key):
        # Ensure non-zero and coprime to capacity (capacity is prime)
        return 1 + (hash(key) % (self.capacity - 1))

    def _probe(self, key):
        """Generate the probe sequence for the given key."""
        start = self._h1(key)
        step = self._h2(key)
        for i in range(self.capacity):
            yield (start + i * step) % self.capacity

    def insert(self, key, value):
        """Insert key-value pair using double hashing."""
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


# === Demonstration ===

if __name__ == "__main__":
    ht = DoubleHashTable(capacity=11)

    data = [("alice", 100), ("bob", 200), ("carol", 300),
            ("dave", 400), ("eve", 500)]
    for k, v in data:
        ht.insert(k, v)

    print(f"Size: {ht.size}")
    for k, _ in data:
        print(f"search('{k}'): {ht.search(k)}")

    ht.delete("carol")
    print(f"After delete, search('carol'): {ht.search('carol')}")
    print(f"search('dave'): {ht.search('dave')}")  # still reachable
```

**Output:**
```
Size: 5
search('alice'): 100
search('bob'): 200
search('carol'): 300
search('dave'): 400
search('eve'): 500
After delete, search('carol'): None
search('dave'): 400
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
