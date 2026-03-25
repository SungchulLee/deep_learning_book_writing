# Quadratic Probing

Linear probing suffers from primary clustering because the probe step is always 1, causing occupied regions to grow contiguously. Quadratic probing addresses this by using a quadratic function of the probe number as the offset, spreading probes across the table and breaking up clusters. The trade-off is that quadratic probing introduces a milder form of clustering called **secondary clustering** and does not guarantee visiting every slot.

## Probe Sequence

For a hash function $h'(k)$ and table size $m$, the quadratic probe sequence is

$$
h(k, i) = \bigl(h'(k) + c_1 i + c_2 i^2\bigr) \bmod m, \qquad i = 0, 1, 2, \ldots
$$

where $c_1$ and $c_2$ are constants with $c_2 \neq 0$. A common choice is $c_1 = 0$ and $c_2 = 1$, giving the simpler form

$$
h(k, i) = \bigl(h'(k) + i^2\bigr) \bmod m
$$

Another popular variant uses alternating signs: the $i$-th probe checks $h'(k) + \lceil i/2 \rceil^2 \cdot (-1)^i$.

## Secondary Clustering

Two distinct keys $k_1 \neq k_2$ that collide at the same initial slot ($h'(k_1) = h'(k_2)$) follow the **same** quadratic probe sequence, because the offset depends only on $i$, not on the key. This phenomenon is called **secondary clustering**.

Secondary clustering is less severe than primary clustering because keys with different initial hash values probe entirely different sequences. Since there are $m$ possible initial hash values, quadratic probing produces $m$ distinct probe sequences --- the same count as linear probing, but with better spread.

## Coverage Guarantee

Unlike linear probing, a quadratic probe sequence does not necessarily visit all $m$ slots. However, if the table size $m$ is prime and the table is at most half full, the first $\lfloor m/2 \rfloor$ probes are guaranteed to hit distinct slots.

!!! note "Half-table coverage theorem"
    If $m$ is prime and $\alpha \le 1/2$, then the quadratic probe sequence $h'(k) + i^2 \pmod{m}$ for $i = 0, 1, \ldots, \lfloor m/2 \rfloor$ produces $\lfloor m/2 \rfloor + 1$ distinct values.

??? note "Proof sketch"
    Suppose $h'(k) + i^2 \equiv h'(k) + j^2 \pmod{m}$ for $0 \le i < j \le \lfloor m/2 \rfloor$. Then $i^2 \equiv j^2 \pmod{m}$, so $m \mid (j-i)(j+i)$. Since $m$ is prime and $0 < j - i < m$ and $0 < j + i < m$, neither factor is divisible by $m$ --- a contradiction. Therefore all $\lfloor m/2 \rfloor + 1$ probes are distinct.

An alternative that guarantees full coverage is to use $m = 2^p$ with the triangular number probe sequence: slot offsets $0, 1, 3, 6, 10, \ldots$ (i.e., $i(i+1)/2$ for $i = 0, 1, \ldots, m-1$), which produces a permutation of all $m$ indices.

## Expected Performance

Quadratic probing performance falls between linear probing (which clusters badly) and double hashing (which approximates uniform hashing). No simple closed-form formula exists for the expected probe count, but empirical results show:

| Load factor $\alpha$ | Quadratic (unsuccessful) | Linear (unsuccessful) | Double (unsuccessful) |
|---|---|---|---|
| 0.50 | ${\sim}2.2$ | 2.5 | 2.0 |
| 0.75 | ${\sim}4.6$ | 8.5 | 4.0 |
| 0.90 | ${\sim}11.4$ | 50.5 | 10.0 |

Quadratic probing is closer to the uniform hashing ideal than linear probing for all load factors.

## Python Implementation

```python
"""
Quadratic probing hash table implementation.

Demonstrates collision resolution using quadratic offsets
to mitigate primary clustering.
"""


# === Quadratic Probing Table ===

class QuadraticProbingTable:
    """Hash table using quadratic probing: h(k,i) = (h'(k) + i^2) mod m."""

    _EMPTY = None
    _DELETED = "<DELETED>"

    def __init__(self, capacity=11):
        # Prime capacity ensures first m/2 probes hit distinct slots
        self.capacity = capacity
        self.size = 0
        self.table = [self._EMPTY] * capacity

    def _probe(self, key):
        """Generate the quadratic probe sequence."""
        start = hash(key) % self.capacity
        for i in range(self.capacity):
            yield (start + i * i) % self.capacity

    def insert(self, key, value):
        """Insert key-value pair using quadratic probing."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY or entry is self._DELETED:
                self.table[idx] = (key, value)
                self.size += 1
                return
            if entry[0] == key:
                self.table[idx] = (key, value)  # update
                return
        raise RuntimeError("Could not insert — table may be too full")

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

    def load_factor(self):
        """Return the current load factor."""
        return self.size / self.capacity


# === Demonstration ===

if __name__ == "__main__":
    ht = QuadraticProbingTable(capacity=11)

    words = ["hello", "world", "foo", "bar", "baz"]
    for i, w in enumerate(words):
        ht.insert(w, i * 10)

    print(f"Load factor: {ht.load_factor():.2f}")
    for w in words:
        print(f"search('{w}'): {ht.search(w)}")

    ht.delete("foo")
    print(f"After delete, search('foo'): {ht.search('foo')}")
    print(f"search('bar'): {ht.search('bar')}")  # still reachable
```

**Output:**
```
Load factor: 0.45
search('hello'): 0
search('world'): 10
search('foo'): 20
search('bar'): 30
search('baz'): 40
After delete, search('foo'): None
search('bar'): 30
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
