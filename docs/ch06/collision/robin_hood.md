# Robin Hood Hashing

Standard linear probing has a fairness problem: some keys land on their first probe while others travel far from their home slot, creating high variance in search times. Robin Hood hashing applies a simple equity principle --- "steal from the rich, give to the poor" --- by allowing a key with a longer probe distance to displace a key with a shorter one. The result is dramatically reduced variance in probe distances, making worst-case searches much faster in practice.

## Probe Distance

The **probe distance** (or **displacement**) of a stored key $k$ is the number of slots between its actual position and its ideal position $h(k)$:

$$
d(k) = \bigl(\text{pos}(k) - h(k)\bigr) \bmod m
$$

In standard linear probing, probe distances can vary widely. Robin Hood hashing maintains the invariant that probe distances along the table are roughly equalized.

## Insertion Algorithm

Robin Hood insertion uses linear probing as its base but adds a displacement rule:

1. Compute $h(k)$ for the new key $k$ and set $d \gets 0$.
2. Examine slot $(h(k) + d) \bmod m$.
3. If the slot is empty, place $k$ there.
4. If the slot is occupied by a key $k'$ with probe distance $d' < d$, **swap** $k$ and $k'$. Continue inserting $k'$ with its updated probe distance.
5. If $d' \ge d$, increment $d$ and continue probing.

The swap rule ensures that no key is ever "too lucky" at the expense of others. Formally, after all insertions the probe distances satisfy

$$
d_{\max} - d_{\min} \le 1 \quad \text{(approximately)}
$$

which makes Robin Hood hashing nearly optimal in terms of variance.

## Variance Reduction

For a table with load factor $\alpha = n/m$, the expected maximum probe distance under Robin Hood hashing is

$$
E[d_{\max}] = O(\log \log n)
$$

compared to $O(\log n)$ for standard linear probing. The variance of probe distances drops from $\Theta(1/(1-\alpha)^2)$ in standard linear probing to $\Theta(1/(1-\alpha))$ in Robin Hood hashing.

This means that while the **average** search time is the same as linear probing --- $\Theta(1/(1-\alpha))$ --- the **worst-case** search time is dramatically better. In practice, Robin Hood tables can operate at higher load factors (up to $\alpha \approx 0.9$) without severe performance degradation.

## Search and Deletion

**Search** for key $k$: probe sequentially from $h(k)$, tracking the probe distance $d$. If a slot is reached where the stored key has a probe distance less than $d$, the search key cannot be further along --- stop and report "not found." This **early termination** makes unsuccessful searches faster than in standard linear probing.

**Deletion**: Robin Hood hashing supports clean deletion via **backward shifting**. After removing a key, shift all subsequent keys in the cluster backward if doing so reduces their probe distance. This avoids tombstones entirely, maintaining the Robin Hood invariant.

## Comparison with Other Open Addressing Strategies

| Property | Linear probing | Robin Hood | Double hashing |
|---|---|---|---|
| Average search | $O(1/(1-\alpha))$ | $O(1/(1-\alpha))$ | $O(1/(1-\alpha))$ |
| Worst-case search | $O(\log n)$ expected | $O(\log \log n)$ expected | $O(\log n)$ expected |
| Probe variance | High | Low | Medium |
| Cache behavior | Excellent | Excellent | Moderate |
| Deletion | Tombstones | Backward shift | Tombstones |
| Implementation | Simple | Moderate | Moderate |

Robin Hood hashing combines the cache friendliness of linear probing with much better worst-case behavior, making it a popular choice in high-performance hash table implementations such as Rust's `HashMap` (prior to version 1.36).

## Python Implementation

```python
"""
Robin Hood hashing implementation.

Demonstrates the displacement-based insertion that equalizes
probe distances across stored keys.
"""


# === Robin Hood Hash Table ===

class RobinHoodHashTable:
    """Hash table using Robin Hood hashing with linear probing."""

    _EMPTY = None

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.keys = [self._EMPTY] * capacity
        self.values = [self._EMPTY] * capacity
        self.dists = [0] * capacity  # probe distances

    def _hash(self, key):
        return hash(key) % self.capacity

    def insert(self, key, value):
        """Insert with Robin Hood displacement."""
        idx = self._hash(key)
        dist = 0

        while True:
            if self.keys[idx] is self._EMPTY:
                self.keys[idx] = key
                self.values[idx] = value
                self.dists[idx] = dist
                self.size += 1
                return
            # Update existing key
            if self.keys[idx] == key:
                self.values[idx] = value
                return
            # Robin Hood swap: displace the "richer" key
            if self.dists[idx] < dist:
                key, self.keys[idx] = self.keys[idx], key
                value, self.values[idx] = self.values[idx], value
                dist, self.dists[idx] = self.dists[idx], dist

            idx = (idx + 1) % self.capacity
            dist += 1

    def search(self, key):
        """Search with early termination on probe distance."""
        idx = self._hash(key)
        dist = 0

        while self.keys[idx] is not self._EMPTY:
            if self.keys[idx] == key:
                return self.values[idx]
            # Early termination: key would have been placed here
            if self.dists[idx] < dist:
                return None
            idx = (idx + 1) % self.capacity
            dist += 1

        return None

    def delete(self, key):
        """Delete with backward shifting (no tombstones)."""
        idx = self._hash(key)
        dist = 0

        # Find the key
        while self.keys[idx] is not self._EMPTY:
            if self.keys[idx] == key:
                break
            if self.dists[idx] < dist:
                return False
            idx = (idx + 1) % self.capacity
            dist += 1
        else:
            return False

        # Backward shift subsequent entries
        self.keys[idx] = self._EMPTY
        self.values[idx] = self._EMPTY
        self.dists[idx] = 0
        self.size -= 1

        next_idx = (idx + 1) % self.capacity
        while (self.keys[next_idx] is not self._EMPTY
               and self.dists[next_idx] > 0):
            self.keys[idx] = self.keys[next_idx]
            self.values[idx] = self.values[next_idx]
            self.dists[idx] = self.dists[next_idx] - 1
            self.keys[next_idx] = self._EMPTY
            self.values[next_idx] = self._EMPTY
            self.dists[next_idx] = 0
            idx = next_idx
            next_idx = (next_idx + 1) % self.capacity

        return True

    def probe_distances(self):
        """Return list of (key, probe_distance) for occupied slots."""
        result = []
        for i in range(self.capacity):
            if self.keys[i] is not self._EMPTY:
                result.append((self.keys[i], self.dists[i]))
        return result


# === Demonstration ===

if __name__ == "__main__":
    ht = RobinHoodHashTable(capacity=8)

    for k, v in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]:
        ht.insert(k, v)

    print("Probe distances:", ht.probe_distances())
    print(f"search('c'): {ht.search('c')}")
    print(f"search('z'): {ht.search('z')}")

    ht.delete("c")
    print(f"After delete, search('c'): {ht.search('c')}")
    print("Probe distances:", ht.probe_distances())
```

**Output:**
```
Probe distances: [('a', 0), ('b', 0), ('d', 0), ('e', 0), ('c', 2)]
search('c'): 3
search('z'): None
After delete, search('c'): None
Probe distances: [('a', 0), ('b', 0), ('d', 0), ('e', 0)]
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Celis, P. "Robin Hood Hashing." PhD Thesis, University of Waterloo, 1986.
