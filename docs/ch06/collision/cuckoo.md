# Cuckoo Hashing

Most collision resolution strategies accept that lookup may degrade to $O(n)$ in the worst case. Cuckoo hashing takes a fundamentally different approach: it guarantees $O(1)$ worst-case lookup and deletion by using two hash functions and displacing existing keys when a collision occurs --- much like a cuckoo bird evicts other eggs from a nest.

## Two-Table Scheme

Cuckoo hashing maintains two tables $T_1$ and $T_2$, each of size $m$, with independent hash functions $h_1$ and $h_2$. Every key $k$ resides in exactly one of two possible locations:

$$
T_1[h_1(k)] \quad \text{or} \quad T_2[h_2(k)]
$$

Because each key has at most two candidate positions and each position holds at most one key, lookups and deletions inspect exactly two slots.

## Lookup and Deletion

**Lookup** for key $k$: check $T_1[h_1(k)]$ and $T_2[h_2(k)]$. If either contains $k$, return it; otherwise report "not found." This runs in $O(1)$ worst-case time.

**Deletion** of key $k$: locate $k$ with the same two-slot check and remove it. Again $O(1)$ worst case.

## Insertion Algorithm

Insertion is where cuckoo hashing becomes interesting. To insert key $k$:

1. If $T_1[h_1(k)]$ is empty, place $k$ there and return.
2. Otherwise, evict the current occupant $k'$ from $T_1[h_1(k)]$ and place $k$ there.
3. Try to place $k'$ in $T_2[h_2(k')]$. If that slot is empty, place $k'$ and return.
4. Otherwise, evict the occupant from $T_2[h_2(k')]$ and repeat the displacement process.
5. If displacements exceed a threshold (typically $O(\log n)$ steps), declare a **cycle** and **rehash** --- choose new hash functions $h_1, h_2$ and reinsert all keys.

The displacement chain can be visualized as a sequence of "kicks":

$$
k \xrightarrow{\text{evicts}} k_1 \xrightarrow{\text{evicts}} k_2 \xrightarrow{\text{evicts}} \cdots
$$

## Cycle Detection and Rehashing

A displacement cycle occurs when the eviction chain returns to a key that was already displaced. With random hash functions and a load factor below $50\%$, cycles are rare. When one is detected, the entire table is rebuilt with fresh hash functions.

The maximum displacement chain length before triggering a rehash is typically set to

$$
\text{MaxKicks} = c \cdot \log n
$$

for a small constant $c$, which balances between unnecessary rehashes and excessively long insertion chains.

## Analysis

Under the assumption that $h_1$ and $h_2$ are drawn from a universal hash family:

| Operation | Time |
|---|---|
| Lookup | $O(1)$ worst case |
| Delete | $O(1)$ worst case |
| Insert | $O(1)$ amortized expected |

The expected amortized cost of insertion is $O(1)$ when the load factor satisfies

$$
\alpha = \frac{n}{2m} < \frac{1}{2}
$$

where $n$ is the number of stored keys and $2m$ is the total number of slots across both tables. Above this threshold, the probability of cycles increases sharply.

The space utilization of basic cuckoo hashing is at most $50\%$, which is less efficient than linear probing or chaining. Extensions such as **bucketized cuckoo hashing** (multiple slots per bucket) can achieve load factors above $90\%$.

## Comparison with Other Strategies

| Property | Chaining | Linear probing | Cuckoo |
|---|---|---|---|
| Worst-case lookup | $O(n)$ | $O(n)$ | $O(1)$ |
| Space overhead | Pointer per entry | None | Two tables |
| Maximum load factor | Unbounded | ${\sim}70\%$ practical | ${\sim}50\%$ basic |
| Cache behavior | Poor | Excellent | Moderate |

The $O(1)$ worst-case guarantee makes cuckoo hashing attractive for real-time systems and hardware implementations where predictable latency is critical.

## Python Implementation

```python
"""
Cuckoo hashing implementation with two tables.

Demonstrates the displacement-based insertion that guarantees
O(1) worst-case lookup and deletion.
"""


# === Cuckoo Hash Table ===

class CuckooHashTable:
    """Hash table using cuckoo hashing with two independent tables."""

    MAX_KICKS = 50  # displacement limit before rehash

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.table1 = [None] * capacity
        self.table2 = [None] * capacity
        self._seed1 = 0
        self._seed2 = 1

    def _h1(self, key):
        return hash((key, self._seed1)) % self.capacity

    def _h2(self, key):
        return hash((key, self._seed2)) % self.capacity

    def lookup(self, key):
        """O(1) worst-case lookup."""
        pos1 = self._h1(key)
        if self.table1[pos1] is not None and self.table1[pos1][0] == key:
            return self.table1[pos1][1]
        pos2 = self._h2(key)
        if self.table2[pos2] is not None and self.table2[pos2][0] == key:
            return self.table2[pos2][1]
        return None

    def delete(self, key):
        """O(1) worst-case deletion."""
        pos1 = self._h1(key)
        if self.table1[pos1] is not None and self.table1[pos1][0] == key:
            self.table1[pos1] = None
            self.size -= 1
            return True
        pos2 = self._h2(key)
        if self.table2[pos2] is not None and self.table2[pos2][0] == key:
            self.table2[pos2] = None
            self.size -= 1
            return True
        return False

    def insert(self, key, value):
        """Insert with cuckoo displacement. Rehashes on cycle."""
        # Check if key already exists
        if self.lookup(key) is not None:
            self.delete(key)

        item = (key, value)
        for _ in range(self.MAX_KICKS):
            # Try table 1
            pos1 = self._h1(item[0])
            if self.table1[pos1] is None:
                self.table1[pos1] = item
                self.size += 1
                return
            # Evict from table 1
            item, self.table1[pos1] = self.table1[pos1], item

            # Try table 2
            pos2 = self._h2(item[0])
            if self.table2[pos2] is None:
                self.table2[pos2] = item
                self.size += 1
                return
            # Evict from table 2
            item, self.table2[pos2] = self.table2[pos2], item

        # Cycle detected — rehash with new seeds
        self._rehash(item)

    def _rehash(self, pending_item):
        """Rebuild both tables with new hash functions."""
        self._seed1 += 2
        self._seed2 += 2
        old_items = []
        for i in range(self.capacity):
            if self.table1[i] is not None:
                old_items.append(self.table1[i])
                self.table1[i] = None
            if self.table2[i] is not None:
                old_items.append(self.table2[i])
                self.table2[i] = None
        self.size = 0
        for k, v in old_items:
            self.insert(k, v)
        self.insert(pending_item[0], pending_item[1])


# === Demonstration ===

if __name__ == "__main__":
    ct = CuckooHashTable(capacity=8)

    keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    for i, key in enumerate(keys):
        ct.insert(key, i + 1)

    for key in keys:
        print(f"lookup('{key}'): {ct.lookup(key)}")

    ct.delete("gamma")
    print(f"After delete, lookup('gamma'): {ct.lookup('gamma')}")
    print(f"Size: {ct.size}")
```

**Output:**
```
lookup('alpha'): 1
lookup('beta'): 2
lookup('gamma'): 3
lookup('delta'): 4
lookup('epsilon'): 5
After delete, lookup('gamma'): None
Size: 4
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Pagh, R. and Rodler, F. F. "Cuckoo Hashing." *Journal of Algorithms*, 51(2), 2004.
