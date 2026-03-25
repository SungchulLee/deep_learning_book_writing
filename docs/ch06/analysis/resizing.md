# Resizing and Rehashing

A hash table with a fixed number of slots faces a dilemma: allocate too few slots and operations become slow as chains grow long; allocate too many and memory is wasted. **Dynamic resizing** solves this by adjusting the table size as the number of stored elements changes. When the load factor $\alpha = n/m$ exceeds an upper threshold, the table grows. When $\alpha$ drops below a lower threshold, the table shrinks. Each resize requires **rehashing** -- recomputing the hash of every stored element under the new table size.

## Why Rehashing Is Necessary

When the table size changes from $m$ to $m'$, the hash function changes from $h(k) = f(k) \bmod m$ to $h'(k) = f(k) \bmod m'$. For most keys, $h(k) \neq h'(k)$, so every element must be reinserted into the new table. Simply copying elements to the same slot indices would place them in incorrect positions.

??? example "Rehashing Changes Slot Assignments"

    Consider a table with $m = 4$ storing keys $\{3, 7, 11, 14\}$ using $h(k) = k \bmod m$:

    $$
    \begin{array}{rcl}
    h(3) = 3, \quad h(7) = 3, \quad h(11) = 3, \quad h(14) = 2
    \end{array}
    $$

    After doubling to $m' = 8$:

    $$
    \begin{array}{rcl}
    h'(3) = 3, \quad h'(7) = 7, \quad h'(11) = 3, \quad h'(14) = 6
    \end{array}
    $$

    Keys 7 and 14 move to different slots. Without rehashing, lookups for these keys would search the wrong slots and fail.

## The Resizing Algorithm

The resize operation consists of three steps:

1. **Allocate** a new table of size $m'$ with empty slots.
2. **Rehash** every element from the old table: compute $h'(k) = f(k) \bmod m'$ and insert $(k, v)$ into the new table.
3. **Deallocate** the old table.

The cost of resizing is $\Theta(n + m')$: $\Theta(n)$ for rehashing all $n$ elements, plus $\Theta(m')$ for initializing the new table. When $m' = \Theta(n)$, the total cost is $\Theta(n)$.

## Growth Policies

The growth policy determines the new table size $m'$ when the current table is too full. The most common policies are:

**Table doubling** ($m' = 2m$). This is the standard choice. After doubling, the load factor drops by half:

$$
\alpha' = \frac{n}{2m} = \frac{\alpha}{2}
$$

If the upper threshold is $\alpha_{\max} = 0.75$, then after doubling, $\alpha' \approx 0.375$, leaving room for many insertions before the next resize.

**Exact growth** ($m' = \lceil n / \alpha_{\text{target}} \rceil$). The table is sized to achieve a specific target load factor. This wastes less memory than doubling but may trigger resizes more frequently.

**Growth by factor $c$** ($m' = \lceil c \cdot m \rceil$, $c > 1$). A generalization of doubling where $c$ can be chosen to balance resize frequency against memory overhead. Smaller $c$ (e.g., $c = 1.5$) wastes less memory but resizes more often.

The choice of growth factor affects the amortized cost. Table doubling ($c = 2$) achieves $O(1)$ amortized cost per insertion. Any constant factor $c > 1$ also achieves $O(1)$ amortized cost, but with different constant factors.

**Amortized cost with factor $c$.** After $n$ insertions starting from size 1, the total cost of all resizes is:

$$
T_{\text{resize}} = \sum_{i=0}^{\lfloor \log_c n \rfloor} c^i = \frac{c^{\lfloor \log_c n \rfloor + 1} - 1}{c - 1} \leq \frac{cn}{c - 1}
$$

The amortized cost per insertion is therefore:

$$
\hat{c} = 1 + \frac{c}{c - 1}
$$

For $c = 2$: $\hat{c} = 1 + 2 = 3$. For $c = 1.5$: $\hat{c} = 1 + 3 = 4$. Smaller growth factors increase the constant in the amortized bound but reduce peak memory usage.

## Shrink Policies

When elements are deleted, the load factor decreases and the table wastes memory. Shrinking the table reclaims this memory:

**Table halving** ($m' = m/2$ when $\alpha < \alpha_{\min}$). The shrink threshold $\alpha_{\min}$ must be strictly less than $\alpha_{\max} / 2$ to prevent **thrashing** -- a pathological pattern where alternating insertions and deletions near the boundary repeatedly trigger resizes.

!!! warning "Thrashing"

    If the growth threshold is $\alpha_{\max} = 1$ and the shrink threshold is $\alpha_{\min} = 1/2$, then a table at capacity that alternates between one insertion (triggering a double) and one deletion (triggering a halve) pays $\Theta(n)$ per operation. Setting $\alpha_{\min} = 1/4$ prevents this because after a halve, many deletions are needed before the next halve, and after a double, many insertions are needed before the next double.

**Standard asymmetric thresholds.** The combination $\alpha_{\max} = 1$, $\alpha_{\min} = 1/4$ guarantees that the load factor stays within $[1/4, 1]$ and the amortized cost per operation is $O(1)$ for any intermixed sequence of insertions and deletions.

## Incremental Rehashing

In latency-sensitive applications, the $\Theta(n)$ cost of a single resize can cause unacceptable pauses. **Incremental rehashing** spreads the cost across multiple operations:

1. Allocate the new table but keep the old table active.
2. On each subsequent insert or lookup, rehash a fixed number of elements (e.g., 2 or 4) from the old table to the new table.
3. Lookups check both tables until migration is complete.
4. Once all elements are migrated, deallocate the old table.

This approach bounds the worst-case cost per operation at $O(1)$ (amortized cost unchanged) at the expense of temporarily using $O(n)$ extra memory during the transition.

Redis uses incremental rehashing in its hash table implementation to avoid blocking the event loop during large resizes.

## Implementation

```python
"""
Dynamic hash table with automatic resizing and rehashing.

Demonstrates table doubling on high load factor and table
halving on low load factor, with thrashing prevention via
asymmetric thresholds.
"""


# === Dynamic Hash Table ===

class DynamicHashTable:
    """Hash table that automatically resizes to maintain performance."""

    GROW_THRESHOLD = 0.75
    SHRINK_THRESHOLD = 0.25
    MIN_SIZE = 4

    def __init__(self, size: int = 4):
        self.size = size
        self.table: list[list[tuple]] = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key: int) -> int:
        return key % self.size

    def _resize(self, new_size: int) -> None:
        """Resize the table and rehash all elements."""
        old_table = self.table
        self.size = new_size
        self.table = [[] for _ in range(new_size)]
        self.count = 0
        for chain in old_table:
            for key, value in chain:
                self.put(key, value)

    def put(self, key: int, value) -> None:
        """Insert or update a key-value pair, resizing if needed."""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
        self.count += 1

        if self.count / self.size > self.GROW_THRESHOLD:
            self._resize(self.size * 2)

    def delete(self, key: int) -> bool:
        """Delete a key, shrinking the table if needed."""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                self.count -= 1
                if (self.size > self.MIN_SIZE
                        and self.count / self.size < self.SHRINK_THRESHOLD):
                    self._resize(max(self.size // 2, self.MIN_SIZE))
                return True
        return False

    def get(self, key: int):
        """Retrieve the value for a key, or None if not found."""
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def load_factor(self) -> float:
        return self.count / self.size


# === Demonstration ===

if __name__ == "__main__":
    ht = DynamicHashTable(size=4)

    print("=== Insertions with automatic growth ===")
    for i in range(1, 13):
        ht.put(i, i * 10)
        print(f"Insert {i:2d}: n={ht.count:2d}, m={ht.size:2d}, "
              f"alpha={ht.load_factor():.3f}")

    print("\n=== Deletions with automatic shrinking ===")
    for i in range(1, 10):
        ht.delete(i)
        print(f"Delete {i:2d}: n={ht.count:2d}, m={ht.size:2d}, "
              f"alpha={ht.load_factor():.3f}")
```

**Output:**
```
=== Insertions with automatic growth ===
Insert  1: n= 1, m= 4, alpha=0.250
Insert  2: n= 2, m= 4, alpha=0.500
Insert  3: n= 3, m= 4, alpha=0.750
Insert  4: n= 4, m= 8, alpha=0.500
Insert  5: n= 5, m= 8, alpha=0.625
Insert  6: n= 6, m= 8, alpha=0.750
Insert  7: n= 7, m=16, alpha=0.438
Insert  8: n= 8, m=16, alpha=0.500
Insert  9: n= 9, m=16, alpha=0.562
Insert 10: n=10, m=16, alpha=0.625
Insert 11: n=11, m=16, alpha=0.688
Insert 12: n=12, m=16, alpha=0.750
=== Deletions with automatic shrinking ===
Delete  1: n=11, m=16, alpha=0.688
Delete  2: n=10, m=16, alpha=0.625
Delete  3: n= 9, m=16, alpha=0.562
Delete  4: n= 8, m=16, alpha=0.500
Delete  5: n= 7, m=16, alpha=0.438
Delete  6: n= 6, m=16, alpha=0.375
Delete  7: n= 5, m=16, alpha=0.312
Delete  8: n= 4, m=16, alpha=0.250
Delete  9: n= 3, m= 8, alpha=0.375
```

## Summary

Dynamic resizing keeps the load factor within a bounded range, ensuring $O(1)$ expected-time operations throughout the lifetime of a hash table. Table doubling on growth and halving on shrink, with asymmetric thresholds to prevent thrashing, achieves $O(1)$ amortized cost per operation. For latency-sensitive applications, incremental rehashing spreads the $\Theta(n)$ cost of a resize across many operations, bounding the worst-case per-operation cost.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [Introduction to Algorithms (CLRS), Chapter 16 — Dynamic Tables](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
