# Division Method

The division method is the simplest and most widely used technique for constructing hash functions. Given an integer key $k$ and a table of size $m$, the hash value is the remainder of $k$ divided by $m$. This approach is easy to implement and efficient to compute, but its effectiveness depends critically on the choice of $m$.

## Definition

The **division method** computes the hash of a key $k$ as:

$$
h(k) = k \bmod m
$$

where $m$ is the number of slots in the hash table and $\bmod$ denotes the modulo operation (the remainder after integer division). The result $h(k)$ always lies in the range $\{0, 1, \ldots, m-1\}$, which directly indexes a slot in the table.

The operation requires a single integer division, making it $O(1)$ in time and requiring no additional memory beyond the key and the table size.

## Choosing the Table Size

The choice of $m$ determines how well the division method distributes keys. Poor choices lead to systematic clustering, while good choices achieve near-uniform distribution.

### Sizes to Avoid

**Powers of two.** If $m = 2^p$, then $k \bmod 2^p$ extracts only the $p$ least significant bits of $k$:

$$
k \bmod 2^p = k\ \&\ (2^p - 1)
$$

where $\&$ denotes bitwise AND. This means the hash ignores all higher-order bits of $k$. When keys share structure in their lower bits (e.g., even numbers, multiples of 4, memory addresses aligned to word boundaries), the distribution degrades severely.

**Powers of ten.** Similarly, $m = 10^p$ extracts only the last $p$ decimal digits. If keys are student IDs, phone numbers, or zip codes, the last few digits often carry less entropy than the full key.

**Numbers with small factors.** If $m$ and the key distribution share a common factor $d > 1$, then only $m/d$ of the $m$ slots receive keys. For example, if all keys are even and $m$ is even, only even-indexed slots are used.

### Recommended Sizes

A good choice for $m$ is a **prime number not close to a power of two**. The reason is twofold:

1. A prime $m$ ensures that $k \bmod m$ depends on all bits of $k$, not just the low-order ones.
2. Avoiding proximity to powers of two prevents the hash from being dominated by a simple bit pattern.

??? example "Impact of Table Size Choice"

    Consider the keys $\{0, 8, 16, 24, 32, 40, 48, 56\}$ (all multiples of 8).

    With $m = 8$ (power of two):

    $$
    h(k) = k \bmod 8 = 0 \quad \text{for all } k
    $$

    Every key maps to slot 0, producing maximum clustering.

    With $m = 7$ (prime):

    $$
    \begin{array}{rcl}
    h(0) = 0,\quad h(8) = 1,\quad h(16) = 2,\quad h(24) = 3 \\
    h(32) = 4,\quad h(40) = 5,\quad h(48) = 6,\quad h(56) = 0
    \end{array}
    $$

    The keys spread across all 7 slots with only one collision, demonstrating the superiority of a prime table size for structured inputs.

## Analysis Under Simple Uniform Hashing

Under the simple uniform hashing assumption (SUHA), each key is equally likely to hash to any slot, and the expected number of keys per slot is the load factor $\alpha = n/m$. In this setting, the division method achieves:

- **Expected chain length:** $\alpha$
- **Expected time for unsuccessful search:** $\Theta(1 + \alpha)$
- **Expected time for successful search:** $\Theta(1 + \alpha/2)$

When $m$ is proportional to $n$ (i.e., $\alpha = O(1)$), all operations run in $O(1)$ expected time.

## Practical Guidelines

The following rules of thumb help select $m$ for the division method:

1. **Pick a prime** $m$ such that $m$ is not close to $2^p$ for any integer $p$.
2. A common heuristic: choose $m$ as a prime near $n / \alpha_{\text{target}}$, where $\alpha_{\text{target}} \approx 0.75$ is the desired load factor.
3. For a table expected to hold about 1000 elements, good choices include $m = 1009$, $m = 1013$, or $m = 1021$ (all primes, none near a power of two).

## Implementation

The following Python implementation demonstrates the division method applied to integer keys with separate chaining for collision resolution.

```python
"""
Division method hash table with separate chaining.

Demonstrates h(k) = k mod m where m is chosen as a prime
for better distribution of hash values.
"""


# === Hash Table with Division Method ===

class DivisionHashTable:
    """Hash table using the division method h(k) = k mod m."""

    def __init__(self, size: int = 11):
        """Initialize with a prime table size (default 11)."""
        self.size = size
        self.table: list[list[tuple]] = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key: int) -> int:
        """Compute hash using the division method."""
        return key % self.size

    def put(self, key: int, value) -> None:
        """Insert or update a key-value pair."""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
        self.count += 1

    def get(self, key: int):
        """Retrieve the value associated with key, or None."""
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def load_factor(self) -> float:
        """Return the current load factor n/m."""
        return self.count / self.size


# === Demonstration ===

if __name__ == "__main__":
    ht = DivisionHashTable(size=7)

    keys = [10, 20, 30, 40, 50]
    for k in keys:
        ht.put(k, k * 10)

    for k in keys:
        print(f"h({k}) = {k} mod 7 = {k % 7}, value = {ht.get(k)}")

    print(f"Load factor: {ht.load_factor():.2f}")
```

**Output:**
```
h(10) = 10 mod 7 = 3, value = 100
h(20) = 20 mod 7 = 6, value = 200
h(30) = 30 mod 7 = 2, value = 300
h(40) = 40 mod 7 = 5, value = 400
h(50) = 50 mod 7 = 1, value = 500
Load factor: 0.71
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
