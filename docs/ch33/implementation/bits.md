# Bit Manipulation

Bitwise operations provide constant-time set operations, compact state representations, and efficient counting -- capabilities that appear throughout competitive programming in bitmask DP, graph coloring, subset enumeration, and data structures like Fenwick trees. Understanding how integers encode sets of bits unlocks an entire class of elegant solutions.

## Fundamental Bitwise Operations

Every integer is a sequence of bits. For a non-negative integer $x$, bit $i$ (counting from 0) has value 1 if and only if the $i$-th power of 2 appears in the binary representation.

$$
x = \sum_{i=0}^{k} b_i \cdot 2^i \quad \text{where } b_i \in \{0, 1\}
$$

The six fundamental bitwise operations are:

| Operation | Symbol (C++/Python) | Meaning |
|---|---|---|
| AND | `a & b` | 1 only where both bits are 1 |
| OR | `a \| b` | 1 where at least one bit is 1 |
| XOR | `a ^ b` | 1 where bits differ |
| NOT | `~a` | Flip all bits |
| Left shift | `a << k` | Multiply by $2^k$ |
| Right shift | `a >> k` | Integer divide by $2^k$ |

## Common Bit Tricks

### Check if Bit $i$ Is Set

```
(x >> i) & 1
```

Shifts bit $i$ to position 0 and masks all other bits.

### Set Bit $i$

```
x | (1 << i)
```

### Clear Bit $i$

```
x & ~(1 << i)
```

### Toggle Bit $i$

```
x ^ (1 << i)
```

### Lowest Set Bit

The expression `x & (-x)` isolates the lowest set bit of $x$. This works because $-x$ in two's complement is $\sim x + 1$, which flips all bits above the lowest set bit.

$$
\text{lowbit}(x) = x \;\&\; (-x)
$$

This operation is the foundation of the Fenwick tree (Binary Indexed Tree).

### Clear the Lowest Set Bit

```
x & (x - 1)
```

This turns off the lowest set bit. Repeatedly applying this counts the number of set bits.

### Check if Power of Two

$x$ is a power of two if and only if $x > 0$ and $x \;\&\; (x - 1) = 0$.

## Counting Set Bits (Popcount)

The number of 1-bits in $x$ is called the **population count** or **Hamming weight**.

```python
"""
Bit manipulation utilities for competitive programming.

Provides popcount, subset enumeration, and Fenwick tree
operations that rely on the lowest-set-bit trick.
"""

# ===================================================================
# Population Count
# ===================================================================

def popcount(x):
    """Count the number of set bits in x."""
    count = 0
    while x:
        x &= x - 1  # Clear lowest set bit
        count += 1
    return count

# ===================================================================
# Subset Enumeration
# ===================================================================

def enumerate_subsets(mask):
    """Enumerate all non-empty subsets of the given bitmask."""
    subsets = []
    sub = mask
    while sub > 0:
        subsets.append(sub)
        sub = (sub - 1) & mask
    return subsets

# ===================================================================
# Fenwick Tree (Binary Indexed Tree)
# ===================================================================

class FenwickTree:
    """Fenwick tree for prefix sum queries and point updates.

    Uses the lowest-set-bit trick: i & (-i) gives the
    lowest set bit, which determines the range each node covers.
    """

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):
        """Add delta to position i (1-indexed)."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):
        """Return prefix sum from 1 to i."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_query(self, l, r):
        """Return sum from index l to r (inclusive, 1-indexed)."""
        return self.query(r) - self.query(l - 1)

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    # Popcount examples
    print(f"popcount(7) = {popcount(7)}")     # 3 (binary: 111)
    print(f"popcount(10) = {popcount(10)}")   # 2 (binary: 1010)

    # Subset enumeration
    mask = 0b1011  # bits {0, 1, 3}
    subs = enumerate_subsets(mask)
    print(f"Subsets of {bin(mask)}: {[bin(s) for s in subs]}")

    # Fenwick tree
    ft = FenwickTree(5)
    for i, v in enumerate([1, 3, 5, 7, 9], 1):
        ft.update(i, v)
    print(f"Sum [1,3]: {ft.range_query(1, 3)}")
    print(f"Sum [2,5]: {ft.range_query(2, 5)}")
```

**Output:**
```
popcount(7) = 3
popcount(10) = 2
Subsets of 0b1011: ['0b1011', '0b1010', '0b1001', '0b1000', '0b11', '0b10', '0b1']
Sum [1,3]: 9
Sum [2,5]: 24
```

## Bitmask as Set Representation

A bitmask of $n$ bits represents a subset of $\{0, 1, \ldots, n-1\}$. This encoding is compact (a single integer) and supports set operations in $O(1)$:

| Set operation | Bitmask operation |
|---|---|
| Union $A \cup B$ | `a \| b` |
| Intersection $A \cap B$ | `a & b` |
| Difference $A \setminus B$ | `a & ~b` |
| Symmetric difference $A \triangle B$ | `a ^ b` |
| Complement $\bar{A}$ | `~a & ((1 << n) - 1)` |
| Membership $i \in A$ | `(a >> i) & 1` |
| Cardinality $|A|$ | `popcount(a)` |

## Subset Enumeration

### All Subsets of $\{0, \ldots, n-1\}$

Iterate from $0$ to $2^n - 1$. Each integer is a subset.

### All Subsets of a Given Mask

To enumerate all subsets of a bitmask $m$ (including the empty set):

```
sub = m
while sub > 0:
    process(sub)
    sub = (sub - 1) & m
process(0)  # empty subset
```

The total number of subsets of $m$ is $2^{\text{popcount}(m)}$. Summed over all masks, the total work for enumerating all subsets of all masks of $n$ bits is:

$$
\sum_{m=0}^{2^n - 1} 2^{\text{popcount}(m)} = 3^n
$$

This identity follows from the binomial theorem: each bit independently contributes a factor of $(1 + 2) = 3$ (it can be absent from $m$, present in $m$ but absent from the subset, or present in both).

## Applications in Competitive Programming

### Bitmask DP

When the state space involves subsets of a small set ($n \le 20$), represent the subset as a bitmask and use it as a DP index. The Travelling Salesman Problem with $n$ cities uses:

$$
dp[\text{mask}][i] = \text{minimum cost to visit the cities in mask, ending at city } i
$$

The total state space is $O(2^n \cdot n)$, and each transition examines $O(n)$ neighbors, giving $O(2^n \cdot n^2)$ total time.

### XOR Properties

XOR has useful algebraic properties for competitive programming:

- **Self-inverse**: $a \oplus a = 0$
- **Identity**: $a \oplus 0 = a$
- **Associative and commutative**: enables prefix XOR for range XOR queries

A classic application: in an array where every element appears twice except one, XOR of all elements yields the unique element.

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
