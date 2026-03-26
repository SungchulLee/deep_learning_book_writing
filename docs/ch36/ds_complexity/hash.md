# Hash Table

Hash tables provide the fastest average-case lookup of any data structure by mapping
keys to array indices through a hash function. The trade-off is that worst-case
performance degrades to $O(n)$ when many keys collide, and hash tables do not support
ordered operations (min, max, predecessor) efficiently.

## Hash Function Basics

A hash function maps a key $k$ from a universe $U$ to an index in a table of size $m$.
The most common method is the division method:

$$
h(k) = k \bmod m
$$

Other methods include the multiplication method, $h(k) = \lfloor m(kA \bmod 1) \rfloor$
for a constant $A$, and universal hashing, which selects $h$ randomly from a family to
guarantee expected $O(1)$ performance.

## Operation Complexities by Resolution Strategy

### Chaining (Separate Chaining)

Each table slot holds a linked list of colliding elements.

| Operation | Average | Worst | Notes |
|---|---|---|---|
| Search | $O(1 + \alpha)$ | $O(n)$ | $\alpha = n/m$ is the load factor |
| Insert | $O(1)$ | $O(1)$ | Insert at head of chain |
| Delete | $O(1 + \alpha)$ | $O(n)$ | Search + delete from list |
| Space | $O(n + m)$ | -- | $n$ elements + $m$ slots |

Under simple uniform hashing, the expected chain length is $\alpha$. Keeping
$\alpha \le 1$ (by resizing) ensures $O(1)$ expected time for all operations.

### Open Addressing

All elements are stored directly in the table array. Collisions are resolved by
probing for the next available slot.

| Probing Method | Search (avg, successful) | Search (avg, unsuccessful) | Clustering |
|---|---|---|---|
| Linear probing | $\frac{1}{2}\!\left(1 + \frac{1}{1-\alpha}\right)$ | $\frac{1}{2}\!\left(1 + \frac{1}{(1-\alpha)^2}\right)$ | Primary clustering |
| Quadratic probing | $\approx 1 - \ln(1-\alpha) - \alpha/2$ | $\frac{1}{1-\alpha} - \alpha - \ln(1-\alpha)$ | Secondary clustering |
| Double hashing | $\frac{1}{\alpha}\ln\!\frac{1}{1-\alpha}$ | $\frac{1}{1-\alpha}$ | No clustering |

!!! warning "Load Factor for Open Addressing"
    Open addressing requires $\alpha < 1$ (the table must have empty slots). As
    $\alpha \to 1$, probe counts grow without bound. Practical implementations
    keep $\alpha \le 0.7$ and resize when exceeded.

## Resizing Complexity

When the load factor exceeds a threshold, the table doubles in size and all elements
are rehashed.

| Operation | Cost | Frequency | Amortized |
|---|---|---|---|
| Single insert (no resize) | $O(1)$ | Most inserts | -- |
| Resize + rehash | $O(n)$ | After $n$ inserts | $O(1)$ |
| Sequence of $n$ inserts | -- | -- | $O(n)$ total |

The amortized $O(1)$ per insert follows from the same doubling argument used for
dynamic arrays: each element is copied $O(\log n)$ times across all resizes, and the
total cost across $n$ inserts is $O(n)$.

## Specialized Hash Structures

| Structure | Space | False Positive | Use Case |
|---|---|---|---|
| Bloom filter | $O(m)$ bits | $\left(1 - e^{-kn/m}\right)^k$ | Membership test (no false negatives) |
| Count-Min sketch | $O(w \times d)$ | Overestimates by $\epsilon n$ | Frequency estimation |
| Cuckoo hash table | $O(n)$ | $O(1)$ worst-case lookup | Guaranteed $O(1)$ search |
| Robin Hood hashing | $O(n)$ | -- | Reduces variance in probe length |

Here $k$ is the number of hash functions, $w$ is the width, and $d$ is the depth of
the sketch.

## Complexity Summary

| Operation | Chaining (avg) | Open Addressing (avg) | Cuckoo (worst) |
|---|---|---|---|
| Search | $O(1)$ | $O(1)$ | $O(1)$ |
| Insert | $O(1)$ | $O(1)$ | $O(1)$ amort. |
| Delete | $O(1)$ | $O(1)$ with tombstones | $O(1)$ |
| Space | $O(n + m)$ | $O(m)$ | $O(n)$ |

!!! tip "Hash Tables vs Balanced BSTs"
    Hash tables provide $O(1)$ average operations but $O(n)$ worst case and no
    ordering. Balanced BSTs provide $O(\log n)$ guaranteed operations with ordered
    iteration. Use hash tables when you need fast lookup without ordering; use BSTs
    when you need min/max, range queries, or sorted traversal.

## Language Implementations

| Language | Hash Map | Hash Set | Ordered Map |
|---|---|---|---|
| Python | `dict` | `set` | -- (use `sorted()`) |
| C++ | `unordered_map` | `unordered_set` | `map` (Red-Black tree) |
| Java | `HashMap` | `HashSet` | `TreeMap` (Red-Black tree) |
| Go | `map` | -- (use `map[K]bool`) | -- |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Knuth, D. *The Art of Computer Programming, Vol. 3: Sorting and Searching*. 2nd ed. Addison-Wesley, 1998.
