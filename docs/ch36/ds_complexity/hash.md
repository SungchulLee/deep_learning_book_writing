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

## Exercises

**Exercise 1.**
Compare chaining and open addressing for hash table collision resolution. What are the tradeoffs in time, space, and cache performance?

??? success "Solution to Exercise 1"
    **Chaining**: each bucket stores a linked list of colliding elements. Insertion: $O(1)$ (prepend to list). Search: $O(1 + \alpha)$ average where $\alpha = n/m$ is the load factor. Space: $n$ elements + $n$ pointers + $m$ bucket pointers. Cache performance: poor (linked list nodes are scattered in memory). **Open addressing**: all elements stored in the table itself. On collision, probe the next slot (linear, quadratic, or double hashing). Insertion/search: $O(1/(1-\alpha))$ average. Must keep $\alpha < 0.7$ for good performance. Space: no pointer overhead, but table must be larger to maintain low load factor. Cache performance: better (elements are contiguous; linear probing has excellent locality). Open addressing wins on cache performance and memory (no pointers). Chaining is simpler and degrades more gracefully at high load factors. $\square$

---

**Exercise 2.**
A hash table with $n$ elements and $m$ buckets has load factor $\alpha = n/m$. Derive the expected number of probes for a successful and unsuccessful search with chaining.

??? success "Solution to Exercise 2"
    **Unsuccessful search** (element not in table): the search must traverse the entire chain at a random bucket. Expected chain length: $\alpha = n/m$. Expected probes: $1 + \alpha$ (one hash computation + traversal). **Successful search**: the expected number of elements examined when searching for a random element. An element inserted at time $i$ (when the table had $i-1$ elements) was placed in a chain of expected length $1 + (i-1)/m$. Averaging over all $n$ elements: $\frac{1}{n} \sum_{i=1}^{n} (1 + (i-1)/m) = 1 + (n-1)/(2m) \approx 1 + \alpha/2$. For $\alpha = 1$: successful search takes $\approx 1.5$ probes, unsuccessful takes $\approx 2$ probes. $\square$

---

**Exercise 3.**
Explain why hash tables do not support efficient range queries or ordered iteration. What data structure should be used instead?

??? success "Solution to Exercise 3"
    Hash functions map keys to pseudorandom bucket positions, destroying any ordering. Two keys that are numerically adjacent (e.g., 100 and 101) map to unrelated buckets. A range query "find all keys in $[a, b]$" must check every bucket ($O(m + n)$), equivalent to scanning the entire table. Ordered iteration requires sorting all elements ($O(n \log n)$). For range queries and ordered access, use a balanced BST ($O(\log n)$ search + $O(k)$ traversal for $k$ results) or a B-tree. A skip list is another option with $O(\log n)$ search and sequential access along the bottom level. If both hash-speed point lookups and range queries are needed, use a combined approach: a hash table for point queries and a sorted index for range queries. $\square$

---

**Exercise 4.**
A hash table experiences a "resize storm" when the load factor threshold is crossed frequently. Describe the problem and propose a solution.

??? success "Solution to Exercise 4"
    If the load factor threshold is $\alpha = 0.75$ and the table grows by doubling, a sequence of alternating insertions and deletions near the threshold causes repeated resize-up and resize-down operations, each costing $O(n)$. Example: $n = 750$ in a 1000-slot table ($\alpha = 0.75$). Insert triggers resize to 2000 slots. Delete brings $n = 749$. If the shrink threshold is $\alpha = 0.25$ ($n < 500$), no shrink yet. But with a shrink threshold of $\alpha = 0.375$ ($n < 750$), the delete triggers shrink back to 1000, and the next insert triggers growth again. Solution: use asymmetric thresholds -- grow at $\alpha = 0.75$, shrink at $\alpha = 0.25$. This ensures that between a grow and a shrink, at least $n/2$ operations occur, amortizing both resizes to $O(1)$ per operation. $\square$

---

**Exercise 5.**
Python dictionaries use open addressing with a probing sequence. Explain why Python dicts maintain insertion order since Python 3.7 and what data structure enables this.

??? success "Solution to Exercise 5"
    Since CPython 3.6 (official guarantee in 3.7), dictionaries maintain insertion order using a **compact dict** design with two arrays: (1) a hash table of indices (sparse array, 1-2 bytes per slot) mapping hash positions to positions in the dense array. (2) A dense array of (hash, key, value) tuples stored in insertion order. Insertion appends to the dense array and records the index in the hash table. Iteration traverses the dense array sequentially (preserving insertion order). Deletion marks entries as deleted in the dense array (tombstone) and the hash table. This design improves memory efficiency (the hash table stores small indices instead of full key-value pairs) and provides insertion-order iteration for free. The trade-off: slightly more complex probing (indirection through the index table), but the overall performance is better due to the dense array's cache-friendly layout. $\square$
