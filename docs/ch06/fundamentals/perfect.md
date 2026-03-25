# Perfect Hashing

Standard hash tables provide $O(1)$ expected-time operations, but the worst case remains $O(n)$ when all keys collide into a single slot. For applications where worst-case guarantees matter -- such as routing tables in network hardware, keyword lookup in compilers, or real-time systems -- this $O(n)$ worst case is unacceptable. **Perfect hashing** eliminates collisions entirely for a static set of keys, guaranteeing $O(1)$ worst-case lookup with $O(n)$ total space.

## Definition

A hash function $h : S \to \{0, 1, \ldots, m-1\}$ is **perfect** for a set $S$ of $n$ keys if it is injective on $S$:

$$
h(k_1) \neq h(k_2) \quad \text{for all } k_1, k_2 \in S \text{ with } k_1 \neq k_2
$$

A perfect hash function maps every key in $S$ to a distinct slot, so no collisions occur and every lookup requires exactly one table probe. The function is defined for a **static** key set -- one that is known in advance and does not change during operation.

A perfect hash function is called **minimal** if $m = n$, meaning it uses exactly as many slots as there are keys, with no wasted space.

## The Two-Level FKS Scheme

The Fredman, Komlós, and Szemerédi (FKS) scheme constructs a perfect hash function using two levels of universal hashing. The construction works as follows:

### First Level

Choose a universal hash function $h : U \to \{0, 1, \ldots, m-1\}$ with $m = n$ slots. This maps the $n$ keys into $n$ buckets, but collisions are expected. Let $n_j$ denote the number of keys that hash to slot $j$:

$$
n_j = |\{k \in S : h(k) = j\}|
$$

### Second Level

For each slot $j$ that receives $n_j > 0$ keys, create a secondary hash table of size $m_j = n_j^2$. Choose a second universal hash function $h_j : U \to \{0, 1, \ldots, m_j - 1\}$ for each slot. Because the secondary table has $n_j^2$ slots for only $n_j$ keys, the birthday paradox argument guarantees that a collision-free $h_j$ exists and can be found after $O(1)$ expected trials.

**Lookup procedure.** To find key $k$:

1. Compute $j = h(k)$ to identify the first-level slot.
2. Compute $h_j(k)$ to index into the secondary table at slot $j$.
3. Check whether the stored key matches $k$.

Each step takes $O(1)$ time, so the total lookup is $O(1)$ in the worst case.

## Space Analysis

The critical question is whether the two-level scheme uses $O(n)$ total space. The total space is:

$$
\text{Total space} = m + \sum_{j=0}^{m-1} m_j = n + \sum_{j=0}^{n-1} n_j^2
$$

We need to bound $\sum_{j} n_j^2$. The number of collisions at the first level is:

$$
C = \sum_{j=0}^{n-1} \binom{n_j}{2} = \frac{1}{2}\left(\sum_{j} n_j^2 - n\right)
$$

For a universal hash function, the expected number of collisions is at most:

$$
\mathbb{E}[C] \leq \binom{n}{2} \cdot \frac{1}{m} = \frac{n(n-1)}{2n} = \frac{n-1}{2}
$$

Therefore:

$$
\mathbb{E}\left[\sum_{j} n_j^2\right] = 2\mathbb{E}[C] + n \leq (n-1) + n = 2n - 1
$$

By choosing a first-level hash function where $\sum_j n_j^2 < 4n$ (which happens with probability at least $1/2$ by Markov's inequality), the total space is $O(n)$.

??? example "Two-Level Perfect Hashing Construction"

    Consider the static key set $S = \{10, 22, 37, 40, 52, 60\}$ with $n = 6$ keys.

    **First level:** Choose $m = 6$ and suppose $h(k) = k \bmod 6$:

    $$
    \begin{array}{rcl}
    h(10) = 4, \quad h(22) = 4, \quad h(37) = 1 \\
    h(40) = 4, \quad h(52) = 4, \quad h(60) = 0
    \end{array}
    $$

    Slot 4 receives $n_4 = 4$ keys (a bad first-level hash). The secondary table at slot 4 has $m_4 = 4^2 = 16$ slots. A universal hash function $h_4$ is chosen to map the 4 keys to 16 slots without collision.

    **Space:** $\sum_j n_j^2 = 1^2 + 1^2 + 4^2 + 0 + 0 + 0 = 18$, which is $3n = 18$. A better first-level hash function would reduce this sum.

## Guarantees

The FKS scheme provides three guarantees:

1. **Worst-case $O(1)$ lookup.** Every key is found in exactly two hash computations and two table probes.
2. **$O(n)$ space.** The total memory used across both levels is linear in $n$.
3. **$O(n)$ expected construction time.** Building the two-level structure requires $O(n)$ time in expectation, since each level involves selecting a universal hash function (which succeeds after $O(1)$ expected trials).

## Static vs Dynamic Perfect Hashing

The FKS scheme assumes the key set $S$ is static. Dynamic perfect hashing extends the idea to support insertions and deletions while maintaining worst-case $O(1)$ lookup:

**Cuckoo hashing** uses two hash functions and two tables. Each key is stored in one of two possible locations. Insertions may trigger a chain of relocations, but lookups always check exactly two positions, giving $O(1)$ worst-case lookup.

**Dynamic FKS** rebuilds secondary tables when they become too full after insertions. The amortized cost of rebuilding is $O(1)$ per operation when the load factor is managed with table doubling.

## Practical Considerations

!!! tip "When to Use Perfect Hashing"

    Perfect hashing is most valuable when:

    - The key set is known at compile time or during initialization (e.g., reserved keywords in a programming language, routing table entries).
    - The application requires guaranteed $O(1)$ worst-case lookup (e.g., real-time systems, hardware lookup tables).
    - The key set changes infrequently, so the one-time construction cost is amortized over many lookups.

    For dynamic key sets with frequent insertions and deletions, standard hash tables with good average-case performance are usually more practical.

## Summary

Perfect hashing achieves the ideal of $O(1)$ worst-case lookup for static key sets by using a two-level construction based on universal hash families. The FKS scheme guarantees $O(n)$ space by sizing secondary tables quadratically in the number of keys per bucket, exploiting the birthday paradox to ensure collision-free secondary hash functions exist. While limited to static key sets in its basic form, extensions like cuckoo hashing bring similar worst-case guarantees to the dynamic setting.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
