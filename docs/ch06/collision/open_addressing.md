# Open Addressing

In chaining, each slot holds a pointer to an external data structure that stores colliding keys. Open addressing takes the opposite approach: all keys reside directly inside the table array. When a collision occurs at slot $h(k)$, the algorithm **probes** alternative slots within the same array following a deterministic sequence until an empty position is found. This eliminates pointer overhead and improves cache locality, at the cost of more complex deletion and degraded performance as the table fills.

## General Probe Sequence

An open-addressed hash table of size $m$ defines a **probe sequence** for each key $k$ as a permutation of $\{0, 1, \ldots, m-1\}$:

$$
\langle\, h(k, 0),\; h(k, 1),\; \ldots,\; h(k, m-1) \,\rangle
$$

The function $h : U \times \{0, \ldots, m-1\} \to \{0, \ldots, m-1\}$ takes both a key and a probe number, and for each fixed $k$ it must produce a permutation of all $m$ slot indices. This ensures that every slot is eventually examined.

## Uniform Hashing Assumption

The **uniform hashing assumption** states that each key's probe sequence is equally likely to be any of the $m!$ permutations of $\{0, 1, \ldots, m-1\}$. This is a theoretical ideal --- no practical hashing scheme achieves it exactly --- but it provides tight bounds on expected performance.

Under uniform hashing with load factor $\alpha = n/m < 1$:

**Unsuccessful search (or insertion):**

$$
E[\text{probes}] \le \frac{1}{1 - \alpha}
$$

**Successful search:**

$$
E[\text{probes}] \le \frac{1}{\alpha} \ln \frac{1}{1 - \alpha}
$$

??? note "Derivation sketch for unsuccessful search"
    The probability that the first probe finds an occupied slot is $\alpha = n/m$. Given that the first slot is occupied, the conditional probability that the second slot is also occupied is at most $(n-1)/(m-1) \le \alpha$. Continuing this argument, the expected number of probes is bounded by

    $$
    \sum_{i=0}^{\infty} \alpha^i = \frac{1}{1 - \alpha}
    $$

## Probing Strategies

Different choices of $h(k, i)$ yield different tradeoffs between simplicity, clustering behavior, and approximation to uniform hashing.

| Strategy | Probe formula | Distinct sequences | Clustering |
|---|---|---|---|
| Linear | $(h'(k) + i) \bmod m$ | $m$ | Primary |
| Quadratic | $(h'(k) + c_1 i + c_2 i^2) \bmod m$ | $m$ | Secondary |
| Double | $(h_1(k) + i \cdot h_2(k)) \bmod m$ | $m^2$ | None |

Linear probing produces only $m$ distinct probe sequences (one per initial hash value), quadratic probing also produces $m$ sequences, and double hashing produces $\Theta(m^2)$ sequences --- the closest practical approximation to the $m!$ ideal.

## Insertion and Search

**Insertion** of key $k$:

1. Compute $h(k, 0)$. If the slot is empty (or marked as deleted), place $k$ there.
2. Otherwise compute $h(k, 1)$, then $h(k, 2)$, and so on until an empty or deleted slot is found.
3. If all $m$ slots have been examined without finding an empty one, the table is full.

**Search** for key $k$:

1. Compute $h(k, 0)$. If the slot contains $k$, return it.
2. If the slot is empty, $k$ is not in the table.
3. If the slot contains a different key (or a tombstone), continue to $h(k, 1)$, etc.

## The Deletion Problem

Naive deletion --- setting a slot to empty --- breaks the probe chain for any key whose insertion probed past the now-empty slot. Two solutions exist:

**Tombstones**: mark deleted slots with a special sentinel value. Search treats tombstones as occupied (continues probing), while insertion treats them as empty (reusable). The downside is that tombstones never disappear without rehashing, causing the effective load factor to increase over time.

**Rehashing on delete**: after deleting a key, reinsert all keys in the same cluster. This maintains a tombstone-free table but costs $O(n)$ in the worst case per deletion.

## Load Factor Constraints

Because open addressing stores all entries within the table itself, the load factor is bounded by

$$
0 \le \alpha = \frac{n}{m} \le 1
$$

In practice, performance degrades rapidly above $\alpha \approx 0.7$. Most implementations trigger a resize (typically doubling $m$) when $\alpha$ exceeds a threshold, keeping the amortized cost of all operations at $O(1)$.

## Open Addressing vs Chaining

| Aspect | Open addressing | Chaining |
|---|---|---|
| Memory layout | All entries in array | Array + linked lists |
| Cache behavior | Excellent (sequential) | Poor (pointer chasing) |
| Max load factor | $\alpha < 1$ (practically ${\le}0.7$) | Unbounded |
| Deletion | Tombstones or rehash | Simple pointer removal |
| Worst-case search | $O(n)$ | $O(n)$ |
| Extra memory | None | One pointer per entry |

Open addressing is preferred when memory is tight and deletions are infrequent. Chaining is preferred when the load factor is unpredictable or deletions are frequent.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
