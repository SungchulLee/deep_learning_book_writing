# Load Factor

The load factor is the single most important parameter governing hash table performance. It quantifies how full the table is and directly determines the expected cost of every operation. Understanding the load factor and choosing appropriate thresholds is essential for building hash tables that deliver on the $O(1)$ expected-time promise.

## Definition

The **load factor** of a hash table with $n$ stored elements and $m$ slots is:

$$
\alpha = \frac{n}{m}
$$

For a hash table using **chaining**, $\alpha$ can be any non-negative real number: $\alpha < 1$ means most slots are empty, $\alpha = 1$ means the number of elements equals the number of slots, and $\alpha > 1$ means some chains necessarily contain more than one element.

For a hash table using **open addressing**, each slot holds at most one element, so $\alpha$ must satisfy $0 \leq \alpha \leq 1$.

The load factor has a direct physical interpretation: under the simple uniform hashing assumption (SUHA), $\alpha$ is the expected number of elements stored in each slot.

## Load Factor and Expected Performance

The expected cost of hash table operations is a function of $\alpha$. Under SUHA with chaining:

**Unsuccessful search:** The search must traverse the entire chain at the hashed slot. The expected chain length is $\alpha$, so:

$$
\mathbb{E}[\text{cost of unsuccessful search}] = \Theta(1 + \alpha)
$$

**Successful search:** On average, the search examines about half the chain before finding the target:

$$
\mathbb{E}[\text{cost of successful search}] = \Theta\!\left(1 + \frac{\alpha}{2}\right)
$$

**Insertion:** Inserting at the head of a chain takes $\Theta(1)$ time (assuming no duplicate check). With a duplicate check, the cost matches an unsuccessful search: $\Theta(1 + \alpha)$.

These expressions reveal the key insight: **when $\alpha = O(1)$, all operations run in $O(1)$ expected time.**

## Load Factor in Open Addressing

For open addressing, the expected probe counts under the uniform hashing assumption are:

$$
\mathbb{E}[\text{unsuccessful search}] \leq \frac{1}{1 - \alpha}
$$

$$
\mathbb{E}[\text{successful search}] \leq \frac{1}{\alpha} \ln \frac{1}{1 - \alpha}
$$

These bounds diverge as $\alpha \to 1$. The following table illustrates the rapid degradation:

| $\alpha$ | Unsuccessful probes | Successful probes |
|---|---|---|
| 0.50 | 2.0 | 1.4 |
| 0.75 | 4.0 | 1.8 |
| 0.90 | 10.0 | 2.6 |
| 0.95 | 20.0 | 3.2 |
| 0.99 | 100.0 | 4.7 |

At $\alpha = 0.90$, an unsuccessful search examines 10 slots on average -- far from the $O(1)$ ideal. This sensitivity explains why open addressing implementations enforce strict load factor limits.

## Space-Time Tradeoff

The load factor encodes a fundamental tradeoff between space and time:

- **Low $\alpha$** (e.g., $\alpha = 0.25$): Fast operations (short chains, few probes) but wastes 75% of allocated memory.
- **High $\alpha$** (e.g., $\alpha = 0.90$): Memory-efficient but slow operations, especially for open addressing.
- **Balanced $\alpha$** (e.g., $\alpha \in [0.5, 0.75]$): Practical compromise between space and time.

The total memory used by a hash table is proportional to $m = n / \alpha$. Doubling $\alpha$ halves the memory requirement but increases expected search time. The optimal $\alpha$ depends on whether the application is memory-constrained or latency-constrained.

## Practical Load Factor Thresholds

Real-world hash table implementations use empirically validated load factor thresholds:

| Implementation | Strategy | Max $\alpha$ | Resize trigger |
|---|---|---|---|
| Java `HashMap` | Chaining | 0.75 | Double on exceeding |
| Python `dict` | Open addressing | 0.67 | Resize to $4n/3$ |
| C++ `unordered_map` | Chaining | 1.0 | Double on exceeding |
| Go `map` | Chaining | 6.5 (per bucket) | Double on exceeding |
| Rust `HashMap` | Robin Hood | 0.875 | Double on exceeding |

These thresholds balance expected-case performance against memory overhead for the typical workloads each language targets.

??? example "Load Factor During Table Growth"

    Start with $m = 4$ and insert elements one at a time, doubling when $\alpha > 0.75$:

    | Elements $n$ | Table size $m$ | $\alpha = n/m$ | Action |
    |---|---|---|---|
    | 1 | 4 | 0.25 | Insert |
    | 2 | 4 | 0.50 | Insert |
    | 3 | 4 | 0.75 | Insert |
    | 4 | 4 | 1.00 | Exceeds threshold; resize to $m = 8$ |
    | 4 | 8 | 0.50 | After resize |
    | 5 | 8 | 0.625 | Insert |
    | 6 | 8 | 0.75 | Insert |
    | 7 | 8 | 0.875 | Exceeds threshold; resize to $m = 16$ |
    | 7 | 16 | 0.4375 | After resize |

    After each resize, the load factor drops by half, keeping $\alpha$ within the range $[0.375, 0.75]$.

## Load Factor and Hash Function Quality

The expected-time formulas assume SUHA, which requires the hash function to distribute keys uniformly. When the hash function is imperfect, the actual chain lengths deviate from $\alpha$. The **variance** of chain length measures this deviation:

$$
\text{Var}[L_j] = \alpha \left(1 - \frac{1}{m}\right) \approx \alpha \quad \text{for large } m
$$

where $L_j$ is the length of chain $j$ under SUHA. A hash function that produces higher variance than this theoretical prediction is distributing keys non-uniformly, effectively increasing the "effective load factor" beyond $\alpha$.

## Summary

The load factor $\alpha = n/m$ governs hash table performance: all operations run in $O(1)$ expected time when $\alpha$ is bounded by a constant. Chaining degrades linearly with $\alpha$, while open addressing degrades dramatically as $\alpha$ approaches 1. Practical implementations maintain $\alpha$ within the range $[0.5, 0.75]$ by triggering resizes when the threshold is exceeded, achieving a balance between memory usage and lookup speed.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
