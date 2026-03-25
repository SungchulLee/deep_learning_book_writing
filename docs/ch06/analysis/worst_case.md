# Worst-Case Analysis

The $O(1)$ expected-time guarantee for hash table operations is an **average-case** result: it holds when keys distribute uniformly across slots. In the worst case, all $n$ keys collide into a single slot, and every operation degrades to $\Theta(n)$ time. Understanding when and why this worst case arises -- and how to defend against it -- is essential for building robust systems.

## The Worst-Case Bound

For any deterministic hash function $h : U \to \{0, 1, \ldots, m-1\}$ and any hash table of size $m$, there exists a set $S$ of $n$ keys such that:

$$
h(k) = j \quad \text{for all } k \in S
$$

for some fixed slot $j$. This is a consequence of the pigeonhole principle: since $|U| \gg m$, at least $|U|/m$ keys in the universe map to the same slot. An adversary who knows $h$ can select $n$ of these keys.

When all $n$ keys occupy a single chain, the hash table degenerates into an unsorted linked list:

- **Search:** $\Theta(n)$ (traverse the entire chain)
- **Insert (with duplicate check):** $\Theta(n)$
- **Delete:** $\Theta(n)$

This $\Theta(n)$ worst case applies to **any** collision resolution strategy -- chaining, linear probing, quadratic probing, or double hashing -- because all $n$ elements occupy the same logical bucket.

## When Does the Worst Case Arise?

The worst case is not merely a theoretical curiosity. It arises in practice in several scenarios:

**Adversarial inputs.** In web servers, an attacker can craft HTTP requests with parameters whose keys all hash to the same slot, causing denial-of-service through hash table degradation. This attack has been demonstrated against multiple web frameworks (PHP, Java, Python, Ruby) using knowledge of their hash functions.

**Systematic key patterns.** When keys share structure that interacts with the hash function, accidental collisions occur. For example:

- Keys that are multiples of the table size: $h(k) = k \bmod m = 0$ for all $k = cm$.
- String keys with a common prefix when the hash function is prefix-sensitive.
- Sequential integer keys with a power-of-two table size, causing clustering in linear probing.

**Poor hash function design.** A hash function that ignores part of the key (e.g., only the last few bits) cannot distinguish keys that differ only in the ignored bits.

## Lower Bound on Worst-Case Search

The following theorem formalizes the impossibility of avoiding the $\Theta(n)$ worst case with any deterministic hash function.

**Theorem.** For any deterministic hash function $h : U \to \{0, 1, \ldots, m-1\}$ with $|U| \geq nm$, there exists a set $S \subseteq U$ of $n$ keys such that $|h(S)| = 1$ (all keys hash to a single slot).

*Proof.* By the pigeonhole principle, at least one slot $j$ has:

$$
|\{k \in U : h(k) = j\}| \geq \frac{|U|}{m} \geq n
$$

Choose $n$ keys from this preimage. All map to slot $j$, so $|h(S)| = 1$. $\square$

This theorem shows that no single hash function can guarantee sub-linear worst-case search for all possible input sets. The defense must come from randomization (universal hashing) or structural guarantees (perfect hashing).

## Defenses Against Worst-Case Behavior

Several techniques transform the $\Theta(n)$ worst case into a more manageable bound:

### Universal Hashing

By choosing the hash function randomly from a universal family at initialization time, no fixed adversarial input set can cause consistent worst-case behavior. For any set of $n$ keys:

$$
\mathbb{E}[\text{chain length at slot } h(k)] \leq 1 + \frac{n-1}{m}
$$

The adversary cannot exploit a hash function they do not know. While the worst case for any single hash function in the family may still be $\Theta(n)$, the probability of selecting that specific hash function is negligible.

### Perfect Hashing

For a static key set known in advance, the FKS two-level scheme constructs a hash function with **zero collisions**, achieving $O(1)$ worst-case lookup in $O(n)$ space.

### Balanced Trees per Bucket

Some implementations replace chains with balanced binary search trees (e.g., red-black trees) when a chain exceeds a threshold. This bounds the worst-case per-bucket search time to $O(\log n_j)$ where $n_j$ is the number of keys in bucket $j$.

Java's `HashMap` switches from linked lists to red-black trees when a chain exceeds 8 elements, limiting the worst-case search time to $O(\log n)$ even when all keys collide.

### Cryptographic Hashing

Using a keyed cryptographic hash (e.g., SipHash) makes it computationally infeasible for an adversary to find colliding keys without knowing the secret key. Python (since version 3.3), Rust, and many other languages use SipHash-based hashing to defend against hash-flooding attacks.

## Worst Case vs Expected Case

The gap between worst-case and expected-case performance highlights a fundamental design tradeoff:

| Aspect | Expected case | Worst case |
|---|---|---|
| Search time | $\Theta(1 + \alpha)$ | $\Theta(n)$ |
| Assumption | Simple Uniform Hashing Assumption (SUHA) or universal hashing | Adversarial input |
| Probability | Overwhelmingly likely | Requires specific input |
| Defense | Good hash function | Randomization or perfect hashing |

In practice, the expected case dominates. Measurements of real hash table workloads show that chain lengths closely follow the Poisson distribution with mean $\alpha$, confirming that the uniform distribution assumption holds for most real-world key distributions.

## Probabilistic Worst-Case Bounds

Even under universal hashing, extremely long chains can occur with small probability. The **maximum chain length** in a hash table with $n$ keys and $m = n$ slots under universal hashing satisfies:

$$
\Pr[\max_j L_j > c \cdot \log n] \leq \frac{1}{n^{c-1}}
$$

for a constant $c > 1$. This means that the longest chain is $O(\log n)$ with high probability, giving a probabilistic worst-case bound of $O(\log n)$ per operation.

With stronger hash families (e.g., $O(\log n)$-independent hash functions), the maximum chain length is $O(\log n / \log \log n)$ with high probability, matching the performance of balanced binary search trees.

??? example "Pathological vs Uniform Distribution"

    Consider $n = 8$ keys and $m = 4$ slots.

    **Pathological case** (all keys hash to slot 0):

    | Slot | Keys | Chain length |
    |---|---|---|
    | 0 | $k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8$ | 8 |
    | 1 | -- | 0 |
    | 2 | -- | 0 |
    | 3 | -- | 0 |

    Search time: $\Theta(8) = \Theta(n)$.

    **Uniform case** (keys distributed evenly):

    | Slot | Keys | Chain length |
    |---|---|---|
    | 0 | $k_1, k_5$ | 2 |
    | 1 | $k_2, k_6$ | 2 |
    | 2 | $k_3, k_7$ | 2 |
    | 3 | $k_4, k_8$ | 2 |

    Search time: $\Theta(1 + \alpha) = \Theta(1 + 2) = \Theta(1)$ since $\alpha = n/m = 2$ is a constant.

## Summary

The worst-case search time for any deterministic hash function is $\Theta(n)$, arising when all keys collide into a single slot. This worst case is not merely theoretical: adversarial inputs, systematic key patterns, and poor hash function design can all trigger it in practice. Universal hashing reduces the worst case to $O(\log n)$ with high probability, perfect hashing eliminates it entirely for static key sets, and hybrid approaches (balanced trees per bucket, cryptographic hashing) provide practical defenses for dynamic settings.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
