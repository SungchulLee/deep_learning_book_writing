# Universal Hashing

Any fixed hash function is vulnerable to adversarial inputs: an attacker who knows the hash function can choose keys that all collide, degrading every operation to $O(n)$. Universal hashing defends against this threat by randomly selecting a hash function from a carefully designed family at initialization time. Because the adversary does not know which function was chosen, no fixed input set can consistently cause poor performance.

## Motivation

Consider a web server that stores session data in a hash table using a fixed hash function $h$. An attacker who discovers $h$ (e.g., by reading the source code) can craft $n$ requests whose session IDs all satisfy $h(k_i) = 0$, forcing every operation into a single chain of length $n$. Universal hashing eliminates this vulnerability: the hash function is chosen randomly at startup, so the attacker cannot predict which inputs will collide.

## Definition

A family $\mathcal{H}$ of hash functions from universe $U$ to $\{0, 1, \ldots, m-1\}$ is **universal** if for every pair of distinct keys $k_1, k_2 \in U$ with $k_1 \neq k_2$:

$$
\Pr_{h \in \mathcal{H}}[h(k_1) = h(k_2)] \leq \frac{1}{m}
$$

where the probability is over the uniform random choice of $h$ from $\mathcal{H}$. Intuitively, the collision probability for any two distinct keys is no worse than what a truly random function would achieve.

This definition says nothing about any single hash function in $\mathcal{H}$. Individual functions may have poor distribution. The guarantee is that a **randomly chosen** function from $\mathcal{H}$ has low collision probability for any fixed pair of keys.

## Expected Performance

Universal hashing provides concrete performance guarantees for hash tables with chaining.

**Theorem.** Let $h$ be chosen uniformly at random from a universal family $\mathcal{H}$, and let $n$ keys be stored in a hash table with $m$ slots using chaining. For any key $k$, the expected length of the chain containing $k$ is at most:

$$
\mathbb{E}[\text{chain length at } h(k)] \leq 1 + \frac{n - 1}{m} = 1 + \alpha - \frac{1}{m}
$$

where $\alpha = n/m$ is the load factor.

*Proof sketch.* For a stored key $k$, define indicator random variables $X_i = \mathbf{1}[h(k_i) = h(k)]$ for each other key $k_i$. The chain length is $1 + \sum_{i \neq k} X_i$. By the universal property, $\mathbb{E}[X_i] \leq 1/m$, so by linearity of expectation:

$$
\mathbb{E}\left[\sum_{i \neq k} X_i\right] \leq \frac{n-1}{m}
$$

This gives $O(1)$ expected time for all operations when $\alpha = O(1)$, matching the performance of simple uniform hashing but now with a **provable** guarantee that holds for any input.

## The Carter-Wegman Family

Carter and Wegman (1979) constructed the first universal hash family. Choose a prime $p \geq |U|$ (the size of the key universe), and define:

$$
h_{a,b}(k) = ((ak + b) \bmod p) \bmod m
$$

where $a \in \{1, 2, \ldots, p-1\}$ and $b \in \{0, 1, \ldots, p-1\}$. The family is:

$$
\mathcal{H}_{p,m} = \{h_{a,b} : a \in \{1, \ldots, p-1\},\ b \in \{0, \ldots, p-1\}\}
$$

This family has $p(p-1)$ members and is universal.

**Proof of universality.** For distinct keys $k_1 \neq k_2$, the values $r_1 = (ak_1 + b) \bmod p$ and $r_2 = (ak_2 + b) \bmod p$ are distinct (since $a \neq 0$ and arithmetic is in $\mathbb{Z}_p$, a field). As $(a, b)$ ranges over all valid pairs, the pair $(r_1, r_2)$ takes each of the $p(p-1)$ possible values of distinct pairs in $\mathbb{Z}_p$ exactly once. The number of pairs $(r_1, r_2)$ with $r_1 \bmod m = r_2 \bmod m$ is at most:

$$
p(p-1) \cdot \frac{1}{m} \cdot \frac{m}{m} \leq \frac{p(p-1)}{m}
$$

Dividing by the total number of hash functions $p(p-1)$ gives:

$$
\Pr[h_{a,b}(k_1) = h_{a,b}(k_2)] \leq \frac{1}{m}
$$

??? example "Carter-Wegman Hash Family in Action"

    Let $p = 17$, $m = 6$, and suppose we randomly select $a = 3$, $b = 4$:

    $$
    h_{3,4}(k) = ((3k + 4) \bmod 17) \bmod 6
    $$

    For keys $\{5, 10, 15, 20, 25, 30\}$:

    $$
    \begin{array}{rcl}
    h_{3,4}(5) &=& (19 \bmod 17) \bmod 6 = 2 \bmod 6 = 2 \\
    h_{3,4}(10) &=& (34 \bmod 17) \bmod 6 = 0 \bmod 6 = 0 \\
    h_{3,4}(15) &=& (49 \bmod 17) \bmod 6 = 15 \bmod 6 = 3 \\
    h_{3,4}(20) &=& (64 \bmod 17) \bmod 6 = 13 \bmod 6 = 1 \\
    h_{3,4}(25) &=& (79 \bmod 17) \bmod 6 = 11 \bmod 6 = 5 \\
    h_{3,4}(30) &=& (94 \bmod 17) \bmod 6 = 9 \bmod 6 = 3
    \end{array}
    $$

    The 6 keys spread across 5 of 6 slots (one collision between 15 and 30), demonstrating good distribution.

## Stronger Universality

A family $\mathcal{H}$ is **strongly universal** (or 2-independent) if for all distinct $k_1, k_2 \in U$ and all $j_1, j_2 \in \{0, \ldots, m-1\}$:

$$
\Pr_{h \in \mathcal{H}}[h(k_1) = j_1 \text{ and } h(k_2) = j_2] = \frac{1}{m^2}
$$

Strong universality implies universality (set $j_1 = j_2$ and sum over $j_1$), but provides additional guarantees useful for variance analysis and perfect hashing constructions. The Carter-Wegman family over $\mathbb{Z}_p$ when $m = p$ (no second modular reduction) is strongly universal.

## Universal Hashing vs Fixed Hash Functions

| Property | Fixed Hash Function | Universal Hashing |
|---|---|---|
| Adversarial resistance | None | Guaranteed |
| Expected chain length | Depends on input | $\leq 1 + \alpha$ for any input |
| Per-lookup overhead | One hash computation | One hash computation |
| Initialization | None | Random selection of $a, b$ |
| Space overhead | None | Store $a, b$ ($O(1)$) |

The only cost of universal hashing is storing the random parameters $a$ and $b$, which is negligible. The performance guarantees hold for **any** input set, not just "typical" inputs.

## Summary

Universal hashing provides provable $O(1)$ expected-time guarantees for hash table operations against any input, including adversarial inputs. By randomly selecting a hash function from a universal family at initialization, the collision probability for any pair of keys is bounded by $1/m$. The Carter-Wegman construction $h_{a,b}(k) = ((ak + b) \bmod p) \bmod m$ provides a concrete, efficient universal family that is widely used in practice and forms the foundation for perfect hashing schemes.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
