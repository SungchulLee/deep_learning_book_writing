# Hash Function Design

Choosing a hash function is one of the most consequential decisions in hash table implementation. A poorly designed hash function clusters keys into a few slots, causing frequent collisions and degrading expected $O(1)$ operations toward $O(n)$. A well-designed hash function distributes keys uniformly across all available slots, keeping each chain short and lookups fast. This section examines the principles that guide hash function design and the criteria for evaluating quality.

## Design Goals

A hash function $h : U \to \{0, 1, \ldots, m-1\}$ maps keys from a universe $U$ to $m$ slots. The ideal hash function satisfies three objectives simultaneously:

**Uniform distribution.** Each slot should receive approximately the same number of keys. If $n$ keys are inserted, the ideal count per slot is $n/m$. Formally, under the simple uniform hashing assumption:

$$
\Pr[h(k) = j] = \frac{1}{m} \quad \text{for all } j \in \{0, 1, \ldots, m-1\}
$$

**Independence.** The slot assigned to one key should not influence the slot assigned to another. This prevents systematic clustering patterns that arise when keys share structure (e.g., consecutive integers, strings with common prefixes).

**Computational efficiency.** The hash function must be computable in $O(1)$ time for fixed-size keys, so that the cost of hashing does not dominate the cost of the table operation.

## Exploiting Key Structure

In practice, hash functions rarely see keys drawn uniformly at random from $U$. Instead, keys exhibit patterns: IP addresses cluster by subnet, dictionary words share common prefixes, and database primary keys are often sequential. A good hash function must scatter these structured inputs uniformly.

The general strategy is to extract the "information content" of a key and spread it across the full range of hash values. Two classical approaches achieve this:

**Division method.** Compute the hash as the remainder when dividing the key by the table size:

$$
h(k) = k \bmod m
$$

This method is simple and fast but sensitive to the choice of $m$. If $m$ is a power of 2, only the low-order bits of $k$ determine $h(k)$, which produces poor distribution for keys with patterns in those bits. Choosing $m$ to be a prime not close to a power of 2 mitigates this problem.

**Multiplication method.** Multiply the key by a constant $A \in (0, 1)$, extract the fractional part, and scale by $m$:

$$
h(k) = \lfloor m \cdot (kA \bmod 1) \rfloor
$$

This method is less sensitive to the choice of $m$ because the multiplication by an irrational-like constant disperses keys across the full range regardless of the table size.

## Choosing the Table Size

The table size $m$ interacts with the hash function to determine distribution quality:

- **Prime table sizes** work well with the division method because $k \bmod p$ uses all bits of $k$ when $p$ is prime.
- **Power-of-two sizes** enable fast modular reduction via bitwise AND ($k\ \&\ (m-1)$) but require hash functions that already mix all bits of $k$ (e.g., the multiplication method or bit-mixing functions).
- **Avoid sizes with small factors.** If $m$ and the key distribution share a common factor $d$, only $m/d$ slots receive keys, wasting $(1 - 1/d)$ of the table.

??? example "Effect of Table Size on Distribution"

    Consider keys $\{10, 20, 30, 40, 50, 60, 70, 80\}$ with $m = 10$:

    $$
    h(k) = k \bmod 10 = 0 \quad \text{for all } k
    $$

    Every key maps to slot 0 because all keys are multiples of 10. Changing to $m = 7$:

    $$
    \begin{array}{rcl}
    h(10) = 3, \quad h(20) = 6, \quad h(30) = 2, \quad h(40) = 5 \\
    h(50) = 1, \quad h(60) = 4, \quad h(70) = 0, \quad h(80) = 3
    \end{array}
    $$

    With $m = 7$, the keys spread across 7 distinct slots (with one collision), demonstrating how a prime table size avoids the pathological clustering.

## Composing Hash Functions for Complex Keys

Real-world keys are often composite: a record might be keyed by (last name, first name, date of birth). A hash function for composite keys combines the hash values of individual components. Two standard composition techniques are:

**Polynomial accumulation.** Treat each component as a coefficient of a polynomial evaluated at a fixed base $r$:

$$
h(k_1, k_2, \ldots, k_d) = \left( \sum_{i=1}^{d} k_i \cdot r^{d-i} \right) \bmod m
$$

The base $r$ is typically a small prime (31, 37, or 131). This approach ensures that permuting the components changes the hash value.

**XOR with rotation.** Combine component hashes using bitwise XOR with bit rotations to break symmetry:

$$
h(k_1, k_2, \ldots, k_d) = h_1(k_1) \oplus \text{rot}(h_2(k_2), s_2) \oplus \cdots \oplus \text{rot}(h_d(k_d), s_d)
$$

where $\text{rot}(x, s)$ rotates the bits of $x$ left by $s$ positions and $\oplus$ denotes bitwise XOR.

## Evaluating Hash Function Quality

Given a hash function and a set of $n$ keys distributed among $m$ slots, the quality of the distribution can be measured by the **chi-squared statistic**:

$$
\chi^2 = \sum_{j=0}^{m-1} \frac{(c_j - n/m)^2}{n/m}
$$

where $c_j$ is the number of keys in slot $j$. Under perfect uniform hashing, $\chi^2$ follows a chi-squared distribution with $m - 1$ degrees of freedom. A hash function whose $\chi^2$ value falls within the expected range provides evidence of uniform distribution.

## Common Pitfalls

!!! warning "Hash Function Design Pitfalls"

    - **Using only part of the key.** A hash that ignores some bits or fields of the key cannot distinguish keys that differ only in the ignored parts.
    - **Symmetric composition.** Using $h(a, b) = h(a) + h(b)$ makes $(a, b)$ and $(b, a)$ collide. Polynomial accumulation or rotation avoids this.
    - **Identity hash.** Setting $h(k) = k \bmod m$ for sequential integer keys with a power-of-two $m$ produces sequential slot assignments, creating clustering under linear probing.
    - **Ignoring the key distribution.** A hash function that is uniform for random inputs may perform poorly on structured real-world data. Universal hashing (covered in a later section) provides distribution guarantees regardless of the input.

## Summary

Effective hash function design requires balancing uniform distribution, computational efficiency, and robustness to structured inputs. The division and multiplication methods provide concrete constructions, while the choice of table size and composition strategy determine whether these constructions achieve their theoretical potential. When the key distribution is unknown or adversarial, universal hashing offers provable guarantees that no fixed hash function can match.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
