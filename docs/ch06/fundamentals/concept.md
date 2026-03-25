# Hash Function Concept

Searching for a specific element in a collection is one of the most frequent operations in computing. Linear search examines every element in $O(n)$ time, and binary search on a sorted array improves this to $O(\log n)$. A natural question arises: can we do even better? Hash functions answer this question affirmatively by mapping keys directly to storage locations, achieving $O(1)$ **expected** time for lookups under reasonable assumptions.

## Motivation

Consider a university registrar that needs to look up student records by student ID. With $n = 50{,}000$ students, a linear scan examines up to 50,000 records per query, while binary search requires about 16 comparisons on a sorted list. A hash table, however, computes the storage location directly from the student ID, typically resolving the query in a single step. This dramatic speedup makes hashing indispensable in databases, compilers, caches, and nearly every large-scale system.

The following table summarizes the search performance of common data access strategies:

$$
\begin{array}{lll}
\textbf{Method} && \textbf{Time Complexity} \\
\hline
\text{Linear Search} && O(n) \\
\text{Binary Search (sorted array)} && O(\log n) \\
\text{Hash Table Lookup (expected)} && O(1) \\
\end{array}
$$

The $O(1)$ figure for hash tables is an **expected-case** bound that relies on the assumption of simple uniform hashing. In the worst case, all keys may collide into a single slot, degrading performance to $O(n)$. The design of good hash functions and collision resolution strategies (covered in subsequent sections) ensures that the worst case arises with negligibly small probability.

## Definition

A **hash function** is a mapping $h : U \to \{0, 1, \ldots, m-1\}$ from a universe $U$ of possible keys to a set of $m$ integer indices called **slots** or **buckets**. Given a key $k \in U$, the value $h(k)$ determines the position in the hash table where $k$ (or its associated data) is stored.

Formally, a hash table $T$ of size $m$ stores each key-value pair $(k, v)$ at index $h(k)$:

$$
T[h(k)] \leftarrow (k, v)
$$

Since $|U|$ is typically much larger than $m$, the mapping $h$ is not injective in general. When two distinct keys $k_1 \neq k_2$ satisfy $h(k_1) = h(k_2)$, a **collision** occurs. Managing collisions is a central challenge in hash table design.

## Simple Uniform Hashing Assumption

The theoretical analysis of hash tables often relies on the **Simple Uniform Hashing Assumption (SUHA)**: each key is equally likely to hash to any of the $m$ slots, independently of where other keys have hashed. Under SUHA, for a hash table with $n$ stored keys and $m$ slots, the expected number of keys per slot is:

$$
\alpha = \frac{n}{m}
$$

This ratio $\alpha$ is called the **load factor**. When $\alpha$ remains bounded by a constant, each slot contains $O(1)$ keys on average, which is why lookups take $O(1)$ expected time.

## Desirable Properties

A well-designed hash function should satisfy three key properties:

**Determinism.** For a given key $k$, the function $h(k)$ always returns the same value. Without determinism, a stored key could not be retrieved.

**Uniformity.** The hash values should be distributed as uniformly as possible across the $m$ slots. If $n$ keys are inserted and each slot receives approximately $n/m$ keys, collisions are minimized. Formally, under SUHA:

$$
\Pr[h(k) = j] = \frac{1}{m} \quad \text{for all } k \in U,\ j \in \{0, 1, \ldots, m-1\}
$$

**Efficiency.** Computing $h(k)$ should take $O(1)$ time (or at worst $O(|k|)$ time for variable-length keys, where $|k|$ denotes the length of the key representation). A hash function that is expensive to evaluate defeats the purpose of constant-time lookup.

## How Hashing Works

The basic operations on a hash table are **insert**, **search**, and **delete**:

1. **Insert** $(k, v)$: Compute $h(k)$ and store the pair at slot $T[h(k)]$.
2. **Search** $k$: Compute $h(k)$ and examine slot $T[h(k)]$ for the key.
3. **Delete** $k$: Compute $h(k)$, locate $k$ in slot $T[h(k)]$, and remove it.

Each operation begins with computing the hash, which takes $O(1)$ time. The remaining cost depends on how many keys occupy the same slot, which is governed by the load factor and the collision resolution strategy.

??? example "Integer Hashing with the Modulo Operator"

    The simplest hash function for integer keys uses the modulo operator. Given a table of size $m = 10$ and keys $\{12, 25, 37, 42, 58\}$, the hash values are:

    $$
    \begin{array}{rcl}
    h(12) &=& 12 \bmod 10 = 2 \\
    h(25) &=& 25 \bmod 10 = 5 \\
    h(37) &=& 37 \bmod 10 = 7 \\
    h(42) &=& 42 \bmod 10 = 2 \\
    h(58) &=& 58 \bmod 10 = 8 \\
    \end{array}
    $$

    Keys 12 and 42 both map to slot 2, producing a collision. The collision resolution strategy (chaining, open addressing, etc.) determines how both keys coexist in the table.

## Hash Functions for Non-Integer Keys

Real-world keys are often strings, floating-point numbers, or composite objects rather than simple integers. To hash such keys, we first convert them into an integer representation and then apply a standard integer hash function.

**String hashing.** A common approach treats each character as a digit in a positional number system. For a string $s = s_0 s_1 \cdots s_{L-1}$ of length $L$, a polynomial hash computes:

$$
h(s) = \left( \sum_{i=0}^{L-1} s_i \cdot r^{L-1-i} \right) \bmod m
$$

where $r$ is a chosen radix (often a small prime like 31 or 37) and $s_i$ is the numeric value of the $i$-th character. This formula ensures that strings with the same characters in different orders produce different hash values.

## Summary

A hash function maps keys from a large universe to a fixed range of table indices, enabling $O(1)$ expected-time operations under the simple uniform hashing assumption. The subsequent sections in this chapter explore specific hash function constructions (division method, multiplication method), strategies to handle the inevitable collisions, and techniques to maintain performance guarantees as the table grows.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [Hashing Technique - Simplified](https://www.youtube.com/watch?v=mFY0J5W8Udk&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=79)
