# LCP Array

A suffix array tells us the sorted order of all suffixes, but it does not directly reveal how much two adjacent suffixes in that order share. Knowing the **Longest Common Prefix (LCP)** between consecutive sorted suffixes unlocks several powerful capabilities: accelerating pattern search from $O(m \log n)$ to $O(m + \log n)$, counting the number of distinct substrings in $O(n)$ time, and simulating top-down traversals of the suffix tree without actually building it. This section defines the LCP array, illustrates it with an example, and shows how it enhances suffix array queries.

## Definition

Given a string $T[0..n]$ (with sentinel) and its suffix array $\text{SA}[0..n]$, the **LCP array** $\text{LCP}[0..n]$ is defined as:

$$
\text{LCP}[k] = \text{lcp}\bigl(\text{suffix}(\text{SA}[k-1]),\; \text{suffix}(\text{SA}[k])\bigr) \quad \text{for } k \geq 1
$$

where $\text{lcp}(s_1, s_2)$ denotes the length of the longest common prefix of strings $s_1$ and $s_2$. By convention, $\text{LCP}[0] = 0$ (there is no suffix before the first one in sorted order).

In words, $\text{LCP}[k]$ measures how many leading characters the $k$-th suffix in sorted order shares with the $(k-1)$-th suffix.

## Worked Example

For $T = \texttt{banana\$}$, the suffix array is $\text{SA} = [6, 5, 3, 1, 0, 4, 2]$:

| Rank $k$ | SA[$k$] | Suffix | LCP[$k$] | Shared prefix with previous |
|----------|---------|--------|----------|-----------------------------|
| 0 | 6 | `$` | 0 | (none) |
| 1 | 5 | `a$` | 0 | (no common prefix with `$`) |
| 2 | 3 | `ana$` | 1 | `a` (shared with `a$`) |
| 3 | 1 | `anana$` | 3 | `ana` (shared with `ana$`) |
| 4 | 0 | `banana$` | 0 | (no common prefix with `anana$`) |
| 5 | 4 | `na$` | 0 | (no common prefix with `banana$`) |
| 6 | 2 | `nana$` | 2 | `na` (shared with `na$`) |

Therefore:

$$
\text{LCP} = [0, 0, 1, 3, 0, 0, 2]
$$

## LCP and Pattern Search

With just the suffix array, binary search for a pattern $P$ of length $m$ requires $O(m \log n)$ time because each of the $O(\log n)$ comparisons costs $O(m)$. The LCP array enables a faster approach.

### The LCP-Enhanced Binary Search

During binary search, we maintain the LCP between the pattern $P$ and the left and right boundaries of the search range. Let $\ell_L = \text{lcp}(P, \text{suffix}(\text{SA}[L]))$ and $\ell_R = \text{lcp}(P, \text{suffix}(\text{SA}[R]))$ be the LCP values at the current boundaries.

At the midpoint $M$, instead of comparing $P$ against suffix($\text{SA}[M]$) from scratch, we use precomputed LCP information to skip $\min(\ell_L, \ell_R)$ characters. This reduces total comparisons across all binary search steps to $O(m + \log n)$:

$$
T_{\text{search}} = O(m + \log n)
$$

This matches the search time of suffix trees while using only the space of two arrays (SA and LCP).

## Counting Distinct Substrings

Every substring of $T$ is a prefix of some suffix. The total number of substrings (including duplicates) starting at suffix($\text{SA}[k]$) is $n - \text{SA}[k]$. However, $\text{LCP}[k]$ of these are shared with the previous suffix in sorted order and are therefore not new. The number of **distinct substrings** is:

$$
D = \sum_{k=0}^{n} \bigl(n - \text{SA}[k]\bigr) - \sum_{k=1}^{n} \text{LCP}[k]
$$

Simplifying:

$$
D = \frac{n(n+1)}{2} + (n+1) - \sum_{k=1}^{n} \text{LCP}[k]
$$

where the first term counts all substrings (including the sentinel) and the second term subtracts duplicates.

??? example "Distinct substrings of 'banana$'"
    For $T = \texttt{banana\$}$ with $n = 6$:

    - Total possible substrings: $\frac{7 \times 8}{2} = 28$ (but subtracting sentinel-only substrings)
    - Sum of LCP values: $0 + 1 + 3 + 0 + 0 + 2 = 6$
    - This gives 22 distinct substrings (including those containing the sentinel)
    - Excluding the sentinel, the distinct substrings of `banana` are: `a`, `an`, `ana`, `anan`, `anana`, `b`, `ba`, `ban`, `bana`, `banan`, `banana`, `n`, `na`, `nan`, `nana`, plus the empty string

## LCP Interval Tree

The LCP array also defines a hierarchy of **LCP intervals** that correspond to internal nodes of the suffix tree. An LCP interval $[i, j]$ with LCP value $\ell$ represents a set of suffixes that all share a common prefix of length at least $\ell$:

$$
[i, j] \text{ is an LCP interval with value } \ell \iff \text{LCP}[i] < \ell,\; \text{LCP}[j+1] < \ell,\; \text{and } \min_{i < k \leq j} \text{LCP}[k] = \ell
$$

This structure enables suffix tree algorithms to run on suffix arrays without constructing the tree explicitly.

## Key Properties

1. **Range minimum queries**: The LCP of any two suffixes $\text{SA}[i]$ and $\text{SA}[j]$ (not just adjacent ones) equals the minimum value in the LCP array between positions $i+1$ and $j$:

$$
\text{lcp}(\text{suffix}(\text{SA}[i]),\; \text{suffix}(\text{SA}[j])) = \min_{i < k \leq j} \text{LCP}[k]
$$

Using a range minimum query (RMQ) data structure, this can be answered in $O(1)$ time after $O(n)$ preprocessing.

2. **Sum bounds**: The sum of all LCP values satisfies:

$$
0 \leq \sum_{k=1}^{n} \text{LCP}[k] \leq \frac{n(n+1)}{2}
$$

The lower bound occurs when all characters are distinct; the upper bound occurs for a string like $\texttt{aaa...a}$.

3. **Relationship to suffix tree**: The LCP array encodes the same information as the internal edge lengths of the suffix tree. The depth of the lowest common ancestor of two leaves in the suffix tree equals the LCP of the corresponding suffixes.

## Reference

- Manber, U. and Myers, G. (1993). *Suffix arrays: A new method for on-line string searches*. SIAM Journal on Computing, 22(5), 935-948.
- Abouelhoda, M. I., Kurtz, S., and Ohlebusch, E. (2004). *Replacing suffix trees with enhanced suffix arrays*. Journal of Discrete Algorithms, 2(1), 53-86.
