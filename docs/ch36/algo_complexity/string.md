# String Algorithm Complexities

Strings introduce challenges that general sequences do not: pattern matching must
handle overlapping occurrences, suffix structures must represent exponentially many
substrings in linear space, and edit operations define a rich metric space.
This page summarizes the time and space complexity of every major string algorithm.

## Single-Pattern Matching

Given a text of length $n$ and a pattern of length $m$, find all occurrences of the
pattern in the text.

| Algorithm | Preprocessing | Search | Total | Space |
|---|---|---|---|---|
| Naive | $O(1)$ | $O(nm)$ | $O(nm)$ | $O(1)$ |
| KMP | $O(m)$ | $O(n)$ | $O(n + m)$ | $O(m)$ |
| Boyer-Moore | $O(m + \Sigma)$ | $O(n/m)$ best, $O(nm)$ worst | $O(n + m + \Sigma)$ | $O(m + \Sigma)$ |
| Rabin-Karp | $O(m)$ | $O(n)$ expected, $O(nm)$ worst | $O(n + m)$ expected | $O(1)$ |
| Z-algorithm | $O(n + m)$ | included | $O(n + m)$ | $O(n + m)$ |

Here $\Sigma$ is the alphabet size. KMP and the Z-algorithm achieve $O(n + m)$
worst-case time by using a failure function or Z-array to avoid re-scanning matched
characters.

!!! tip "Boyer-Moore in Practice"
    Boyer-Moore's best case of $O(n/m)$ occurs when mismatches happen at the end
    of the pattern, allowing large shifts. For natural-language text with a large
    alphabet, Boyer-Moore is often the fastest algorithm in practice despite its
    $O(nm)$ worst case.

## Multiple-Pattern Matching

Search for $k$ patterns simultaneously in a text of length $n$.

| Algorithm | Preprocessing | Search | Total | Space |
|---|---|---|---|---|
| Aho-Corasick | $O(M)$ | $O(n + z)$ | $O(M + n + z)$ | $O(M \cdot \Sigma)$ |
| Rabin-Karp (multi) | $O(M)$ | $O(n \cdot k)$ expected | $O(M + nk)$ | $O(k)$ |

Here $M = \sum m_i$ is the total pattern length and $z$ is the number of matches.
Aho-Corasick builds a trie with failure links, giving $O(n + z)$ search time
regardless of the number of patterns.

## Suffix Structures

Suffix arrays and suffix trees represent all suffixes of a string, enabling powerful
substring queries.

| Structure | Construction | Space | Pattern Search | LCP Query |
|---|---|---|---|---|
| Suffix array | $O(n \log n)$ or $O(n)$ | $O(n)$ | $O(m \log n)$ | $O(1)$ with LCP array |
| Suffix array + LCP | $O(n)$ (Kasai) | $O(n)$ | $O(m + \log n)$ | $O(1)$ |
| Suffix tree | $O(n)$ (Ukkonen) | $O(n \cdot \Sigma)$ | $O(m)$ | $O(1)$ |
| Suffix automaton | $O(n)$ | $O(n)$ | $O(m)$ | via parent links |

!!! warning "Suffix Tree Space"
    While suffix trees provide $O(m)$ pattern search, they use $O(n \cdot \Sigma)$
    space in practice (large constant factors). Suffix arrays with LCP arrays are
    preferred when space is tight, at the cost of slightly slower queries.

## String Distance and Comparison

| Algorithm | Time | Space | Problem |
|---|---|---|---|
| Edit distance (Levenshtein) | $O(mn)$ | $O(mn)$ or $O(\min(m,n))$ | Minimum edits to transform one string into another |
| LCS (longest common subseq.) | $O(mn)$ | $O(\min(m,n))$ | Longest common subsequence |
| LCS (longest common substr.) | $O(mn)$ | $O(mn)$ | Longest common substring |
| LCS via suffix array | $O(n + m)$ | $O(n + m)$ | Concatenate with separator |
| Hamming distance | $O(n)$ | $O(1)$ | Count positions that differ |

The edit distance recurrence is:

$$
dp[i][j] = \min\bigl(dp[i-1][j] + 1,\; dp[i][j-1] + 1,\; dp[i-1][j-1] + \delta_{ij}\bigr)
$$

where $\delta_{ij} = 0$ if $s_1[i] = s_2[j]$ and $\delta_{ij} = 1$ otherwise.

## Palindrome Algorithms

| Algorithm | Time | Space | Purpose |
|---|---|---|---|
| Manacher's | $O(n)$ | $O(n)$ | Find all maximal palindromic substrings |
| Eertree (palindromic tree) | $O(n)$ | $O(n)$ | Count distinct palindromic substrings |
| DP palindrome check | $O(n^2)$ | $O(n^2)$ | All palindromic substrings |
| Longest palindromic subseq. | $O(n^2)$ | $O(n^2)$ or $O(n)$ | DP on the string |

## Hashing for Strings

Rolling hash functions enable $O(1)$ per-position hash computation after $O(m)$
preprocessing.

| Method | Hash Time | Collision Probability | Space |
|---|---|---|---|
| Polynomial rolling hash | $O(1)$ per shift | $O(m/p)$ per comparison | $O(1)$ |
| Double hashing | $O(1)$ per shift | $O(m/p^2)$ per comparison | $O(1)$ |
| Prefix hash array | $O(1)$ per substring | $O(1/p)$ per comparison | $O(n)$ |

Here $p$ is the modulus. Using two independent hash functions (double hashing) reduces
the false-positive rate quadratically.

## Practical Feasibility

| $n$ | Naive $O(nm)$ | KMP $O(n)$ | Suffix array $O(n \log n)$ | Suffix tree $O(n)$ |
|---|---|---|---|---|
| $10^4$ | fast | instant | instant | instant |
| $10^5$ | moderate | fast | fast | fast |
| $10^6$ | slow | fast | fast | moderate (space) |
| $10^7$ | infeasible | moderate | moderate | slow (space) |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Gusfield, D. *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press, 1997.
