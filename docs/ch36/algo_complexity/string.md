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

## Exercises

**Exercise 1.**
Compare the time complexities of brute-force string matching, KMP, and Rabin-Karp for a pattern of length $m$ in a text of length $n$.

??? success "Solution to Exercise 1"
    **Brute force**: $O(nm)$ worst case (try every alignment, compare up to $m$ characters). **KMP**: $O(n + m)$ worst case ($O(m)$ to build the failure function, $O(n)$ to scan the text). Never backtracks in the text. **Rabin-Karp**: $O(n + m)$ expected time using rolling hashes. Worst case: $O(nm)$ if all hash values collide (every position is a false positive requiring character-by-character verification). KMP is preferred for guaranteed linear time. Rabin-Karp is preferred for multiple pattern matching (hash each pattern, use a hash set for $O(1)$ lookup). $\square$

---

**Exercise 2.**
Building a suffix array takes $O(n \log n)$ or $O(n)$ time. Once built, how fast can we find all occurrences of a pattern of length $m$? Compare with a suffix tree.

??? success "Solution to Exercise 2"
    **Suffix array**: binary search for the pattern's position in the sorted suffix array. Each comparison takes $O(m)$ (compare $m$ characters). Total: $O(m \log n)$. With an LCP array, this can be improved to $O(m + \log n)$. Finding all $k$ occurrences: $O(m \log n + k)$. **Suffix tree**: traverse from the root following the pattern's characters. Each character match takes $O(1)$ (with edge labels). Total: $O(m)$ to find the locus node, then $O(k)$ to enumerate all leaves below it. Total: $O(m + k)$. Suffix trees are faster for pattern matching but use 10--20x more memory than suffix arrays. Suffix arrays with LCP arrays are the practical choice for large texts. $\square$

---

**Exercise 3.**
The longest common substring of two strings of lengths $m$ and $n$ can be found in $O(mn)$ via DP or $O((m+n) \log(m+n))$ via suffix arrays. Describe both approaches.

??? success "Solution to Exercise 3"
    **DP approach**: build a table $dp[i][j]$ where $dp[i][j]$ is the length of the longest common suffix ending at positions $i$ and $j$. If $s_1[i] = s_2[j]$, then $dp[i][j] = dp[i-1][j-1] + 1$; else $dp[i][j] = 0$. The answer is $\max(dp[i][j])$. Time: $O(mn)$, space $O(\min(m,n))$ with rolling array. **Suffix array approach**: concatenate the two strings with a separator: $s_1 \# s_2$. Build a suffix array and LCP array in $O((m+n) \log(m+n))$ or $O(m+n)$. The longest common substring is the maximum LCP value between adjacent suffixes that belong to different strings. Scan the LCP array in $O(m+n)$. Total: $O((m+n) \log(m+n))$ or $O(m+n)$ with linear suffix array construction. $\square$

---

**Exercise 4.**
Aho-Corasick matches multiple patterns simultaneously in $O(n + m + z)$ time, where $z$ is the number of matches. Explain why this is faster than running KMP for each pattern separately.

??? success "Solution to Exercise 4"
    With $k$ patterns of total length $m$, running KMP separately costs $O(k \cdot n + m)$: each pattern requires a full scan of the text. For $k = 1000$ and $n = 10^6$, this is $10^9$ operations. Aho-Corasick builds a trie of all patterns ($O(m)$), augmented with failure links similar to KMP's failure function. The text is scanned once ($O(n)$), following trie transitions and failure links. At each position, all matching patterns are reported. Total: $O(n + m + z)$. The key savings: the text is scanned only once regardless of $k$. For the example above: $O(10^6 + m + z)$, which is $1000\times$ faster than separate KMP runs. Aho-Corasick is used in intrusion detection systems, antivirus scanners, and search engines for multi-pattern matching. $\square$

---

**Exercise 5.**
Explain why the Z-algorithm and KMP achieve the same $O(n + m)$ complexity for single-pattern matching but use different auxiliary arrays. What is the relationship between the Z-array and KMP's failure function?

??? success "Solution to Exercise 5"
    Both KMP and the Z-algorithm preprocess the pattern (or the concatenation $P \# T$) in linear time. **KMP's failure function** $\pi[i]$: the length of the longest proper prefix of $P[0..i]$ that is also a suffix. It enables skipping redundant comparisons by shifting the pattern. **Z-array** $Z[i]$: the length of the longest substring starting at $i$ that matches a prefix of the string. It directly identifies matches (if $Z[i] \ge m$, a match starts at $i - m - 1$ in the text). The two are related: $\pi$ and $Z$ encode the same information about the string's self-overlap structure, but in dual form. Given $Z$, one can compute $\pi$ in $O(n)$ and vice versa. KMP processes the text left-to-right with a state machine; the Z-algorithm processes the concatenation with a window-based approach. Both make exactly $O(n + m)$ character comparisons. $\square$
