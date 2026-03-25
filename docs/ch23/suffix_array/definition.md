# Suffix Array Definition

Searching for patterns inside a long text is one of the most fundamental problems in computer science, with applications ranging from genome analysis to full-text search engines. A naive approach checks every possible starting position, but this becomes impractical for texts with millions or billions of characters. The **suffix array** provides an elegant solution: by sorting all suffixes of the text, binary search can locate any pattern in $O(m \log n)$ time, where $m$ is the pattern length and $n$ is the text length. This section defines the suffix array formally and illustrates its construction with a concrete example.

## Suffixes of a String

Let $T[0..n-1]$ be a string of length $n$ over a finite alphabet $\Sigma$. In practice, we often append a special sentinel character $\$$ (lexicographically smaller than every character in $\Sigma$) to obtain $T[0..n]$ of length $n+1$. The sentinel ensures that no suffix is a prefix of another suffix, which simplifies the sorted ordering.

The $i$-th **suffix** of $T$ is the substring starting at position $i$ and extending to the end of the string:

$$
\text{suffix}(i) = T[i..n]
$$

For a string of length $n+1$ (including the sentinel), there are exactly $n+1$ suffixes, one for each starting position $0, 1, \ldots, n$.

??? example "Suffixes of 'banana$'"
    For $T = \texttt{banana\$}$ (length 7), the suffixes are:

    | Index $i$ | suffix($i$) |
    |-----------|-------------|
    | 0 | `banana$` |
    | 1 | `anana$` |
    | 2 | `nana$` |
    | 3 | `ana$` |
    | 4 | `na$` |
    | 5 | `a$` |
    | 6 | `$` |

## Formal Definition

The **suffix array** $\text{SA}$ of a string $T[0..n]$ is a permutation of the indices $\{0, 1, \ldots, n\}$ such that the suffixes appear in lexicographic order:

$$
T[\text{SA}[0]..n] < T[\text{SA}[1]..n] < \cdots < T[\text{SA}[n]..n]
$$

where $<$ denotes strict lexicographic comparison. Equivalently, $\text{SA}[k]$ is the starting index of the suffix that ranks $k$-th in the sorted order.

The suffix array can also be viewed as the result of sorting all suffix indices by their corresponding suffix strings:

$$
\text{SA} = \text{argsort}\bigl(\text{suffix}(0),\; \text{suffix}(1),\; \ldots,\; \text{suffix}(n)\bigr)
$$

The **inverse suffix array** $\text{SA}^{-1}$ maps each suffix position to its rank in the sorted order:

$$
\text{SA}^{-1}[i] = k \quad \iff \quad \text{SA}[k] = i
$$

In other words, $\text{SA}^{-1}[i]$ tells us the rank of suffix($i$) among all sorted suffixes.

## Worked Example

Consider the string $T = \texttt{banana\$}$. Sorting all suffixes lexicographically produces the following order:

| Rank $k$ | SA[$k$] | Suffix |
|----------|---------|--------|
| 0 | 6 | `$` |
| 1 | 5 | `a$` |
| 2 | 3 | `ana$` |
| 3 | 1 | `anana$` |
| 4 | 0 | `banana$` |
| 5 | 4 | `na$` |
| 6 | 2 | `nana$` |

Therefore:

$$
\text{SA} = [6, 5, 3, 1, 0, 4, 2]
$$

The inverse suffix array is:

$$
\text{SA}^{-1} = [4, 3, 6, 2, 5, 1, 0]
$$

For instance, $\text{SA}^{-1}[0] = 4$ because `banana$` (the suffix starting at position 0) has rank 4 in the sorted order.

## Pattern Searching with a Suffix Array

Once the suffix array is built, searching for a pattern $P[0..m-1]$ in $T$ reduces to finding all suffixes that begin with $P$. Since the suffix array stores suffixes in sorted order, all matches form a contiguous range. Binary search locates the leftmost and rightmost positions in $O(m \log n)$ time.

The search procedure finds the range $[\ell, r]$ such that for all $k \in [\ell, r]$, the suffix starting at $\text{SA}[k]$ has $P$ as a prefix. The number of occurrences is $r - \ell + 1$, and each occurrence starts at position $\text{SA}[k]$.

??? example "Searching for 'ana' in 'banana$'"
    With $P = \texttt{ana}$ and $\text{SA} = [6, 5, 3, 1, 0, 4, 2]$:

    - Binary search finds that suffixes at ranks 2 and 3 begin with `ana`
    - $\text{SA}[2] = 3$ corresponds to `ana$`
    - $\text{SA}[3] = 1$ corresponds to `anana$`
    - Pattern `ana` occurs at positions 1 and 3 in the original string

## Comparison with Suffix Trees

The suffix array is closely related to the **suffix tree**, which stores all suffixes in a compressed trie. While the suffix tree supports the same queries, it requires significantly more memory (typically 10-20 times the text size in practice). The suffix array achieves similar functionality with just $n$ integers of storage, making it far more space-efficient.

| Property | Suffix Array | Suffix Tree |
|----------|-------------|-------------|
| Space | $O(n)$ integers | $O(n)$ but large constant |
| Construction | $O(n)$ optimal | $O(n)$ (Ukkonen) |
| Pattern search | $O(m \log n)$ or $O(m + \log n)$ with LCP | $O(m)$ |
| Practical memory | ~$4n$ bytes | ~$20n$ bytes |

When augmented with the **Longest Common Prefix (LCP) array**, the suffix array can match the $O(m + \log n)$ search time of suffix trees while retaining its space advantage.

## Key Properties

Several properties of suffix arrays follow directly from the definition:

1. **Uniqueness**: With the sentinel character, the suffix array is a unique permutation of $\{0, 1, \ldots, n\}$ because no two suffixes are identical.

2. **Invertibility**: The suffix array $\text{SA}$ and its inverse $\text{SA}^{-1}$ are each other's inverses as permutations, so $\text{SA}[\text{SA}^{-1}[i]] = i$ and $\text{SA}^{-1}[\text{SA}[k]] = k$.

3. **Adjacent suffixes share prefixes**: Suffixes that are adjacent in the suffix array tend to share long common prefixes, which is the basis for the LCP array.

4. **All substrings are represented**: Every substring of $T$ is a prefix of some suffix, so the suffix array implicitly encodes all substrings of $T$.

## Reference

- Manber, U. and Myers, G. (1993). *Suffix arrays: A new method for on-line string searches*. SIAM Journal on Computing, 22(5), 935-948.
- Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press.
