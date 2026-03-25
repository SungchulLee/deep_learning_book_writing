# Longest Common Substring

Finding the longest string that appears as a contiguous substring of two or more given strings is a fundamental problem with applications in bioinformatics (comparing DNA sequences), plagiarism detection, and data deduplication. The naive approach of checking all pairs of substrings takes $O(n^2 m)$ time for two strings of lengths $n$ and $m$. Using suffix trees or suffix arrays, the problem can be solved in $O(n + m)$ time. This section presents both the suffix tree approach and the suffix array approach.

## Problem Statement

Given two strings $S_1$ of length $n$ and $S_2$ of length $m$, find the longest string $w$ such that $w$ is a contiguous substring of both $S_1$ and $S_2$.

Formally:

$$
\text{LCS}(S_1, S_2) = \arg\max_{w} |w| \quad \text{such that } w \text{ is a substring of both } S_1 \text{ and } S_2
$$

The **length** of the LCS is often denoted $\text{lcstr}(S_1, S_2)$.

!!! warning "LCS vs LCS"
    Do not confuse **Longest Common Substring** (contiguous) with **Longest Common Subsequence** (not necessarily contiguous). The subsequence problem is solved by dynamic programming in $O(nm)$ time, while the substring problem is solved in $O(n + m)$ using suffix structures.

## Suffix Tree Approach

### Generalized Suffix Tree

Build a **generalized suffix tree** for both strings by concatenating them with distinct sentinels:

$$
T = S_1 \cdot \texttt{\#} \cdot S_2 \cdot \texttt{\$}
$$

where $\texttt{\#}$ and $\texttt{\$}$ are distinct characters not appearing in either string. The generalized suffix tree of $T$ contains all suffixes of both $S_1$ and $S_2$.

### Marking Internal Nodes

After building the tree, label each leaf based on which string its suffix belongs to:

- A leaf is an **$S_1$-leaf** if its suffix starts at a position $i < n$ (within $S_1$)
- A leaf is an **$S_2$-leaf** if its suffix starts at a position $i > n$ (within $S_2$)

An internal node $v$ is **shared** if its subtree contains at least one $S_1$-leaf and at least one $S_2$-leaf. This marking can be computed in $O(n + m)$ time by a bottom-up traversal.

### Finding the LCS

The LCS corresponds to the **deepest shared internal node** -- the shared node with the longest path label:

$$
\text{LCS} = \text{path}(v^*) \quad \text{where } v^* = \arg\max_{\substack{v \text{ shared} \\ v \text{ internal}}} \text{depth}(v)
$$

**Time complexity**: Building the generalized suffix tree takes $O(n + m)$ time (using Ukkonen's algorithm), and the bottom-up marking and maximum-depth search take $O(n + m)$ time. The total is:

$$
T(n, m) = O(n + m)
$$

## Suffix Array Approach

### Concatenation and Construction

Concatenate the strings as $T = S_1 \cdot \texttt{\#} \cdot S_2 \cdot \texttt{\$}$ and build the suffix array and LCP array of $T$.

### Scanning for the LCS

The LCS length equals the maximum LCP value between adjacent suffixes in the suffix array that originate from **different strings**:

$$
\text{lcstr}(S_1, S_2) = \max_{\substack{1 \leq k \leq |T| \\ \text{SA}[k] \text{ and } \text{SA}[k-1] \\ \text{from different strings}}} \text{LCP}[k]
$$

A suffix starting at position $i$ belongs to $S_1$ if $i \leq n - 1$ and to $S_2$ if $i \geq n + 1$ (position $n$ is the separator $\texttt{\#}$).

??? example "LCS of 'abcde' and 'bcdef'"
    Concatenate: $T = \texttt{abcde\#bcdef\$}$

    Build SA and LCP arrays, then scan for the maximum LCP between suffixes from different strings.

    The suffixes `bcde#bcdef$` (from $S_1$, position 1) and `bcdef$` (from $S_2$, position 6) share the prefix `bcde`, giving $\text{LCP} = 4$.

    Therefore, $\text{LCS} = \texttt{bcde}$ with length 4.

## Extension to Multiple Strings

The LCS problem extends naturally to $k$ strings $S_1, S_2, \ldots, S_k$. Concatenate all strings with distinct separators:

$$
T = S_1 \cdot \texttt{c}_1 \cdot S_2 \cdot \texttt{c}_2 \cdots S_k \cdot \texttt{c}_k
$$

Build the suffix tree or suffix array of $T$. For the suffix tree, find the deepest internal node whose subtree contains leaves from all $k$ strings. For the suffix array, use a sliding window on the LCP array to find the maximum LCP value across a range that contains suffixes from all $k$ strings.

**Time complexity**: $O(n_1 + n_2 + \cdots + n_k)$ for the suffix tree approach.

## Implementation

```python
"""
Longest Common Substring using suffix array and LCP array.
"""


# === Suffix Array Construction ===

def build_suffix_array(text: str) -> list[int]:
    """Build suffix array using prefix doubling."""
    n = len(text)
    rank = [ord(c) for c in text]
    sa = list(range(n))
    k = 1
    while k < n:
        def key(i, _k=k, _r=rank[:]):
            return (_r[i], _r[i + _k] if i + _k < n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        for j in range(1, n):
            prev = (rank[sa[j - 1]],
                    rank[sa[j - 1] + k] if sa[j - 1] + k < n else -1)
            curr = (rank[sa[j]],
                    rank[sa[j] + k] if sa[j] + k < n else -1)
            new_rank[sa[j]] = new_rank[sa[j - 1]] + (1 if curr != prev else 0)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


# === Kasai's Algorithm ===

def build_lcp(text: str, sa: list[int]) -> list[int]:
    """Compute LCP array using Kasai's algorithm."""
    n = len(sa)
    rank = [0] * n
    for k in range(n):
        rank[sa[k]] = k
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r > 0:
            j = sa[r - 1]
            while i + h < n and j + h < n and text[i + h] == text[j + h]:
                h += 1
            lcp[r] = h
            h = max(h - 1, 0)
        else:
            h = 0
    return lcp


# === Longest Common Substring ===

def longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring of s1 and s2.

    Parameters
    ----------
    s1 : str
        First input string.
    s2 : str
        Second input string.

    Returns
    -------
    str
        The longest common substring.
    """
    separator = "#"
    sentinel = "$"
    text = s1 + separator + s2 + sentinel
    n1 = len(s1)

    sa = build_suffix_array(text)
    lcp = build_lcp(text, sa)

    best_len = 0
    best_pos = 0

    for k in range(1, len(text)):
        pos_prev = sa[k - 1]
        pos_curr = sa[k]

        # Check that suffixes come from different strings
        from_s1_prev = pos_prev < n1
        from_s1_curr = pos_curr < n1

        if from_s1_prev != from_s1_curr and lcp[k] > best_len:
            best_len = lcp[k]
            best_pos = sa[k]

    return text[best_pos:best_pos + best_len]


# === Main ===

if __name__ == "__main__":
    s1 = "abcdefg"
    s2 = "cdefxyz"
    result = longest_common_substring(s1, s2)
    print(f"S1: '{s1}'")
    print(f"S2: '{s2}'")
    print(f"LCS: '{result}' (length {len(result)})")

    s1 = "banana"
    s2 = "ananas"
    result = longest_common_substring(s1, s2)
    print(f"\nS1: '{s1}'")
    print(f"S2: '{s2}'")
    print(f"LCS: '{result}' (length {len(result)})")
```

## Complexity Comparison

| Method | Time | Space |
|--------|------|-------|
| Naive (all substring pairs) | $O(n^2 m)$ | $O(1)$ |
| Dynamic programming | $O(nm)$ | $O(nm)$ or $O(\min(n,m))$ |
| Suffix tree | $O(n + m)$ | $O(n + m)$ |
| Suffix array + LCP | $O(n + m)$ | $O(n + m)$ |

The suffix-based approaches achieve optimal linear time and are preferred for large inputs.

## Reference

- Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press, Chapter 7.
- Hui, L. C. K. (1992). *Color set size problem with applications to string matching*. CPM 1992, LNCS 644, pp. 230-243.
