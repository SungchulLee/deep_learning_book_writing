# Suffix Tree Definition

Many string algorithms need to answer queries about all substrings of a text simultaneously: finding the longest repeated substring, locating all occurrences of a pattern, or computing the longest common substring of two strings. A data structure that indexes every suffix of a text provides immediate access to every substring, since every substring is a prefix of some suffix. The **suffix tree** is a compressed trie of all suffixes that achieves this indexing in $O(n)$ space and enables $O(m)$-time pattern matching, where $m$ is the pattern length. This section defines the suffix tree formally and examines its structural properties.

## From Suffix Trie to Suffix Tree

The **suffix trie** of a string $T[0..n]$ is an ordinary trie containing all $n+1$ suffixes of $T$. Each edge is labeled with a single character, and each suffix corresponds to a path from the root to a leaf. While conceptually simple, the suffix trie has $O(n^2)$ nodes in the worst case because every substring of $T$ corresponds to a distinct node.

The **suffix tree** compresses the suffix trie by merging every chain of nodes with exactly one child into a single edge. Each edge in the suffix tree is labeled with a **substring** of $T$ rather than a single character. This compression ensures that the suffix tree has at most $n + 1$ leaves (one per suffix) and at most $n$ internal nodes, for a total of $O(n)$ nodes.

!!! note "Sentinel character"
    To ensure that no suffix is a prefix of another (which would cause some suffixes to end at internal nodes rather than leaves), we append a unique sentinel character $\$$ that does not appear elsewhere in $T$. With the sentinel, every suffix ends at a distinct leaf.

## Formal Definition

The **suffix tree** $\mathcal{T}$ of a string $T[0..n]$ (with sentinel $T[n] = \$$) is a rooted tree satisfying the following properties:

1. **Exactly $n+1$ leaves**, labeled $0, 1, \ldots, n$. Leaf $i$ represents suffix($i$) = $T[i..n]$.

2. **Every internal node** (except possibly the root) has **at least two children**.

3. **Each edge** is labeled with a non-empty substring of $T$. The labels of edges from any node to its children begin with **distinct characters** (this is the branching property that makes the trie compressed).

4. **Path label**: The concatenation of edge labels on the path from the root to leaf $i$ equals suffix($i$) = $T[i..n]$.

5. **Edge representation**: Each edge label is stored as a pair of indices $(l, r)$ representing the substring $T[l..r]$, rather than copying the characters. This ensures $O(n)$ total space.

## Node Properties

Each node $v$ in the suffix tree has an associated **path label** $\text{path}(v)$, which is the concatenation of all edge labels from the root to $v$. The **string depth** of $v$ is the length of its path label: $\text{depth}(v) = |\text{path}(v)|$.

For leaf $i$, the path label is the full suffix: $\text{path}(\text{leaf}_i) = T[i..n]$.

For an internal node $v$, the path label $\text{path}(v)$ is a **repeated substring** of $T$ -- it appears in $T$ starting at every position corresponding to a leaf in the subtree rooted at $v$.

## Worked Example

Consider $T = \texttt{banana\$}$ (length 7). The suffix tree has 7 leaves (one for each suffix) and internal nodes corresponding to repeated substrings.

The suffixes are:

| Leaf | Suffix |
|------|--------|
| 0 | `banana$` |
| 1 | `anana$` |
| 2 | `nana$` |
| 3 | `ana$` |
| 4 | `na$` |
| 5 | `a$` |
| 6 | `$` |

The suffix tree structure (edges shown as substring labels):

```
Root
├── "$" → Leaf 6
├── "a" → Node A
│   ├── "$" → Leaf 5
│   └── "na" → Node B
│       ├── "$" → Leaf 3
│       └── "na$" → Leaf 1
├── "banana$" → Leaf 0
└── "na" → Node C
    ├── "$" → Leaf 4
    └── "na$" → Leaf 2
```

Internal nodes and their path labels:

- **Node A**: path = `a` (the substring `a` repeats at positions 1, 3, 5)
- **Node B**: path = `ana` (the substring `ana` repeats at positions 1, 3)
- **Node C**: path = `na` (the substring `na` repeats at positions 2, 4)

## Size and Space Complexity

The suffix tree of a string of length $n+1$ (with sentinel) has:

- Exactly $n + 1$ **leaves**
- At most $n$ **internal nodes** (since every internal node has at least 2 children, and a tree with $n+1$ leaves and minimum branching factor 2 has at most $n$ internal nodes)
- At most $2n + 1$ **total nodes**
- At most $2n$ **edges**

Each edge stores two integers $(l, r)$, and each node stores a pointer to its parent and children. The total space is:

$$
S(n) = O(n)
$$

However, the constant factor matters: in practice, a suffix tree uses approximately 10-20 times the space of the text itself, which is why suffix arrays (using about 4 times the text size) are often preferred for large texts.

## Pattern Matching

To search for a pattern $P[0..m-1]$ in $T$, start at the root and follow edges whose labels match the characters of $P$:

1. At each node, find the outgoing edge whose label begins with the next character of $P$
2. Compare the remaining characters of $P$ against the edge label
3. If all $m$ characters match, $P$ occurs in $T$. The leaves in the subtree below the match point give all occurrence positions.
4. If a mismatch occurs, $P$ does not appear in $T$.

**Time complexity**: $O(m)$ for determining whether $P$ occurs, since each character of $P$ is compared exactly once. Reporting all $k$ occurrences takes an additional $O(k)$ time by traversing the subtree.

## Relationship to Suffix Arrays

The suffix tree and suffix array encode the same information in different forms. The leaves of the suffix tree, read left to right, give the suffix array. Conversely, a suffix tree can be constructed from a suffix array and LCP array in $O(n)$ time.

| Feature | Suffix Tree | Suffix Array + LCP |
|---------|------------|-------------------|
| Construction | $O(n)$ | $O(n)$ |
| Pattern search | $O(m)$ | $O(m + \log n)$ |
| Space (practical) | ~20$n$ bytes | ~8$n$ bytes |
| Substring queries | Direct | Via LCP intervals |

The suffix tree is more flexible for complex queries but less memory-efficient. The enhanced suffix array (with LCP) can simulate almost all suffix tree operations.

## Key Properties

1. **Every substring corresponds to a path**: Any substring $T[i..j]$ labels a path starting from the root. This path may end at a node or in the middle of an edge.

2. **Internal nodes are branching points**: An internal node with path label $w$ indicates that $w$ is followed by at least two distinct characters in $T$. The number of leaves in the subtree equals the number of occurrences of $w$ in $T$.

3. **Deepest internal node gives the LRS**: The internal node with the longest path label corresponds to the longest repeated substring of $T$.

4. **Leaf count gives occurrence count**: The number of leaves in the subtree rooted at any node $v$ equals the number of times $\text{path}(v)$ occurs as a substring of $T$.

## Reference

- Weiner, P. (1973). *Linear pattern matching algorithms*. IEEE Symposium on Switching and Automata Theory, pp. 1-11.
- Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press, Chapter 5.
