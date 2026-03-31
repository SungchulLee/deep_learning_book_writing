# Trie Complexity Analysis

Understanding the time and space complexity of trie operations is essential for deciding when a trie is the right data structure for a problem. This page provides a systematic analysis of each operation, compares tries with alternative data structures, and identifies the scenarios where tries offer a clear advantage.

## Operation Complexities

Let $m$ denote the length of the query string, $n$ the number of stored strings, $L$ the total length of all stored strings, and $|\Sigma|$ the alphabet size.

| Operation | Time | Notes |
|:---|:---:|:---|
| Insert | $O(m)$ | Create at most $m$ new nodes |
| Search (exact) | $O(m)$ | Follow at most $m$ edges |
| Delete | $O(m)$ | May require cleanup of empty nodes |
| Prefix search | $O(p + k)$ | $p$ = prefix length, $k$ = output size |
| Longest prefix match | $O(m)$ | Used in IP routing |

All operations are **independent of $n$** -- a critical advantage when the dataset is large but individual strings are short.

## Space Complexity

The space depends on the node representation:

| Representation | Space per node | Total space |
|:---|:---:|:---|
| Array of size $\lvert\Sigma\rvert$ | $O(\lvert\Sigma\rvert)$ | $O(L \cdot \lvert\Sigma\rvert)$ |
| Hash map | $O(\text{avg children})$ | $O(L)$ on average |
| Compressed (Patricia) | $O(1)$ amortized | $O(n)$ nodes, $O(L)$ for labels |

For small alphabets (e.g., DNA with $|\Sigma| = 4$), the array representation is practical. For large alphabets (e.g., Unicode), hash maps or compressed tries are preferred.

## Comparison with Other Data Structures

| Data Structure | Exact Lookup | Prefix Search | Sorted Order | Space |
|:---|:---:|:---:|:---:|:---|
| Trie | $O(m)$ | $O(p + k)$ | Yes (via DFS) | $O(L \cdot \lvert\Sigma\rvert)$ or $O(L)$ |
| Hash table | $O(m)$ avg | $O(n \cdot m)$ | No | $O(L)$ |
| Balanced BST | $O(m \log n)$ | $O(m \log n + k)$ | Yes | $O(L)$ |
| Sorted array | $O(m \log n)$ | $O(m \log n + k)$ | Yes | $O(L)$ |

The trie's distinguishing advantage is **prefix search in $O(p + k)$** -- no other standard data structure matches this without additional indexing.

!!! tip "When to Choose a Trie"
    Tries are the best choice when:

    - **Prefix-based queries** are frequent (autocomplete, spell-check, IP routing)
    - **Strings share long common prefixes**, making the trie compact
    - **Worst-case guarantees** matter: trie operations have no adversarial inputs (unlike hash tables)

    Tries are a poor choice when the alphabet is very large, strings share few prefixes, and only exact lookup is needed -- a hash table will be simpler and faster.

## References

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
