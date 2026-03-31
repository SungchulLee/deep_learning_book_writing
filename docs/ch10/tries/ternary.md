# Ternary Search Trees

A **ternary search tree** (TST) combines the time efficiency of a trie with the space efficiency of a binary search tree. Each node in a TST stores a single character and has three children: **left** (characters less than the node's character), **middle** (characters equal, advancing to the next position), and **right** (characters greater). This structure avoids the large per-node arrays of a standard trie while still supporting fast string operations.

## Motivation

A standard trie with an array-based representation allocates $|\Sigma|$ child pointers per node, most of which are often `null`. For large alphabets (e.g., Unicode), this wastes enormous space. A TST replaces each node's $|\Sigma|$-way branch with a BST over the characters that actually appear, requiring only three pointers per node regardless of alphabet size.

## Structure

Each TST node stores:

- A **character** $c$
- An **end-of-word** flag
- Three child pointers: **left**, **middle**, **right**

To search for a string, compare the current query character with the node's character. If less, go left; if greater, go right; if equal, go middle and advance to the next query character.

## Complexity

| Operation | Average Time | Worst-Case Time |
|:---|:---:|:---:|
| Search | $O(m + \log n)$ | $O(m \cdot n)$ |
| Insert | $O(m + \log n)$ | $O(m \cdot n)$ |
| Prefix search | $O(p + \log n + k)$ | -- |

Here $m$ is the string length, $n$ is the number of stored strings, $p$ is the prefix length, and $k$ is the output size. The $\log n$ term comes from the BST structure at each level; with balanced insertion order, it stays logarithmic.

## Comparison with Other Trie Variants

| Property | Standard Trie | Compressed Trie | TST |
|:---|:---:|:---:|:---:|
| Space per node | $O(\lvert\Sigma\rvert)$ | $O(\lvert\Sigma\rvert)$ | $O(1)$ |
| Search time | $O(m)$ | $O(m)$ | $O(m + \log n)$ |
| Prefix search | Excellent | Excellent | Good |
| Implementation | Simple | Moderate | Moderate |

!!! tip "When to Use a TST"
    Ternary search trees are a good choice when the alphabet is large, memory is constrained, and prefix-based operations are still needed. They are commonly used in spell checkers and IP routing engines.

## References

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
