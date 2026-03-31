# Compressed Tries (Patricia Tries)

A standard trie can waste significant space when long strings share no common prefixes, because each character occupies its own node. A **compressed trie** (also called a **Patricia trie** or **radix tree**) eliminates this waste by merging chains of single-child nodes into a single edge labeled with the entire character sequence. This reduces the number of nodes from $O(\text{total characters})$ to $O(n)$, where $n$ is the number of stored strings.

## Motivation

In a standard trie storing the words `{"romane", "romanus", "romulus", "rubens"}`, the path from the root through `r-o-m` consists of nodes with only one child each. A compressed trie collapses such chains into single edges, storing the substring on the edge label rather than spreading it across multiple nodes.

## Structure

Each edge in a compressed trie is labeled with a **string** (not just a single character). Internal nodes exist only at branching points -- positions where two or more stored strings diverge. This guarantees that every internal node has at least two children.

**Key property**: A compressed trie storing $n$ strings has at most $2n - 1$ nodes (at most $n$ leaves and $n - 1$ internal nodes).

## Operations

Insertion, search, and deletion work similarly to a standard trie but must handle string-labeled edges:

- **Search**: At each node, find the outgoing edge whose label is a prefix of the remaining query. If no such edge exists, the query is not in the trie.
- **Insert**: Follow the query as far as possible. If the query diverges mid-edge, **split** the edge at the divergence point by introducing a new internal node.
- **Delete**: Remove the leaf node. If its parent becomes a single-child internal node, merge the parent with its remaining child.

## Complexity

| Operation | Time | Space |
|:---|:---:|:---:|
| Search | $O(m)$ | -- |
| Insert | $O(m)$ | $O(m)$ |
| Delete | $O(m)$ | -- |

Here $m$ is the length of the query string. The overall space for $n$ strings of total length $L$ is $O(n + L)$ when edge labels are stored as substring references, compared to $O(L \cdot |\Sigma|)$ for a standard trie.

!!! tip "When to Use Compressed Tries"
    Compressed tries are most beneficial when the stored strings are long and share moderate prefixes. They are widely used in IP routing tables, dictionaries, and file system path lookups where memory efficiency matters.

## References

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
