# Trie Structure

A **trie** (pronounced "try", from re**trie**val) is a tree-shaped data structure for storing a set of strings over a finite alphabet $\Sigma$. Each path from the root to a marked node spells out one stored string, and strings that share a common prefix share the corresponding path in the trie. This prefix-sharing property makes tries extremely efficient for operations like prefix search, autocomplete, and spell-checking.

## Definition

A trie for an alphabet $\Sigma$ is a rooted tree where:

- Each edge is labeled with a character from $\Sigma$
- No two edges leaving the same node share the same label
- Certain nodes are marked as **endpoints**, indicating that the path from the root to that node spells a complete word

For a set of $n$ strings with total length $L$ over an alphabet of size $|\Sigma|$, the trie has at most $L + 1$ nodes (one per character plus the root).

## Node Representation

The most common implementation stores each node as a dictionary (hash map) mapping characters to child nodes, plus a boolean flag indicating whether the node marks the end of a stored word.

```python
"""Basic trie structure with insertion and search.

Each node stores a dictionary of children and an end-of-word flag.
"""


# === Trie Node and Trie ===
class TrieNode:
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.end = False     # True if this node marks a complete word


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Insert a word into the trie."""
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.end = True

    def search(self, word):
        """Return True if the exact word exists in the trie."""
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.end


# === Main ===
if __name__ == "__main__":
    t = Trie()
    for w in ["apple", "app", "bat"]:
        t.insert(w)
    for w in ["app", "ap", "bat", "bad"]:
        print(f"{w}: {t.search(w)}")
```

**Output:**
```
app: True
ap: False
bat: True
bad: False
```

## Space Complexity

The worst-case space for a trie depends on the representation:

| Representation | Space per node | Total space |
|:---|:---|:---|
| Array of size $\lvert\Sigma\rvert$ | $O(\lvert\Sigma\rvert)$ | $O(L \cdot \lvert\Sigma\rvert)$ |
| Hash map | $O(\text{children count})$ | $O(L)$ average |

The hash-map representation (used above) is more space-efficient when the alphabet is large or when most nodes have few children.

!!! note "Trie vs Hash Table"
    A hash table supports $O(1)$ average lookup for exact strings, but a trie supports $O(p)$ prefix search (where $p$ is the prefix length) -- something a hash table cannot do without scanning all keys. Tries are the data structure of choice when prefix-based queries are common.

## References

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
