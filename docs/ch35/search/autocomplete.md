# Autocomplete with Tries

Autocomplete is one of the most visible applications of the trie data structure. When a user types a prefix into a search box, the system must rapidly retrieve all stored strings (or the top-ranked ones) that begin with that prefix. A trie handles this naturally: navigate to the node corresponding to the prefix in $O(p)$ time, then collect all descendants. This section explains the algorithm, analyzes its complexity, and provides a complete implementation.

## Algorithm

Autocomplete via trie proceeds in three steps:

1. **Navigate to the prefix node**: Starting from the root, follow the path for each character of the prefix. If the path breaks, return an empty result.
2. **Collect all completions**: From the prefix node, perform a DFS (or BFS) to enumerate every path that leads to a word-endpoint node.
3. **Rank results** (optional): Sort completions by frequency, recency, or relevance score stored at each endpoint node.

## Implementation

```python
"""Autocomplete system using a trie data structure.

Stores words with associated frequency counts and returns
completions ranked by frequency.
"""


# === Trie Node and Autocomplete Trie ===
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False
        self.freq = 0


class AutocompleteTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, freq=1):
        """Insert a word with an associated frequency."""
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.end = True
        node.freq += freq

    def autocomplete(self, prefix, limit=5):
        """Return up to `limit` completions for the given prefix, sorted by frequency."""
        node = self.root
        for c in prefix:
            if c not in node.children:
                return []
            node = node.children[c]
        results = []
        self._collect(node, prefix, results)
        results.sort(key=lambda x: -x[1])
        return [word for word, freq in results[:limit]]

    def _collect(self, node, path, results):
        if node.end:
            results.append((path, node.freq))
        for c, child in node.children.items():
            self._collect(child, path + c, results)


# === Main ===
if __name__ == "__main__":
    trie = AutocompleteTrie()
    for word, freq in [("apple", 50), ("app", 30), ("application", 20),
                       ("bat", 10), ("ball", 15), ("banana", 25)]:
        trie.insert(word, freq)

    print("Prefix 'app':", trie.autocomplete("app"))
    print("Prefix 'ba':", trie.autocomplete("ba"))
    print("Prefix 'xyz':", trie.autocomplete("xyz"))
```

**Output:**
```
Prefix 'app': ['apple', 'app', 'application']
Prefix 'ba': ['banana', 'ball', 'bat']
Prefix 'xyz': []
```

## Complexity

| Phase | Time |
|:---|:---:|
| Navigate to prefix | $O(p)$ |
| Collect all completions | $O(k)$ |
| Sort by frequency | $O(k \log k)$ |
| **Total** | $O(p + k \log k)$ |

Here $p$ is the prefix length and $k$ is the total size of all matching words. In practice, $k$ is bounded by the vocabulary size, and the sort can be avoided entirely by maintaining a priority queue or precomputed top-$k$ list at each node.

## References

[Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
