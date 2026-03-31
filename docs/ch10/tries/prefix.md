# Prefix Search

One of the most powerful features of a trie is its ability to efficiently find all strings that share a common prefix. Unlike hash tables -- which can check membership in $O(1)$ but cannot enumerate strings by prefix without scanning every entry -- a trie locates the prefix node in $O(p)$ time (where $p$ is the prefix length) and then collects all descendants, making prefix-based queries natural and efficient.

## Algorithm

Prefix search proceeds in two phases:

1. **Navigate to the prefix node**: Starting from the root, follow the path corresponding to the prefix characters. If the path does not exist, no strings in the trie have that prefix.
2. **Collect all descendants**: From the prefix node, traverse all paths to leaf nodes (e.g., using DFS), collecting every complete word found.

## Implementation

```python
"""Trie with prefix search (autocomplete-style enumeration).

Demonstrates how to find all stored words sharing a given prefix.
"""


# === Trie Node and Trie ===
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.end = True

    def _find_node(self, prefix):
        """Navigate to the node at the end of the prefix path."""
        node = self.root
        for c in prefix:
            if c not in node.children:
                return None
            node = node.children[c]
        return node

    def starts_with(self, prefix):
        """Return all words in the trie that start with the given prefix."""
        node = self._find_node(prefix)
        if node is None:
            return []
        results = []
        self._collect(node, prefix, results)
        return results

    def _collect(self, node, path, results):
        if node.end:
            results.append(path)
        for c, child in sorted(node.children.items()):
            self._collect(child, path + c, results)


# === Main ===
if __name__ == "__main__":
    t = Trie()
    for w in ["apple", "app", "application", "bat", "ball", "ban"]:
        t.insert(w)
    print("Prefix 'app':", t.starts_with("app"))
    print("Prefix 'ba':", t.starts_with("ba"))
    print("Prefix 'xyz':", t.starts_with("xyz"))
```

**Output:**
```
Prefix 'app': ['app', 'apple', 'application']
Prefix 'ba': ['ball', 'ban', 'bat']
Prefix 'xyz': []
```

## Complexity

| Phase | Time |
|:---|:---:|
| Navigate to prefix node | $O(p)$ |
| Collect all descendants | $O(k)$ |
| **Total** | $O(p + k)$ |

Here $p$ is the length of the prefix and $k$ is the total number of characters in all matching words. This is optimal: every matching word must be visited at least once.

## References

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
