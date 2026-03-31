# Trie Insertion and Search

The two fundamental operations on a trie are **insertion** (adding a word to the data structure) and **search** (checking whether a word is present). Both operations traverse the trie character by character, following or creating edges as needed. Their time complexity is $O(m)$ where $m$ is the length of the word -- independent of how many words the trie contains.

## Insertion

To insert a word, start at the root and walk one character at a time. If the next character already has a child node, follow it. If not, create a new node. After processing the last character, mark the final node as a word endpoint.

## Search

To search for a word, follow the same path from the root. If at any point the next character has no corresponding child, the word is not in the trie. If the path exists but the final node is not marked as a word endpoint, the word is also absent (it may be a proper prefix of another word).

## Implementation

```python
"""Trie insertion and search operations.

Demonstrates the two core operations on a trie data structure,
including the distinction between prefix existence and word existence.
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
        """Insert a word into the trie. Time: O(m), Space: O(m)."""
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.end = True

    def search(self, word):
        """Return True if the exact word is in the trie."""
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

Note that `"ap"` returns `False` even though the path `a -> p` exists in the trie. The prefix is present, but it was never inserted as a complete word (its node's `end` flag is `False`).

## Complexity

| Operation | Time | Space |
|:---|:---:|:---:|
| Insert | $O(m)$ | $O(m)$ worst case (new nodes) |
| Search | $O(m)$ | $O(1)$ |

Here $m$ is the length of the word. Both operations are independent of the total number of words $n$ stored in the trie.

## References

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
